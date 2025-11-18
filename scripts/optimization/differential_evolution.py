import numpy as np
from scipy.optimize import differential_evolution
from scripts.lens_factory import create_lens
from scripts import consts as C
from scripts.raytrace_helpers import sample_rays
from scripts.raytrace_helpers_vectorized import trace_system_vectorized as trace_system
from scripts.optimization.fiber_position_optimizer import evaluate_both_fiber_positions
import logging

logger = logging.getLogger(__name__)


def calc_sag(R, ap_rad):
    """Calculate sag = R - sqrt(R² - ap_rad²) for lens surface curvature"""
    R_abs = abs(R)
    if ap_rad >= R_abs:
        return 0.0
    return R_abs - np.sqrt(R_abs**2 - ap_rad**2)


def evaluate_config_fast(params, d1, d2, origins, dirs, n_rays, alpha=0.7, medium='air', flipped1=False, flipped2=True):
    z_l1, z_l2 = params
    
    # Calculate physical extents accounting for surface sag
    ap_rad1 = d1['dia'] / 2.0
    ap_rad2 = d2['dia'] / 2.0
    
    # Lens 1 back extent (accounting for back surface curvature)
    l1_end = z_l1 + d1['tc_mm']
    if 'R2_mm' in d1 and d1.get('lens_type') in ['Bi-Convex', 'Aspheric']:
        sag1_back = calc_sag(d1['R2_mm'], ap_rad1)
        l1_end += sag1_back
    
    # Lens 2 front extent (accounting for front surface curvature)
    l2_start = z_l2
    if 'R1_mm' in d2 and d2.get('lens_type') in ['Bi-Convex', 'Aspheric']:
        sag2_front = calc_sag(d2['R1_mm'], ap_rad2)
        l2_start -= sag2_front
    
    min_gap = 0.5
    if z_l1 < C.SOURCE_TO_LENS_OFFSET or l2_start <= l1_end + min_gap:
        return 1e6
    
    z_fiber = z_l2 + d2['f_mm']
    
    lens1 = create_lens(d1, z_l1, flipped=flipped1)
    lens2 = create_lens(d2, z_l2, flipped=flipped2)
    
    accepted, transmission = trace_system(origins, dirs, lens1, lens2, z_fiber,
                           C.FIBER_CORE_DIAM_MM/2.0, C.ACCEPTANCE_HALF_RAD,
                           medium, C.PRESSURE_ATM, C.TEMPERATURE_K, C.HUMIDITY_FRACTION)
    avg_transmission = np.mean(transmission[accepted]) if np.any(accepted) else 0.0
    coupling = (np.count_nonzero(accepted) / n_rays) * avg_transmission * C.GEOMETRIC_LOSS_FACTOR
    
    return alpha * (1 - coupling) + (1 - alpha) * z_fiber / 80.0


def get_bounds(f1, f2):
    z_l1_max = max(C.SOURCE_TO_LENS_OFFSET + 5.0, f1 * 1.5)
    z_l2_max = max(z_l1_max + 5.0, f1 * 1.5 + f2 * 2.5)
    return [
        (C.SOURCE_TO_LENS_OFFSET, z_l1_max),
        (C.SOURCE_TO_LENS_OFFSET + 1.0, z_l2_max)
    ]


def optimize(lenses, name1, name2, n_rays=1000, alpha=0.7, medium='air', orientation_mode='both', seed=None):
    d1, d2 = lenses[name1], lenses[name2]
    
    # Set seed if provided for reproducibility
    if seed is not None:
        np.random.seed(seed)
    
    origins, dirs = sample_rays(n_rays)
    
    bounds = get_bounds(d1['f_mm'], d2['f_mm'])
    
    # Test both orientations
    results = []
    orientations = [
        (False, True, 'ScffcF'),   # lens1 curved-first, lens2 flat-first
        (True, False, 'SfccfF')    # lens1 flat-first, lens2 curved-first
    ]
    
    # Filter orientations based on mode
    if orientation_mode != 'both':
        orientations = [o for o in orientations if o[2] == orientation_mode]
    
    for flipped1, flipped2, orientation_name in orientations:
        result = differential_evolution(
            evaluate_config_fast, bounds,
            args=(d1, d2, origins, dirs, n_rays, alpha, medium, flipped1, flipped2),
            maxiter=50, popsize=10, workers=1
        )
        
        # Check if optimization returned a valid result (not the penalty value)
        if result.fun >= 1e5:  # If objective >= 100k, it's the penalty
            logger.error(f"OPTIMIZATION FAILED: Returned penalty value {result.fun:.0f}")
            logger.error(f"  No valid configuration found for {name1}+{name2} with orientation {orientation_name}")
            logger.error(f"  Skipping this orientation...")
            continue  # Skip this orientation
        
        z_l1, z_l2 = result.x
        
        # Double-check: Validate final result doesn't have overlap
        ap_rad1 = d1['dia'] / 2.0
        ap_rad2 = d2['dia'] / 2.0
        
        l1_end = z_l1 + d1['tc_mm']
        if 'R2_mm' in d1 and d1.get('lens_type') in ['Bi-Convex', 'Aspheric']:
            sag1_back = calc_sag(d1['R2_mm'], ap_rad1)
            l1_end += sag1_back
        
        l2_start = z_l2
        if 'R1_mm' in d2 and d2.get('lens_type') in ['Bi-Convex', 'Aspheric']:
            sag2_front = calc_sag(d2['R1_mm'], ap_rad2)
            l2_start -= sag2_front
        
        final_gap = l2_start - l1_end
        
        if final_gap < 0.5:
            logger.error(f"DOUBLE-CHECK FAILED! Gap={final_gap:.3f} < 0.5 mm (this shouldn't happen!)")
            if final_gap < 0:
                logger.error(f"  LENSES OVERLAP BY {abs(final_gap):.3f} mm!")
            logger.error(f"  Skipping this result...")
            continue  # Skip this invalid result
        
        # Use seed for final evaluation to ensure reproducibility
        origins_final, dirs_final = sample_rays(2000, seed=seed)
        lens1 = create_lens(d1, z_l1, flipped=flipped1)
        lens2 = create_lens(d2, z_l2, flipped=flipped2)
        
        # Evaluate both fiber positioning methods and keep the best
        z_fiber, coupling, fiber_method = evaluate_both_fiber_positions(
            lens1, lens2, z_l2, d2['f_mm'], origins_final, dirs_final, medium
        )
        
        # Get accepted rays for the chosen fiber position
        accepted, transmission = trace_system(origins_final, dirs_final, lens1, lens2,
                               z_fiber, C.FIBER_CORE_DIAM_MM/2.0, C.ACCEPTANCE_HALF_RAD,
                               medium, C.PRESSURE_ATM, C.TEMPERATURE_K, C.HUMIDITY_FRACTION)
        
        results.append({
            'lens1': name1, 'lens2': name2,
            'f1_mm': d1['f_mm'], 'f2_mm': d2['f_mm'],
            'z_l1': z_l1, 'z_l2': z_l2, 'z_fiber': z_fiber,
            'total_len_mm': z_fiber,
            'coupling': coupling,
            'orientation': orientation_name,
            'fiber_position_method': fiber_method,
            'origins': origins_final, 'dirs': dirs_final, 'accepted': accepted
        })
    
    # If no valid configurations found, return appropriate empty result
    if not results:
        logger.warning(f"No valid configurations found for {name1}+{name2}")
        return [] if orientation_mode == 'both' else None
    
    # Return list if 'both' mode, otherwise return single result
    if orientation_mode == 'both':
        return results
    else:
        return results[0]
