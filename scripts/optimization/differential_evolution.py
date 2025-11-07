import numpy as np
from scipy.optimize import differential_evolution
from scripts.lens_factory import create_lens
from scripts import consts as C
from scripts.raytrace_helpers import sample_rays
from scripts.raytrace_helpers_vectorized import trace_system_vectorized as trace_system


def evaluate_config_fast(params, d1, d2, origins, dirs, n_rays, alpha=0.7, medium='air', flipped1=False, flipped2=True):
    z_l1, z_l2 = params
    
    min_gap = 0.5
    if z_l1 < C.SOURCE_TO_LENS_OFFSET or z_l2 <= z_l1 + d1['tc_mm'] + min_gap:
        return 1e6
    
    z_fiber = z_l2 + d2['f_mm']
    
    lens1 = create_lens(d1, z_l1, flipped=flipped1)
    lens2 = create_lens(d2, z_l2, flipped=flipped2)
    
    accepted, transmission = trace_system(origins, dirs, lens1, lens2, z_fiber, 
                           C.FIBER_CORE_DIAM_MM/2.0, C.ACCEPTANCE_HALF_RAD, 
                           medium, C.PRESSURE_ATM, C.TEMPERATURE_K, C.HUMIDITY_FRACTION)
    
    avg_transmission = np.mean(transmission[accepted]) if np.any(accepted) else 0.0
    coupling = (np.count_nonzero(accepted) / n_rays) * avg_transmission
    
    return alpha * (1 - coupling) + (1 - alpha) * z_fiber / 80.0


def get_bounds(f1, f2):
    z_l1_max = max(C.SOURCE_TO_LENS_OFFSET + 5.0, f1 * 1.5)
    z_l2_max = max(z_l1_max + 5.0, f1 * 1.5 + f2 * 2.5)
    return [
        (C.SOURCE_TO_LENS_OFFSET, z_l1_max),
        (C.SOURCE_TO_LENS_OFFSET + 1.0, z_l2_max)
    ]


def optimize(lenses, name1, name2, n_rays=1000, alpha=0.7, medium='air', orientation_mode='both'):
    d1, d2 = lenses[name1], lenses[name2]
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
        
        z_l1, z_l2 = result.x
        z_fiber = z_l2 + d2['f_mm']
        
        origins_final, dirs_final = sample_rays(2000)
        lens1 = create_lens(d1, z_l1, flipped=flipped1)
        lens2 = create_lens(d2, z_l2, flipped=flipped2)
        
        accepted, transmission = trace_system(origins_final, dirs_final, lens1, lens2,
                               z_fiber, C.FIBER_CORE_DIAM_MM/2.0, C.ACCEPTANCE_HALF_RAD,
                               medium, C.PRESSURE_ATM, C.TEMPERATURE_K, C.HUMIDITY_FRACTION)
        
        avg_transmission = np.mean(transmission[accepted]) if np.any(accepted) else 0.0
        coupling = (np.count_nonzero(accepted) / 2000) * avg_transmission
        
        results.append({
            'lens1': name1, 'lens2': name2,
            'f1_mm': d1['f_mm'], 'f2_mm': d2['f_mm'],
            'z_l1': z_l1, 'z_l2': z_l2, 'z_fiber': z_fiber,
            'total_len_mm': z_fiber,
            'coupling': coupling,
            'orientation': orientation_name,
            'origins': origins_final, 'dirs': dirs_final, 'accepted': accepted
        })
    
    # Return list if 'both' mode, otherwise return single result
    if orientation_mode == 'both':
        return results
    else:
        return results[0] if results else None
