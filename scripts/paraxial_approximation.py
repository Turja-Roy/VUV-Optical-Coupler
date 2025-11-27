"""
Paraxial approximation for lens coupling evaluation.

This module provides fast analytical estimates of coupling efficiency using:
- Thin lens approximation
- ABCD matrix formalism for Gaussian beam propagation
- Paraxial ray tracing

Use as a theoretical baseline and screening tool before full ray tracing.
"""

import numpy as np
import math
from typing import Dict, Tuple, Optional
from scripts import consts as C
from scripts.calcs import fused_silica_n, medium_refractive_index


def thin_lens_focal_length(R1, R2, n_glass, n_medium=1.0):
    """
    Calculate focal length using lensmaker's equation (thin lens approximation).
    
    Parameters:
    -----------
    R1 : float
        Radius of curvature of first surface (mm). Positive if convex.
    R2 : float  
        Radius of curvature of second surface (mm). Positive if center on +z side.
        For plano-convex, R2 = infinity (use 0 or very large value).
    n_glass : float
        Refractive index of lens material
    n_medium : float
        Refractive index of surrounding medium
        
    Returns:
    --------
    f : float
        Effective focal length in mm
        
    Note:
    -----
    Lensmaker's equation: 1/f = (n_glass/n_medium - 1) * (1/R1 - 1/R2)
    """
    if R2 == 0 or abs(R2) > 1e6:  # Plano-convex case
        R2 = np.inf
    
    if np.isinf(R1):
        term = -1.0 / R2
    elif np.isinf(R2):
        term = 1.0 / R1
    else:
        term = 1.0 / R1 - 1.0 / R2
    
    f_inv = (n_glass / n_medium - 1.0) * term
    
    if abs(f_inv) < 1e-9:
        return np.inf
    
    return 1.0 / f_inv


def thick_lens_cardinal_points(R1, R2, tc, n_glass, n_medium=1.0):
    """
    Calculate cardinal points for a thick lens (more accurate than thin lens).
    
    Parameters:
    -----------
    R1, R2 : float
        Radii of curvature (mm)
    tc : float
        Center thickness (mm)
    n_glass : float
        Refractive index of lens
    n_medium : float
        Refractive index of medium
        
    Returns:
    --------
    dict with keys:
        'f' : effective focal length (mm)
        'f_front' : front focal length (mm, from front vertex to front focal point)
        'f_back' : back focal length (mm, from back vertex to back focal point)
        'H1' : front principal plane position (mm from front vertex, positive = right)
        'H2' : back principal plane position (mm from back vertex, negative = left)
        'P1' : front principal point z-position (absolute)
        'P2' : back principal point z-position (absolute)
    """
    # Handle plano-convex case
    if R2 == 0 or abs(R2) > 1e6:
        R2 = np.inf
    if R1 == 0 or abs(R1) > 1e6:
        R1 = np.inf
    
    n = n_glass / n_medium
    
    # Power of each surface
    P1 = (n - 1) / R1 if not np.isinf(R1) else 0
    P2 = (1 - n) / R2 if not np.isinf(R2) else 0
    
    # Effective focal length
    P = P1 + P2 - (tc / n_glass) * P1 * P2
    
    if abs(P) < 1e-9:
        f = np.inf
    else:
        f = n_medium / P
    
    # Back focal length (from back vertex to back focal point)
    if abs(P) < 1e-9:
        BFL = np.inf
    else:
        BFL = n_medium * (1 - tc * P1 / n_glass) / P
    
    # Front focal length (from front vertex to front focal point)
    if abs(P) < 1e-9:
        FFL = np.inf
    else:
        FFL = -n_medium * (1 - tc * P2 / n_glass) / P
    
    # Principal plane positions (from vertices)
    H1 = -f * tc * P2 / (n_glass * P) if abs(P) > 1e-9 else 0  # From front vertex
    H2 = f * tc * P1 / (n_glass * P) if abs(P) > 1e-9 else 0   # From back vertex
    
    return {
        'f': f,
        'f_front': FFL,
        'f_back': BFL,
        'H1': H1,
        'H2': H2,
        'P1': H1,  # Position relative to front vertex
        'P2': tc + H2  # Position relative to front vertex
    }


def abcd_propagation(d, n=1.0):
    """ABCD matrix for propagation through distance d."""
    return np.array([[1, d/n], [0, 1]])


def abcd_thin_lens(f):
    """ABCD matrix for thin lens with focal length f."""
    return np.array([[1, 0], [-1/f, 1]])


def abcd_thick_lens(R1, R2, tc, n_glass, n_medium=1.0):
    """
    ABCD matrix for thick lens using principal planes.
    
    Returns matrix that transforms from front vertex to back vertex.
    """
    cardinal = thick_lens_cardinal_points(R1, R2, tc, n_glass, n_medium)
    f = cardinal['f']
    H1 = cardinal['H1']
    H2 = cardinal['H2']
    
    # Propagate to front principal plane, apply thin lens, propagate to back vertex
    M = abcd_propagation(tc - H1 - H2, n_medium)
    M = M @ abcd_thin_lens(f)
    
    return M


def gaussian_beam_radius(z, w0, wavelength, n=1.0):
    """
    Calculate Gaussian beam radius at distance z from waist.
    
    Parameters:
    -----------
    z : float
        Distance from beam waist (mm)
    w0 : float
        Beam waist radius (mm)
    wavelength : float
        Wavelength (nm)
    n : float
        Refractive index
        
    Returns:
    --------
    w : float
        Beam radius at distance z (mm)
    """
    # Convert wavelength to mm
    lambda_mm = wavelength * 1e-6
    
    # Rayleigh range
    z_R = math.pi * w0**2 * n / lambda_mm
    
    # Beam radius
    w = w0 * math.sqrt(1 + (z / z_R)**2)
    
    return w


def evaluate_paraxial_coupling(lens1_data: Dict, lens2_data: Dict,
                               z_l1: float, z_l2: float, z_fiber: float,
                               wavelength_nm: float = C.WAVELENGTH_NM,
                               medium: str = 'air',
                               source_radius: float = C.SOURCE_ARC_DIAM_MM / 2.0,
                               source_angle_deg: float = C.MAX_ANGLE_DEG,
                               fiber_radius: float = C.FIBER_CORE_DIAM_MM / 2.0,
                               fiber_na: float = C.NA) -> Dict:
    """
    Evaluate coupling efficiency using paraxial approximation.
    
    Uses ABCD matrix method to propagate beam through two-lens system.
    
    Parameters:
    -----------
    lens1_data, lens2_data : dict
        Lens specifications from database with keys:
        'focal_length_mm', 'radius_r1_mm', 'radius_r2_mm', 
        'center_thickness_mm', 'diameter_mm'
    z_l1, z_l2, z_fiber : float
        Axial positions (mm)
    wavelength_nm : float
        Wavelength in nm
    medium : str
        Propagation medium ('air', 'argon', 'helium')
    source_radius : float
        Source radius (mm)
    source_angle_deg : float
        Maximum source divergence angle (degrees)
    fiber_radius : float
        Fiber core radius (mm)
    fiber_na : float
        Fiber numerical aperture
        
    Returns:
    --------
    dict with keys:
        'coupling_paraxial' : estimated coupling efficiency (0-1)
        'spot_radius' : spot radius at fiber (mm)
        'magnification' : transverse magnification
        'output_angle' : output cone half-angle (degrees)
        'position_match' : how well spot is centered (0-1, 1=perfect)
        'size_match' : how well spot size matches fiber (0-1, 1=perfect)
        'angle_match' : how well output angle matches fiber NA (0-1, 1=perfect)
        'f1_eff' : effective focal length of lens 1 (mm)
        'f2_eff' : effective focal length of lens 2 (mm)
        'total_length' : total system length (mm)
    """
    # Get refractive indices
    n_medium = medium_refractive_index(wavelength_nm, medium)
    n_glass = fused_silica_n(wavelength_nm)
    
    # Extract lens parameters
    R1_L1 = lens1_data['radius_r1_mm']
    R2_L1 = lens1_data.get('radius_r2_mm', 0) or 0  # 0 for plano-convex
    tc1 = lens1_data['center_thickness_mm']
    
    R1_L2 = lens2_data['radius_r1_mm']
    R2_L2 = lens2_data.get('radius_r2_mm', 0) or 0
    tc2 = lens2_data['center_thickness_mm']
    
    # Calculate thick lens parameters for both lenses
    lens1_cardinal = thick_lens_cardinal_points(R1_L1, R2_L1, tc1, n_glass, n_medium)
    lens2_cardinal = thick_lens_cardinal_points(R1_L2, R2_L2, tc2, n_glass, n_medium)
    
    f1 = lens1_cardinal['f']
    f2 = lens2_cardinal['f']
    
    # Principal plane positions (relative to front vertex)
    H1_L1 = lens1_cardinal['H1']  # Front principal plane of L1
    H2_L1 = lens1_cardinal['H2']  # Back principal plane of L1
    P1_L1 = z_l1 + H1_L1  # Absolute position of front principal plane
    P2_L1 = z_l1 + tc1 + H2_L1  # Absolute position of back principal plane
    
    H1_L2 = lens2_cardinal['H1']
    H2_L2 = lens2_cardinal['H2']
    P1_L2 = z_l2 + H1_L2
    P2_L2 = z_l2 + tc2 + H2_L2
    
    # Distances between principal planes
    # NOTE: Source is at z=0 (arc position). Cooling jacket window is at z=26mm,
    # so lenses must be placed at z >= 27mm. The ray path is:
    # Source (z=0) -> travels through cooling jacket -> L1 (z=z_l1) -> L2 -> fiber
    d1 = P1_L1  # Source to first principal plane of L1 (source at z=0)
    d12 = P1_L2 - P2_L1  # Back principal plane of L1 to front principal plane of L2
    d2 = z_fiber - P2_L2  # Back principal plane of L2 to fiber
    
    # === Method 1: ABCD Matrix for Gaussian Beam ===
    # Model source as Gaussian beam with waist at source
    w0_source = source_radius / 2.0  # Approximate 1/e² radius
    
    # Build ABCD matrix for entire system
    M = np.eye(2)
    M = M @ abcd_propagation(d1, n_medium)  # Source to L1
    M = M @ abcd_thin_lens(f1)  # L1 (thin lens at principal planes)
    M = M @ abcd_propagation(d12, n_medium)  # L1 to L2
    M = M @ abcd_thin_lens(f2)  # L2
    M = M @ abcd_propagation(d2, n_medium)  # L2 to fiber
    
    A, B = M[0, 0], M[0, 1]
    C_matrix, D = M[1, 0], M[1, 1]
    
    # === Method 2: Marginal Ray Tracing ===
    # Trace marginal ray (at edge of source with maximum angle)
    y0 = source_radius  # Starting height
    u0 = math.tan(math.radians(source_angle_deg))  # Starting angle (paraxial)
    
    # At fiber
    y_fiber = A * y0 + B * u0
    u_fiber = C_matrix * y0 + D * u0
    
    # Output angle
    output_angle_rad = math.atan(abs(u_fiber))
    output_angle_deg = math.degrees(output_angle_rad)
    
    # Spot radius at fiber (from marginal ray)
    spot_radius = abs(y_fiber)
    
    # Transverse magnification (for central rays)
    magnification = A
    
    # === Coupling Efficiency Estimation ===
    
    # 1. Position match (is beam centered?)
    # For axial source, beam should be centered
    position_match = 1.0  # Assume centered (can refine if off-axis)
    
    # 2. Size match (spot size vs fiber core)
    # Ideal: spot radius ≈ fiber radius
    # Use Gaussian overlap: exp(-2 * (Δr/w)²)
    if spot_radius > 0:
        size_ratio = spot_radius / fiber_radius
        if size_ratio < 0.1:
            size_ratio = 0.1  # Avoid division by zero
        
        # Optimal when size_ratio ≈ 1
        # Penalty for being too large (loss) or too small (not using full NA)
        size_match = math.exp(-0.5 * (math.log(size_ratio))**2)
    else:
        size_match = 0.0
    
    # 3. Angle match (output angle vs fiber acceptance)
    fiber_acceptance_angle_deg = math.degrees(math.asin(fiber_na / n_medium))
    
    if output_angle_deg > 0:
        angle_ratio = output_angle_deg / fiber_acceptance_angle_deg
        # Penalty if angle exceeds fiber NA (loss), or is much smaller (inefficient)
        if angle_ratio <= 1.0:
            # Underfilled: less penalty
            angle_match = angle_ratio**0.5  # Some penalty for underfilling
        else:
            # Overfilled: strong penalty (rays rejected)
            # Approximate solid angle ratio
            angle_match = (fiber_acceptance_angle_deg / output_angle_deg)**2
    else:
        angle_match = 0.0
    
    # 4. Combined coupling efficiency
    # Geometric coupling ≈ overlap of spot with fiber × angular acceptance
    coupling_paraxial = position_match * size_match * angle_match
    
    # Account for Fresnel losses (approximate, 4% per surface for uncoated)
    # 4 surfaces total (2 lenses × 2 surfaces each)
    # T_surface ≈ (1 - ((n_glass - n_medium)/(n_glass + n_medium))²)
    R_surface = ((n_glass - n_medium) / (n_glass + n_medium))**2
    T_surface = 1 - R_surface
    transmission = T_surface**4
    
    coupling_paraxial *= transmission
    
    # 5. Source solid angle vignetting (from cooling jacket)
    coupling_paraxial *= C.GEOMETRIC_LOSS_FACTOR
    
    # Clamp to [0, 1]
    coupling_paraxial = max(0.0, min(1.0, coupling_paraxial))
    
    return {
        'coupling_paraxial': coupling_paraxial,
        'spot_radius': spot_radius,
        'spot_diameter': 2 * spot_radius,
        'magnification': magnification,
        'output_angle': output_angle_deg,
        'fiber_acceptance_angle': fiber_acceptance_angle_deg,
        'position_match': position_match,
        'size_match': size_match,
        'angle_match': angle_match,
        'f1_eff': f1,
        'f2_eff': f2,
        'f1_bfl': lens1_cardinal['f_back'],
        'f2_bfl': lens2_cardinal['f_back'],
        'transmission': transmission,
        'total_length': z_fiber - C.WINDOW_DISTANCE_MM,
        'spacing_l1_l2': d12 + tc1/2 + tc2/2,  # Center-to-center
        'A': A, 'B': B, 'C': C_matrix, 'D': D  # ABCD matrix elements
    }


def optimize_paraxial_positions_fast(lens1_data: Dict, lens2_data: Dict,
                                     wavelength_nm: float = C.WAVELENGTH_NM,
                                     medium: str = 'air') -> Optional[Dict]:
    """
    Fast estimation using analytical approach (single evaluation).
    
    Uses focal length formulas to estimate good starting positions analytically.
    Much faster than grid search, suitable for screening thousands of combinations.
    
    Parameters:
    -----------
    lens1_data, lens2_data : dict
        Lens specifications
    wavelength_nm : float
        Wavelength (nm)
    medium : str
        Propagation medium
        
    Returns:
    --------
    dict : Estimated best configuration
    """
    # Get refractive indices
    n_medium = medium_refractive_index(wavelength_nm, medium)
    n_glass = fused_silica_n(wavelength_nm)
    
    # Extract lens parameters
    R1_L1 = lens1_data['radius_r1_mm']
    R2_L1 = lens1_data.get('radius_r2_mm', 0) or 0
    tc1 = lens1_data['center_thickness_mm']
    
    R1_L2 = lens2_data['radius_r1_mm']
    R2_L2 = lens2_data.get('radius_r2_mm', 0) or 0
    tc2 = lens2_data['center_thickness_mm']
    
    # Calculate thick lens parameters
    lens1_cardinal = thick_lens_cardinal_points(R1_L1, R2_L1, tc1, n_glass, n_medium)
    lens2_cardinal = thick_lens_cardinal_points(R1_L2, R2_L2, tc2, n_glass, n_medium)
    
    f1 = lens1_cardinal['f']
    f2 = lens2_cardinal['f']
    
    # Analytical positioning based on imaging the source onto the fiber
    # Place first lens at minimum allowed distance from source
    # SOURCE_TO_LENS_OFFSET = 27mm (cooling jacket window exit)
    z_l1 = C.SOURCE_TO_LENS_OFFSET + 5  # Start after window with minimal clearance
    
    # Object distance for L1 (from front principal plane)
    H1_L1 = lens1_cardinal['H1']
    s1 = z_l1 + H1_L1  # Object distance from front principal plane
    
    # Image distance from L1 (thin lens equation: 1/f = 1/s + 1/s')
    if abs(f1) > 0.1:
        s1_prime = 1.0 / (1.0/f1 - 1.0/s1) if abs(1.0/f1 - 1.0/s1) > 1e-6 else 1000
    else:
        s1_prime = 1000  # Very weak lens
    
    # Position of intermediate image (from back principal plane of L1)
    H2_L1 = lens1_cardinal['H2']
    z_image1 = z_l1 + tc1 + H2_L1 + s1_prime
    
    # Place L2 to relay this image to fiber
    # Object distance for L2 should be a few focal lengths
    s2 = max(f2 * 0.8, 10)  # At least 10mm or 0.8*f2
    z_l2 = z_image1 - s2
    
    # Image distance from L2
    H1_L2 = lens2_cardinal['H1']
    s2_obj = s2 - H1_L2  # Adjust for principal plane
    if abs(f2) > 0.1:
        s2_prime = 1.0 / (1.0/f2 - 1.0/s2_obj) if abs(1.0/f2 - 1.0/s2_obj) > 1e-6 else 1000
    else:
        s2_prime = 1000
    
    # Fiber position
    H2_L2 = lens2_cardinal['H2']
    z_fiber = z_l2 + tc2 + H2_L2 + s2_prime
    
    # Evaluate this configuration
    result = evaluate_paraxial_coupling(
        lens1_data, lens2_data,
        z_l1, z_l2, z_fiber,
        wavelength_nm, medium
    )
    
    result['z_l1'] = z_l1
    result['z_l2'] = z_l2
    result['z_fiber'] = z_fiber
    
    return result


def optimize_paraxial_positions(lens1_data: Dict, lens2_data: Dict,
                                wavelength_nm: float = C.WAVELENGTH_NM,
                                medium: str = 'air',
                                z_l1_range: Optional[Tuple[float, float]] = None,
                                n_samples: int = 8) -> Optional[Dict]:
    """
    Find optimal positions for two lenses using paraxial approximation.
    
    Uses coarse grid search over lens positions. For fast screening, use
    optimize_paraxial_positions_fast() instead.
    
    Parameters:
    -----------
    lens1_data, lens2_data : dict
        Lens specifications
    wavelength_nm : float
        Wavelength (nm)
    medium : str
        Propagation medium
    z_l1_range : tuple, optional
        Range for first lens position (min, max) in mm.
        If None, defaults to (SOURCE_TO_LENS_OFFSET, SOURCE_TO_LENS_OFFSET + 40)
        Note: First lens MUST be after cooling jacket exit (z >= 27mm)
    n_samples : int
        Number of samples per dimension (default: 8 for speed)
        
    Returns:
    --------
    dict : Best configuration found with coupling, positions, etc.
    """
    # Set default range if not provided - must start after cooling jacket window
    if z_l1_range is None:
        z_l1_min = C.SOURCE_TO_LENS_OFFSET  # 27mm - after cooling jacket exit
        z_l1_max = C.SOURCE_TO_LENS_OFFSET + 40  # Allow reasonable range
        z_l1_range = (z_l1_min, z_l1_max)
    
    best_result = None
    best_coupling = 0.0
    
    # Coarse search with fewer samples for speed
    z_l1_values = np.linspace(z_l1_range[0], z_l1_range[1], n_samples)
    
    for z_l1 in z_l1_values:
        # Estimate reasonable range for z_l2 based on focal lengths
        f1_approx = lens1_data['focal_length_mm']
        f2_approx = lens2_data['focal_length_mm']
        
        # Place L2 roughly one focal length away
        z_l2_min = z_l1 + lens1_data['center_thickness_mm'] + f1_approx * 0.5
        z_l2_max = z_l1 + lens1_data['center_thickness_mm'] + f1_approx * 2.0
        
        z_l2_values = np.linspace(z_l2_min, z_l2_max, n_samples)
        
        for z_l2 in z_l2_values:
            # Place fiber roughly one focal length after L2
            z_fiber_min = z_l2 + lens2_data['center_thickness_mm'] + f2_approx * 0.5
            z_fiber_max = z_l2 + lens2_data['center_thickness_mm'] + f2_approx * 2.0
            
            z_fiber_values = np.linspace(z_fiber_min, z_fiber_max, n_samples)
            
            for z_fiber in z_fiber_values:
                result = evaluate_paraxial_coupling(
                    lens1_data, lens2_data,
                    z_l1, z_l2, z_fiber,
                    wavelength_nm, medium
                )
                
                if result['coupling_paraxial'] > best_coupling:
                    best_coupling = result['coupling_paraxial']
                    best_result = result
                    best_result['z_l1'] = z_l1
                    best_result['z_l2'] = z_l2
                    best_result['z_fiber'] = z_fiber
    
    return best_result


def evaluate_all_combinations(lens_db, wavelength_nm: float = C.WAVELENGTH_NM,
                              medium: str = 'air',
                              min_coupling: float = 0.01,
                              progress_callback=None):
    """
    Evaluate all lens combinations from database using paraxial approximation.
    
    Parameters:
    -----------
    lens_db : LensDatabase
        Lens database instance
    wavelength_nm : float
        Wavelength (nm)
    medium : str
        Propagation medium
    min_coupling : float
        Minimum coupling to report (filter out very poor combinations)
    progress_callback : callable
        Optional callback(current, total, lens_pair) for progress updates
        
    Returns:
    --------
    list : Results sorted by coupling (best first)
    """
    from itertools import combinations_with_replacement
    
    # Get all lenses
    all_lenses = lens_db.get_all_lenses()
    
    print(f"\nEvaluating {len(all_lenses)} lenses...")
    print(f"Total combinations: {len(all_lenses) * (len(all_lenses) + 1) // 2}")
    
    results = []
    total_combos = 0
    
    for i, lens1 in enumerate(all_lenses):
        for j, lens2 in enumerate(all_lenses[i:], start=i):
            total_combos += 1
            
            if progress_callback and total_combos % 100 == 0:
                progress_callback(total_combos, 
                                len(all_lenses) * (len(all_lenses) + 1) // 2,
                                f"{lens1['item_number']}+{lens2['item_number']}")
            
            # Coarse grid search (5^3 = 125 evaluations per pair, ~2.5ms per pair)
            best = optimize_paraxial_positions(lens1, lens2, wavelength_nm, medium, n_samples=5)
            
            if best and best['coupling_paraxial'] >= min_coupling:
                results.append({
                    'lens1': lens1['item_number'],
                    'lens2': lens2['item_number'],
                    'coupling': best['coupling_paraxial'],
                    'z_l1': best['z_l1'],
                    'z_l2': best['z_l2'],
                    'z_fiber': best['z_fiber'],
                    'total_length': best['total_length'],
                    'spot_radius': best['spot_radius'],
                    'output_angle': best['output_angle'],
                    'f1_eff': best['f1_eff'],
                    'f2_eff': best['f2_eff'],
                    'size_match': best['size_match'],
                    'angle_match': best['angle_match'],
                    'method': 'paraxial'
                })
    
    # Sort by coupling (best first)
    results.sort(key=lambda x: x['coupling'], reverse=True)
    
    return results
