"""
Vectorized ray tracing functions for high-performance batch ray processing.

This module provides NumPy-vectorized implementations of ray-sphere intersections,
refraction calculations, and lens tracing. Expected speedup: 10-50x over serial version.

All functions operate on batches of rays simultaneously using NumPy broadcasting.
"""

import numpy as np
import math
import warnings
from scripts import consts as C
from scripts.calcs import transmission_through_medium, medium_refractive_index, calculate_attenuation_coefficient

# Suppress expected RuntimeWarnings from NaN operations on failed rays
# This is intentional behavior - failed rays are marked as NaN and filtered by success masks
warnings.filterwarnings('ignore', 'invalid value encountered', RuntimeWarning)


def transmission_through_medium_vectorized(distances_mm, wavelength_nm, medium, 
                                          pressure_atm=1.0, temp_k=293.15, humidity_fraction=0.01):
    """
    Vectorized atmospheric transmission calculation for batch of distances.
    
    Parameters
    ----------
    distances_mm : ndarray, shape (n_rays,)
        Propagation distances in mm
    wavelength_nm : float
        Wavelength in nanometers
    medium : str
        Medium type ('air', 'argon', 'helium')
    pressure_atm : float
        Atmospheric pressure
    temp_k : float
        Temperature in Kelvin
    humidity_fraction : float
        Water vapor fraction (0.0-0.1)
    
    Returns
    -------
    transmission : ndarray, shape (n_rays,)
        Transmission factors (0-1)
    """
    alpha = calculate_attenuation_coefficient(wavelength_nm, medium, pressure_atm, temp_k, humidity_fraction)
    
    if alpha == 0.0:
        return np.ones_like(distances_mm)
    
    return np.exp(-alpha * distances_mm)


def intersect_ray_sphere_vectorized(origins, directions, center, radius):
    """
    Vectorized ray-sphere intersection for batch of rays.
    
    Parameters
    ----------
    origins : ndarray, shape (n_rays, 3)
        Ray origin points
    directions : ndarray, shape (n_rays, 3)
        Ray direction vectors (should be normalized)
    center : ndarray, shape (3,)
        Sphere center point
    radius : float
        Sphere radius
    
    Returns
    -------
    t_values : ndarray, shape (n_rays,)
        Intersection parameter t for each ray (NaN if no intersection)
    hit_mask : ndarray, shape (n_rays,), dtype=bool
        True where rays intersect the sphere
    """
    # Vectorized quadratic formula for ray-sphere intersection
    oc = origins - center  # (n_rays, 3)
    
    # Coefficients for quadratic equation at^2 + bt + c = 0
    a = np.sum(directions * directions, axis=1)  # (n_rays,)
    b = 2.0 * np.sum(oc * directions, axis=1)    # (n_rays,)
    c = np.sum(oc * oc, axis=1) - radius * radius  # (n_rays,)
    
    # Discriminant
    disc = b * b - 4 * a * c  # (n_rays,)
    
    # Initialize output
    t_values = np.full(origins.shape[0], np.nan)
    hit_mask = disc >= 0
    
    if not np.any(hit_mask):
        return t_values, hit_mask
    
    # Calculate t values only for rays that hit
    sqrt_disc = np.sqrt(disc[hit_mask])
    a_hit = a[hit_mask]
    b_hit = b[hit_mask]
    
    t1 = (-b_hit - sqrt_disc) / (2 * a_hit)
    t2 = (-b_hit + sqrt_disc) / (2 * a_hit)
    
    # Choose smallest positive t
    # Set negative values to inf so they don't get selected by minimum
    t1_valid = np.where(t1 > 1e-9, t1, np.inf)
    t2_valid = np.where(t2 > 1e-9, t2, np.inf)
    
    t_selected = np.minimum(t1_valid, t2_valid)
    
    # Update hit mask - only rays with finite t are valid hits
    valid_hits = np.isfinite(t_selected)
    t_values[hit_mask] = t_selected
    hit_mask[hit_mask] = valid_hits
    
    return t_values, hit_mask


def refract_vec_vectorized(normals, v_in, n1, n2):
    """
    Vectorized Snell's law refraction for batch of rays.
    
    Parameters
    ----------
    normals : ndarray, shape (n_rays, 3)
        Surface normal vectors (will be normalized)
    v_in : ndarray, shape (n_rays, 3)
        Incident ray directions (will be normalized)
    n1 : float
        Incident medium refractive index
    n2 : float
        Transmitted medium refractive index
    
    Returns
    -------
    v_out : ndarray, shape (n_rays, 3)
        Refracted ray directions (NaN for total internal reflection)
    success : ndarray, shape (n_rays,), dtype=bool
        True where refraction succeeded (no TIR)
    """
    # Normalize inputs
    n_vec = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    v = v_in / np.linalg.norm(v_in, axis=1, keepdims=True)
    
    # Snell's law in vector form
    cos_i = -np.sum(n_vec * v, axis=1)  # (n_rays,)
    eta = n1 / n2
    k = 1.0 - eta * eta * (1.0 - cos_i * cos_i)  # (n_rays,)
    
    # Check for total internal reflection
    success = k >= 0
    
    # Calculate refracted rays (set TIR rays to NaN)
    v_out = np.full_like(v_in, np.nan)
    
    if np.any(success):
        cos_i_success = cos_i[success]
        k_success = k[success]
        v_success = v[success]
        n_success = n_vec[success]
        
        sqrt_k = np.sqrt(k_success)
        factor1 = eta
        factor2 = eta * cos_i_success[:, np.newaxis] - sqrt_k[:, np.newaxis]
        
        v_out[success] = factor1 * v_success + factor2 * n_success
        
        # Normalize output
        v_out[success] = v_out[success] / np.linalg.norm(v_out[success], axis=1, keepdims=True)
    
    return v_out, success


def trace_planoconvex_vectorized(origins, directions, n_medium, lens_params, flipped=False):
    """
    Vectorized ray tracing through a plano-convex lens.
    
    Parameters
    ----------
    origins : ndarray, shape (n_rays, 3)
        Ray starting points
    directions : ndarray, shape (n_rays, 3)
        Ray direction vectors
    n_medium : float
        Refractive index of surrounding medium
    lens_params : dict
        Lens parameters with keys:
        - 'vertex_z_front': front surface z position
        - 'R_front_mm': radius of curvature
        - 'center_thickness_mm': thickness at center
        - 'edge_thickness_mm': thickness at edge
        - 'ap_rad_mm': aperture radius
        - 'n_glass': glass refractive index
    flipped : bool
        If True, lens is flipped (flat face first)
    
    Returns
    -------
    origins_out : ndarray, shape (n_rays, 3)
        Exit points from lens (NaN for failed rays)
    directions_out : ndarray, shape (n_rays, 3)
        Exit directions from lens (NaN for failed rays)
    success : ndarray, shape (n_rays,), dtype=bool
        True where rays successfully traced through lens
    """
    n_rays = origins.shape[0]
    n_glass = lens_params['n_glass']
    ap_rad = lens_params['ap_rad_mm']
    
    if not flipped:
        return _trace_curved_flat_vectorized(origins, directions, n_medium, lens_params)
    else:
        return _trace_flat_curved_vectorized(origins, directions, n_medium, lens_params)


def _trace_curved_flat_vectorized(origins, directions, n_medium, lens_params):
    """Vectorized trace through curved-then-flat lens."""
    n_rays = origins.shape[0]
    n_glass = lens_params['n_glass']
    vertex_z_front = lens_params['vertex_z_front']
    R_front = lens_params['R_front_mm']
    center_thickness = lens_params['center_thickness_mm']
    edge_thickness = lens_params['edge_thickness_mm']
    ap_rad = lens_params['ap_rad_mm']
    
    # Initialize output arrays
    origins_out = np.full((n_rays, 3), np.nan)
    directions_out = np.full((n_rays, 3), np.nan)
    success = np.zeros(n_rays, dtype=bool)
    
    # Front surface (sphere)
    center = np.array([0.0, 0.0, vertex_z_front + R_front])
    t_front, hit_front = intersect_ray_sphere_vectorized(origins, directions, center, R_front)
    
    if not np.any(hit_front):
        return origins_out, directions_out, success
    
    # Calculate intersection points
    p_front = origins + t_front[:, np.newaxis] * directions  # (n_rays, 3)
    
    # Check aperture
    r_front = np.sqrt(p_front[:, 0]**2 + p_front[:, 1]**2)
    aperture_ok = r_front <= ap_rad
    active = hit_front & aperture_ok
    
    if not np.any(active):
        return origins_out, directions_out, success
    
    # Surface normals (pointing out of glass)
    normals = (p_front - center) / R_front  # (n_rays, 3)
    
    # Refract into glass
    dirs_in, refract_ok = refract_vec_vectorized(normals, directions, n_medium, n_glass)
    active = active & refract_ok
    
    if not np.any(active):
        return origins_out, directions_out, success
    
    # Propagate to back surface (flat plane at vertex_z_back)
    vertex_z_back = vertex_z_front + center_thickness
    
    # Ray-plane intersection: t = (plane_z - ray_z) / ray_dir_z
    # Safe division - only compute for active rays
    t_back = np.full(n_rays, np.nan)
    active_mask = active & (np.abs(dirs_in[:, 2]) > 1e-9)
    if np.any(active_mask):
        t_back[active_mask] = (vertex_z_back - p_front[active_mask, 2]) / dirs_in[active_mask, 2]
    
    # Check if intersection is forward (t > 0)
    t_positive = t_back > 0
    active = active & t_positive
    
    if not np.any(active):
        return origins_out, directions_out, success
    
    p_back = p_front + t_back[:, np.newaxis] * dirs_in  # (n_rays, 3)
    
    # Check back aperture
    r_back = np.sqrt(p_back[:, 0]**2 + p_back[:, 1]**2)
    aperture_back_ok = r_back <= ap_rad
    active = active & aperture_back_ok
    
    if not np.any(active):
        return origins_out, directions_out, success
    
    # Refract out of glass (planar surface, normal = -z)
    normals_back = np.tile([0, 0, -1], (n_rays, 1))
    dirs_out, refract_out_ok = refract_vec_vectorized(normals_back, dirs_in, n_glass, n_medium)
    active = active & refract_out_ok
    
    # Store successful rays
    origins_out[active] = p_back[active]
    directions_out[active] = dirs_out[active]
    success[active] = True
    
    return origins_out, directions_out, success


def _trace_flat_curved_vectorized(origins, directions, n_medium, lens_params):
    """Vectorized trace through flat-then-curved lens."""
    n_rays = origins.shape[0]
    n_glass = lens_params['n_glass']
    vertex_z_front = lens_params['vertex_z_front']
    R_front = lens_params['R_front_mm']
    center_thickness = lens_params['center_thickness_mm']
    ap_rad = lens_params['ap_rad_mm']
    vertex_z_back = vertex_z_front + center_thickness
    center_z_back = vertex_z_back - R_front
    
    # Initialize output arrays
    origins_out = np.full((n_rays, 3), np.nan)
    directions_out = np.full((n_rays, 3), np.nan)
    success = np.zeros(n_rays, dtype=bool)
    
    # Front surface (planar at vertex_z_front)
    # Check for parallel rays
    active = np.abs(directions[:, 2]) > 1e-9
    
    if not np.any(active):
        return origins_out, directions_out, success
    
    # Calculate intersection with plane
    t_front = np.full(n_rays, np.nan)
    t_front[active] = (vertex_z_front - origins[active, 2]) / directions[active, 2]
    
    # Check for forward intersection
    forward = t_front > 0
    active = active & forward
    
    if not np.any(active):
        return origins_out, directions_out, success
    
    # Intersection points
    p_front = origins + t_front[:, np.newaxis] * directions
    
    # Check front aperture
    r_front = np.sqrt(p_front[:, 0]**2 + p_front[:, 1]**2)
    aperture_ok = r_front <= ap_rad
    active = active & aperture_ok
    
    if not np.any(active):
        return origins_out, directions_out, success
    
    # Refract into glass (normal = [0, 0, -1])
    normals_front = np.tile([0, 0, -1], (n_rays, 1))
    dirs_in, refract_ok = refract_vec_vectorized(normals_front, directions, n_medium, n_glass)
    active = active & refract_ok
    
    if not np.any(active):
        return origins_out, directions_out, success
    
    # Back surface (sphere)
    center_back = np.array([0.0, 0.0, center_z_back])
    t_back, hit_back = intersect_ray_sphere_vectorized(p_front, dirs_in, center_back, R_front)
    active = active & hit_back
    
    if not np.any(active):
        return origins_out, directions_out, success
    
    # Back intersection points
    p_back = p_front + t_back[:, np.newaxis] * dirs_in
    
    # Check back aperture
    r_back = np.sqrt(p_back[:, 0]**2 + p_back[:, 1]**2)
    aperture_back_ok = r_back <= ap_rad
    active = active & aperture_back_ok
    
    if not np.any(active):
        return origins_out, directions_out, success
    
    # Surface normals (pointing out of glass)
    normals_back = -(p_back - center_back) / R_front
    
    # Refract out of glass
    dirs_out, refract_out_ok = refract_vec_vectorized(normals_back, dirs_in, n_glass, n_medium)
    active = active & refract_out_ok
    
    # Store successful rays
    origins_out[active] = p_back[active]
    directions_out[active] = dirs_out[active]
    success[active] = True
    
    return origins_out, directions_out, success


def trace_biconvex_vectorized(origins, directions, n_medium, lens_params, flipped=False):
    """
    Vectorized ray tracing through a bi-convex lens.
    
    BiConvex lenses have two spherical surfaces (front and back).
    
    Parameters
    ----------
    origins : ndarray, shape (n_rays, 3)
        Ray starting points
    directions : ndarray, shape (n_rays, 3)
        Ray direction vectors
    n_medium : float
        Refractive index of surrounding medium
    lens_params : dict
        Lens parameters with keys:
        - 'vertex_z_front': front surface z position
        - 'R_front_mm': radius of curvature for front surface
        - 'R_back_mm': radius of curvature for back surface
        - 'center_thickness_mm': thickness at center
        - 'ap_rad_mm': aperture radius
        - 'n_glass': glass refractive index
        - 'center_z_front': center of front sphere
        - 'center_z_back': center of back sphere
    flipped : bool
        If True, lens is flipped (radii are already swapped in lens object)
    
    Returns
    -------
    origins_out : ndarray, shape (n_rays, 3)
        Exit points from lens (NaN for failed rays)
    directions_out : ndarray, shape (n_rays, 3)
        Exit directions from lens (NaN for failed rays)
    success : ndarray, shape (n_rays,), dtype=bool
        True where rays successfully traced through lens
    """
    n_rays = origins.shape[0]
    n_glass = lens_params['n_glass']
    ap_rad = lens_params['ap_rad_mm']
    R_front = lens_params['R_front_mm']
    R_back = lens_params['R_back_mm']
    center_front = np.array([0.0, 0.0, lens_params['center_z_front']])
    center_back = np.array([0.0, 0.0, lens_params['center_z_back']])
    
    # Initialize output arrays
    origins_out = np.full((n_rays, 3), np.nan)
    directions_out = np.full((n_rays, 3), np.nan)
    success = np.zeros(n_rays, dtype=bool)
    
    # Front surface (sphere)
    t_front, hit_front = intersect_ray_sphere_vectorized(origins, directions, center_front, R_front)
    
    if not np.any(hit_front):
        return origins_out, directions_out, success
    
    # Calculate intersection points
    p_front = origins + t_front[:, np.newaxis] * directions  # (n_rays, 3)
    
    # Check aperture
    r_front = np.sqrt(p_front[:, 0]**2 + p_front[:, 1]**2)
    aperture_ok = r_front <= ap_rad
    active = hit_front & aperture_ok
    
    if not np.any(active):
        return origins_out, directions_out, success
    
    # Surface normals (pointing out of glass)
    normals_front = (p_front - center_front) / R_front  # (n_rays, 3)
    
    # Refract into glass
    dirs_in, refract_ok = refract_vec_vectorized(normals_front, directions, n_medium, n_glass)
    active = active & refract_ok
    
    if not np.any(active):
        return origins_out, directions_out, success
    
    # Back surface (sphere)
    t_back, hit_back = intersect_ray_sphere_vectorized(p_front, dirs_in, center_back, R_back)
    active = active & hit_back
    
    if not np.any(active):
        return origins_out, directions_out, success
    
    # Back intersection points
    p_back = p_front + t_back[:, np.newaxis] * dirs_in
    
    # Check back aperture
    r_back = np.sqrt(p_back[:, 0]**2 + p_back[:, 1]**2)
    aperture_back_ok = r_back <= ap_rad
    active = active & aperture_back_ok
    
    if not np.any(active):
        return origins_out, directions_out, success
    
    # Surface normals at back (pointing out of glass)
    # For convex back surface, center is on -z side, so outward normal points away from center
    normals_back = -(p_back - center_back) / R_back
    
    # Refract out of glass
    dirs_out, refract_out_ok = refract_vec_vectorized(normals_back, dirs_in, n_glass, n_medium)
    active = active & refract_out_ok
    
    # Store successful rays
    origins_out[active] = p_back[active]
    directions_out[active] = dirs_out[active]
    success[active] = True
    
    return origins_out, directions_out, success


def trace_aspheric_vectorized(origins, directions, n_medium, lens_params, flipped=False):
    """
    Vectorized ray tracing through an aspheric lens.
    
    NOTE: Currently uses spherical approximation (k=0) for both surfaces.
    Aspheric lenses have an aspheric front surface (approximated as sphere) 
    and a spherical, plano, or concave back surface.
    
    Parameters
    ----------
    origins : ndarray, shape (n_rays, 3)
        Ray starting points
    directions : ndarray, shape (n_rays, 3)
        Ray direction vectors
    n_medium : float
        Refractive index of surrounding medium
    lens_params : dict
        Lens parameters with keys:
        - 'vertex_z_front': front surface z position
        - 'vertex_z_back': back surface z position
        - 'R_front_mm': base radius of aspheric front surface (approximated as sphere)
        - 'R_back_mm': radius of back surface (can be positive, negative, or >1000 for plano)
        - 'ap_rad_mm': aperture radius
        - 'n_glass': glass refractive index
        - 'center_z_front': center of front sphere
        - 'center_z_back': center of back sphere (None for plano)
        - 'back_surface_type': 'plano', 'convex', or 'concave'
    flipped : bool
        If True, lens is flipped (radii are already swapped in lens object)
    
    Returns
    -------
    origins_out : ndarray, shape (n_rays, 3)
        Exit points from lens (NaN for failed rays)
    directions_out : ndarray, shape (n_rays, 3)
        Exit directions from lens (NaN for failed rays)
    success : ndarray, shape (n_rays,), dtype=bool
        True where rays successfully traced through lens
    """
    n_rays = origins.shape[0]
    n_glass = lens_params['n_glass']
    ap_rad = lens_params['ap_rad_mm']
    R_front = lens_params['R_front_mm']
    R_back = lens_params['R_back_mm']
    vertex_z_back = lens_params['vertex_z_back']
    center_front = np.array([0.0, 0.0, lens_params['center_z_front']])
    back_surface_type = lens_params['back_surface_type']
    
    # Initialize output arrays
    origins_out = np.full((n_rays, 3), np.nan)
    directions_out = np.full((n_rays, 3), np.nan)
    success = np.zeros(n_rays, dtype=bool)
    
    # Front surface (aspheric, approximated as sphere with k=0)
    t_front, hit_front = intersect_ray_sphere_vectorized(origins, directions, center_front, R_front)
    
    if not np.any(hit_front):
        return origins_out, directions_out, success
    
    # Calculate intersection points
    p_front = origins + t_front[:, np.newaxis] * directions  # (n_rays, 3)
    
    # Check aperture
    r_front = np.sqrt(p_front[:, 0]**2 + p_front[:, 1]**2)
    aperture_ok = r_front <= ap_rad
    active = hit_front & aperture_ok
    
    if not np.any(active):
        return origins_out, directions_out, success
    
    # Surface normals (pointing out of glass)
    normals_front = (p_front - center_front) / R_front  # (n_rays, 3)
    
    # Refract into glass
    dirs_in, refract_ok = refract_vec_vectorized(normals_front, directions, n_medium, n_glass)
    active = active & refract_ok
    
    if not np.any(active):
        return origins_out, directions_out, success
    
    # Back surface - handle three types: plano, convex, concave
    if back_surface_type == 'plano':
        # Planar back surface
        dz_ok = np.abs(dirs_in[:, 2]) > 1e-9
        active = active & dz_ok
        
        if not np.any(active):
            return origins_out, directions_out, success
        
        t_back = np.full(n_rays, np.nan)
        t_back[active] = (vertex_z_back - p_front[active, 2]) / dirs_in[active, 2]
        
        # Check for forward propagation
        forward = t_back > 0
        active = active & forward
        
        if not np.any(active):
            return origins_out, directions_out, success
        
        # Back intersection points
        p_back = p_front + t_back[:, np.newaxis] * dirs_in
        
        # Check back aperture
        r_back = np.sqrt(p_back[:, 0]**2 + p_back[:, 1]**2)
        aperture_back_ok = r_back <= ap_rad
        active = active & aperture_back_ok
        
        if not np.any(active):
            return origins_out, directions_out, success
        
        # Surface normal for plane (points outward from glass, toward -z)
        normals_back = np.tile([0, 0, -1], (n_rays, 1))
        
    elif back_surface_type == 'convex':
        # Convex spherical back surface
        center_back = np.array([0.0, 0.0, lens_params['center_z_back']])
        t_back, hit_back = intersect_ray_sphere_vectorized(p_front, dirs_in, center_back, R_back)
        active = active & hit_back
        
        if not np.any(active):
            return origins_out, directions_out, success
        
        # Back intersection points
        p_back = p_front + t_back[:, np.newaxis] * dirs_in
        
        # Check back aperture
        r_back = np.sqrt(p_back[:, 0]**2 + p_back[:, 1]**2)
        aperture_back_ok = r_back <= ap_rad
        active = active & aperture_back_ok
        
        if not np.any(active):
            return origins_out, directions_out, success
        
        # Surface normals at back (pointing out of glass, toward +z)
        normals_back = -(p_back - center_back) / R_back
        
    else:  # concave
        # Concave spherical back surface
        center_back = np.array([0.0, 0.0, lens_params['center_z_back']])
        t_back, hit_back = intersect_ray_sphere_vectorized(p_front, dirs_in, center_back, abs(R_back))
        active = active & hit_back
        
        if not np.any(active):
            return origins_out, directions_out, success
        
        # Back intersection points
        p_back = p_front + t_back[:, np.newaxis] * dirs_in
        
        # Check back aperture
        r_back = np.sqrt(p_back[:, 0]**2 + p_back[:, 1]**2)
        aperture_back_ok = r_back <= ap_rad
        active = active & aperture_back_ok
        
        if not np.any(active):
            return origins_out, directions_out, success
        
        # Surface normals at back (pointing out of glass)
        normals_back = (p_back - center_back) / abs(R_back)
    
    # Refract out of glass
    dirs_out, refract_out_ok = refract_vec_vectorized(normals_back, dirs_in, n_glass, n_medium)
    active = active & refract_out_ok
    
    # Store successful rays
    origins_out[active] = p_back[active]
    directions_out[active] = dirs_out[active]
    success[active] = True
    
    return origins_out, directions_out, success


def trace_system_vectorized(origins, dirs, lens1, lens2,
                            z_fiber, fiber_rad, acceptance_half_rad, 
                            medium='air', pressure_atm=1.0, temp_k=293.15, humidity_fraction=0.01):
    """
    Vectorized ray tracing through two-lens system to fiber.
    
    This is a drop-in replacement for trace_system() with ~10-50x speedup.
    
    Parameters
    ----------
    origins : ndarray, shape (n_rays, 3)
        Ray starting points
    dirs : ndarray, shape (n_rays, 3)
        Ray direction vectors
    lens1 : PlanoConvex
        First lens object
    lens2 : PlanoConvex
        Second lens object
    z_fiber : float
        Z-position of fiber face
    fiber_rad : float
        Fiber core radius
    acceptance_half_rad : float
        Fiber half-acceptance angle in radians
    medium : str
        Propagation medium ('air', 'argon', 'helium')
    pressure_atm : float
        Atmospheric pressure
    temp_k : float
        Temperature in Kelvin
    humidity_fraction : float
        Water vapor fraction (0.0-0.1)
    
    Returns
    -------
    accepted : ndarray, shape (n_rays,), dtype=bool
        True for rays that coupled into fiber
    transmission_factors : ndarray, shape (n_rays,)
        Atmospheric transmission factors for each ray
    """
    n_rays = origins.shape[0]
    n_medium = medium_refractive_index(C.WAVELENGTH_NM, medium, pressure_atm, temp_k)
    
    # Detect lens type and extract appropriate parameters
    def get_lens_params_and_tracer(lens):
        """Extract parameters and select appropriate tracer function for a lens."""
        lens_type = type(lens).__name__
        
        # Common parameters for all lens types
        params = {
            'vertex_z_front': lens.vertex_z_front,
            'R_front_mm': lens.R_front_mm,
            'center_thickness_mm': lens.center_thickness_mm,
            'ap_rad_mm': lens.ap_rad_mm,
            'n_glass': lens.n_glass
        }
        
        if lens_type == 'PlanoConvex':
            # PlanoConvex: has edge_thickness_mm
            params['edge_thickness_mm'] = lens.edge_thickness_mm
            return params, trace_planoconvex_vectorized
            
        elif lens_type == 'BiConvex':
            # BiConvex: has R_back_mm and sphere centers
            params['R_back_mm'] = lens.R_back_mm
            params['center_z_front'] = lens.center_z_front
            params['center_z_back'] = lens.center_z_back
            return params, trace_biconvex_vectorized
            
        elif lens_type == 'Aspheric':
            # Aspheric: has R_back_mm, back surface type, and centers
            params['R_back_mm'] = lens.R_back_mm
            params['vertex_z_back'] = lens.vertex_z_back
            params['center_z_front'] = lens.center_z_front
            params['center_z_back'] = lens.center_z_back
            params['back_surface_type'] = lens.back_surface_type
            return params, trace_aspheric_vectorized
            
        else:
            raise ValueError(f"Unsupported lens type for vectorized tracing: {lens_type}")
    
    # Get parameters and tracer for each lens
    lens1_params, trace_lens1 = get_lens_params_and_tracer(lens1)
    lens2_params, trace_lens2 = get_lens_params_and_tracer(lens2)
    
    # Trace through lens 1
    origins1_out, dirs1_out, success1 = trace_lens1(
        origins, dirs, n_medium, lens1_params, flipped=lens1.flipped
    )
    
    # Early exit if all rays failed
    if not np.any(success1):
        return np.zeros(n_rays, dtype=bool), np.ones(n_rays)
    
    # Calculate transmission to lens 1
    z_l1 = lens1.z_center
    d1 = z_l1 - origins[:, 2]
    T1 = transmission_through_medium_vectorized(d1, C.WAVELENGTH_NM, medium, 
                                               pressure_atm, temp_k, humidity_fraction)
    
    # Trace through lens 2
    origins2_out, dirs2_out, success2 = trace_lens2(
        origins1_out, dirs1_out, n_medium, lens2_params, flipped=lens2.flipped
    )
    
    # Combine success masks
    success_both = success1 & success2
    
    if not np.any(success_both):
        return np.zeros(n_rays, dtype=bool), np.ones(n_rays)
    
    # Calculate transmission between lenses
    z_l2 = lens2.z_center
    d2 = z_l2 - origins1_out[:, 2]
    d2 = np.where(success1, d2, 0.0)  # Zero out failed rays
    T2 = transmission_through_medium_vectorized(d2, C.WAVELENGTH_NM, medium,
                                               pressure_atm, temp_k, humidity_fraction)
    
    # Propagate to fiber
    # Check for rays not going forward
    dz_ok = np.abs(dirs2_out[:, 2]) > 1e-9
    active = success_both & dz_ok
    
    # Calculate fiber intersection
    t_fiber = np.full(n_rays, np.nan)
    active_mask = active & (dirs2_out[:, 2] != 0)
    if np.any(active_mask):
        t_fiber[active_mask] = (z_fiber - origins2_out[active_mask, 2]) / dirs2_out[active_mask, 2]
    
    # Check for forward propagation
    forward = t_fiber > 0
    active = active & forward
    
    # Calculate positions at fiber
    p_fiber = origins2_out + t_fiber[:, np.newaxis] * dirs2_out
    
    # Calculate transmission to fiber
    d3 = np.abs(z_fiber - origins2_out[:, 2])
    d3 = np.where(success_both, d3, 0.0)
    T3 = transmission_through_medium_vectorized(d3, C.WAVELENGTH_NM, medium,
                                               pressure_atm, temp_k, humidity_fraction)
    
    # Check spatial acceptance (fiber radius)
    r_fiber = np.sqrt(p_fiber[:, 0]**2 + p_fiber[:, 1]**2)
    spatial_ok = r_fiber <= fiber_rad
    active = active & spatial_ok
    
    # Check angular acceptance (NA)
    # Calculate angle from z-axis
    dir_norms = np.linalg.norm(dirs2_out, axis=1)
    cos_theta = np.where(dir_norms > 0, np.abs(dirs2_out[:, 2]) / dir_norms, 0.0)
    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    angular_ok = theta <= acceptance_half_rad
    active = active & angular_ok
    
    # Calculate total transmission
    transmission_factors = T1 * T2 * T3
    
    return active, transmission_factors
