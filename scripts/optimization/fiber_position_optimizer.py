"""
Helper functions for optimizing fiber position.

This module provides utilities to test both hardcoded (focal length) and optimized
fiber positions, returning the configuration with better coupling.
"""

import numpy as np
from scipy.optimize import minimize_scalar
from scripts import consts as C
from scripts.raytrace_helpers_vectorized import trace_system_vectorized as trace_system


def optimize_fiber_position(lens1, lens2, z_l2, f2, origins, dirs, medium='air'):
    """
    Find optimal fiber position by 1D optimization.
    
    Parameters
    ----------
    lens1 : PlanoConvex
        First lens object
    lens2 : PlanoConvex
        Second lens object
    z_l2 : float
        Position of second lens
    f2 : float
        Focal length of second lens
    origins : ndarray
        Ray origins
    dirs : ndarray
        Ray directions
    medium : str
        Propagation medium
    
    Returns
    -------
    z_fiber_opt : float
        Optimal fiber position
    coupling_opt : float
        Coupling efficiency at optimal position
    """
    n_rays = origins.shape[0]
    
    def eval_fiber_pos(z_fiber):
        """Evaluate coupling for a given fiber position."""
        accepted, transmission = trace_system(
            origins, dirs, lens1, lens2, z_fiber,
            C.FIBER_CORE_DIAM_MM/2.0, C.ACCEPTANCE_HALF_RAD,
            medium, C.PRESSURE_ATM, C.TEMPERATURE_K, C.HUMIDITY_FRACTION
        )
        avg_transmission = np.mean(transmission[accepted]) if np.any(accepted) else 0.0
        coupling = (np.count_nonzero(accepted) / n_rays) * avg_transmission
        return -coupling  # Negative because minimize_scalar minimizes
    
    # Search range: 0.5*f2 to 1.5*f2 from second lens
    bounds = (z_l2 + 0.5 * f2, z_l2 + 1.5 * f2)
    
    result = minimize_scalar(eval_fiber_pos, bounds=bounds, method='bounded')
    
    z_fiber_opt = result.x
    coupling_opt = -result.fun  # Convert back to positive coupling
    
    return z_fiber_opt, coupling_opt


def evaluate_both_fiber_positions(lens1, lens2, z_l2, f2, origins, dirs, medium='air'):
    """
    Evaluate coupling for both hardcoded (focal length) and optimized fiber positions.
    
    Returns the configuration with better coupling.
    
    Parameters
    ----------
    lens1 : PlanoConvex
        First lens object
    lens2 : PlanoConvex
        Second lens object
    z_l2 : float
        Position of second lens
    f2 : float
        Focal length of second lens
    origins : ndarray
        Ray origins
    dirs : ndarray
        Ray directions
    medium : str
        Propagation medium
    
    Returns
    -------
    z_fiber : float
        Best fiber position
    coupling : float
        Coupling efficiency at best position
    fiber_position_method : str
        Method used ('focal_length' or 'optimized')
    """
    n_rays = origins.shape[0]
    
    # Method 1: Hardcoded at focal length
    z_fiber_focal = z_l2 + f2
    accepted_focal, transmission_focal = trace_system(
        origins, dirs, lens1, lens2, z_fiber_focal,
        C.FIBER_CORE_DIAM_MM/2.0, C.ACCEPTANCE_HALF_RAD,
        medium, C.PRESSURE_ATM, C.TEMPERATURE_K, C.HUMIDITY_FRACTION
    )
    avg_transmission_focal = np.mean(transmission_focal[accepted_focal]) if np.any(accepted_focal) else 0.0
    coupling_focal = (np.count_nonzero(accepted_focal) / n_rays) * avg_transmission_focal
    
    # Method 2: Optimized position
    z_fiber_opt, coupling_opt = optimize_fiber_position(
        lens1, lens2, z_l2, f2, origins, dirs, medium
    )
    
    # Return the better one
    if coupling_opt > coupling_focal:
        return z_fiber_opt, coupling_opt, 'optimized'
    else:
        return z_fiber_focal, coupling_focal, 'focal_length'
