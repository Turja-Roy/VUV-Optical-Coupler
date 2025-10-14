import numpy as np
from scipy.optimize import differential_evolution, dual_annealing, minimize
from tqdm import tqdm
from ..PlanoConvex import PlanoConvex
from .. import consts as C
from ..raytrace_helpers import sample_rays, trace_system


def evaluate_config_fast(params, d1, d2, origins, dirs, n_rays, alpha=0.7):
    """
    Fast objective function combining coupling and total length.
    
    Parameters:
    - params: [z_l1, z_l2] positions
    - alpha: weight for coupling (1-alpha for length minimization)
    
    Returns:
    - objective value (lower is better)
    """
    z_l1, z_l2 = params
    
    # Constraints
    if z_l1 < C.SOURCE_TO_LENS_OFFSET:
        return 1e6
    if z_l2 <= z_l1 + 0.1:
        return 1e6
    
    # Calculate fiber position
    z_fiber = z_l2 + d2['f_mm']
    
    # Create lenses
    lens1 = PlanoConvex(vertex_z_front=z_l1,
                        R_front_mm=d1['R_mm'],
                        thickness_mm=d1['t_mm'],
                        ap_rad_mm=d1['dia']/2.0)
    lens2 = PlanoConvex(vertex_z_front=z_l2,
                        R_front_mm=d2['R_mm'],
                        thickness_mm=d2['t_mm'],
                        ap_rad_mm=d2['dia']/2.0)
    
    # Trace rays
    accepted = trace_system(origins, dirs, lens1, lens2,
                           z_fiber, C.FIBER_CORE_DIAM_MM/2.0,
                           C.ACCEPTANCE_HALF_RAD)
    coupling = np.count_nonzero(accepted) / n_rays
    
    # Multi-objective: minimize -coupling and length
    # Normalize length to ~[0,1] scale
    normalized_length = z_fiber / 200.0  # typical max length
    
    # Combined objective (lower is better)
    objective = alpha * (1 - coupling) + (1 - alpha) * normalized_length
    
    return objective


def optimize_differential_evolution(d1, d2, origins, dirs, n_rays, alpha=0.7):
    """
    Use Differential Evolution - good for global optimization.
    Fast and robust for non-smooth objectives.
    """
    f1, f2 = d1['f_mm'], d2['f_mm']
    
    # Define bounds
    bounds = [
        (C.SOURCE_TO_LENS_OFFSET, f1 * 1.5),  # z_l1
        (C.SOURCE_TO_LENS_OFFSET + f2 * 0.5, f1 * 1.5 + f2 * 2.5)  # z_l2
    ]
    
    result = differential_evolution(
        evaluate_config_fast,
        bounds,
        args=(d1, d2, origins, dirs, n_rays, alpha),
        maxiter=50,
        popsize=10,
        seed=42,
        atol=0.001,
        tol=0.001,
        workers=1
    )
    
    z_l1, z_l2 = result.x
    z_fiber = z_l2 + f2
    
    return z_l1, z_l2, z_fiber, result.fun


def optimize_dual_annealing(d1, d2, origins, dirs, n_rays, alpha=0.7):
    """
    Use Dual Annealing - combines simulated annealing with local search.
    Good balance between exploration and exploitation.
    """
    f1, f2 = d1['f_mm'], d2['f_mm']
    
    bounds = [
        (C.SOURCE_TO_LENS_OFFSET, f1 * 1.5),
        (C.SOURCE_TO_LENS_OFFSET + f2 * 0.5, f1 * 1.5 + f2 * 2.5)
    ]
    
    result = dual_annealing(
        evaluate_config_fast,
        bounds,
        args=(d1, d2, origins, dirs, n_rays, alpha),
        maxiter=300,
        seed=42
    )
    
    z_l1, z_l2 = result.x
    z_fiber = z_l2 + f2
    
    return z_l1, z_l2, z_fiber, result.fun


def optimize_nelder_mead(d1, d2, origins, dirs, n_rays, alpha=0.7, x0=None):
    """
    Use Nelder-Mead simplex algorithm - gradient-free local optimization.
    Fast but needs good initial guess.
    """
    f1, f2 = d1['f_mm'], d2['f_mm']
    
    # Initial guess
    if x0 is None:
        x0 = [f1 * 0.8, f1 * 0.8 + f2 * 1.2]
    
    # Bounds (soft constraints via penalty)
    bounds = [
        (C.SOURCE_TO_LENS_OFFSET, f1 * 1.5),
        (C.SOURCE_TO_LENS_OFFSET + f2 * 0.5, f1 * 1.5 + f2 * 2.5)
    ]
    
    result = minimize(
        evaluate_config_fast,
        x0,
        args=(d1, d2, origins, dirs, n_rays, alpha),
        method='Nelder-Mead',
        options={'maxiter': 200, 'xatol': 0.01, 'fatol': 0.001}
    )
    
    z_l1, z_l2 = result.x
    z_fiber = z_l2 + f2
    
    return z_l1, z_l2, z_fiber, result.fun


def optimize_powell(d1, d2, origins, dirs, n_rays, alpha=0.7, x0=None):
    """
    Use Powell's method - another gradient-free optimizer.
    Can be faster than Nelder-Mead for some problems.
    """
    f1, f2 = d1['f_mm'], d2['f_mm']
    
    if x0 is None:
        x0 = [f1 * 0.8, f1 * 0.8 + f2 * 1.2]
    
    result = minimize(
        evaluate_config_fast,
        x0,
        args=(d1, d2, origins, dirs, n_rays, alpha),
        method='Powell',
        options={'maxiter': 200, 'xtol': 0.01, 'ftol': 0.001}
    )
    
    z_l1, z_l2 = result.x
    z_fiber = z_l2 + f2
    
    return z_l1, z_l2, z_fiber, result.fun


def run_optimization(run_date, lenses, name1, name2, 
                     method='differential_evolution',
                     n_rays=1000, alpha=0.7):
    """
    Run optimization for a lens pair using specified method.
    
    Parameters:
    - method: 'differential_evolution', 'dual_annealing', 'nelder_mead', 'powell'
    - n_rays: number of rays for evaluation
    - alpha: weight for coupling vs. length (higher = prioritize coupling)
    """
    d1 = lenses[name1]
    d2 = lenses[name2]
    f1 = d1['f_mm']
    f2 = d2['f_mm']
    
    # Generate ray set
    origins, dirs = sample_rays(n_rays)
    
    # Run optimization
    if method == 'differential_evolution':
        z_l1, z_l2, z_fiber, obj_val = optimize_differential_evolution(
            d1, d2, origins, dirs, n_rays, alpha)
    elif method == 'dual_annealing':
        z_l1, z_l2, z_fiber, obj_val = optimize_dual_annealing(
            d1, d2, origins, dirs, n_rays, alpha)
    elif method == 'nelder_mead':
        z_l1, z_l2, z_fiber, obj_val = optimize_nelder_mead(
            d1, d2, origins, dirs, n_rays, alpha)
    elif method == 'powell':
        z_l1, z_l2, z_fiber, obj_val = optimize_powell(
            d1, d2, origins, dirs, n_rays, alpha)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Final evaluation with more rays for accuracy
    origins_final, dirs_final = sample_rays(2000)
    lens1 = PlanoConvex(vertex_z_front=z_l1,
                        R_front_mm=d1['R_mm'],
                        thickness_mm=d1['t_mm'],
                        ap_rad_mm=d1['dia']/2.0)
    lens2 = PlanoConvex(vertex_z_front=z_l2,
                        R_front_mm=d2['R_mm'],
                        thickness_mm=d2['t_mm'],
                        ap_rad_mm=d2['dia']/2.0)
    
    accepted = trace_system(origins_final, dirs_final, lens1, lens2,
                           z_fiber, C.FIBER_CORE_DIAM_MM/2.0,
                           C.ACCEPTANCE_HALF_RAD)
    coupling = np.count_nonzero(accepted) / 2000
    
    result = {
        'lens1': name1,
        'lens2': name2,
        'f1_mm': f1,
        'f2_mm': f2,
        'z_l1': z_l1,
        'z_l2': z_l2,
        'z_fiber': z_fiber,
        'total_len_mm': z_fiber,
        'coupling': coupling,
        'origins': origins_final,
        'dirs': dirs_final,
        'accepted': accepted
    }
    
    return result
