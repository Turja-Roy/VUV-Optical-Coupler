import numpy as np
from scipy.optimize import differential_evolution, dual_annealing, minimize
from ..PlanoConvex import PlanoConvex
from .. import consts as C
from ..raytrace_helpers import sample_rays, trace_system


def evaluate_config_fast(params, d1, d2, origins, dirs, n_rays, alpha=0.7):
    z_l1, z_l2 = params
    
    if z_l1 < C.SOURCE_TO_LENS_OFFSET or z_l2 <= z_l1 + 0.1:
        return 1e6
    
    z_fiber = z_l2 + d2['f_mm']
    
    lens1 = PlanoConvex(z_l1, d1['R_mm'], d1['t_mm'], d1['dia']/2.0)
    lens2 = PlanoConvex(z_l2, d2['R_mm'], d2['t_mm'], d2['dia']/2.0)
    
    accepted = trace_system(origins, dirs, lens1, lens2, z_fiber, 
                           C.FIBER_CORE_DIAM_MM/2.0, C.ACCEPTANCE_HALF_RAD)
    coupling = np.count_nonzero(accepted) / n_rays
    
    return alpha * (1 - coupling) + (1 - alpha) * z_fiber / 200.0


def _get_bounds(f1, f2):
    return [
        (C.SOURCE_TO_LENS_OFFSET, f1 * 1.5),
        (C.SOURCE_TO_LENS_OFFSET + f2 * 0.5, f1 * 1.5 + f2 * 2.5)
    ]


def _extract_result(result, f2):
    z_l1, z_l2 = result.x
    return z_l1, z_l2, z_l2 + f2, result.fun


def optimize_differential_evolution(d1, d2, origins, dirs, n_rays, alpha=0.7):
    bounds = _get_bounds(d1['f_mm'], d2['f_mm'])
    result = differential_evolution(
        evaluate_config_fast, bounds,
        args=(d1, d2, origins, dirs, n_rays, alpha),
        maxiter=50, popsize=10, seed=42, atol=0.001, tol=0.001, workers=1
    )
    return _extract_result(result, d2['f_mm'])


def optimize_dual_annealing(d1, d2, origins, dirs, n_rays, alpha=0.7):
    bounds = _get_bounds(d1['f_mm'], d2['f_mm'])
    result = dual_annealing(
        evaluate_config_fast, bounds,
        args=(d1, d2, origins, dirs, n_rays, alpha), maxiter=300, seed=42
    )
    return _extract_result(result, d2['f_mm'])


def optimize_nelder_mead(d1, d2, origins, dirs, n_rays, alpha=0.7, x0=None):
    f1, f2 = d1['f_mm'], d2['f_mm']
    x0 = x0 or [f1 * 0.8, f1 * 0.8 + f2 * 1.2]
    result = minimize(
        evaluate_config_fast, x0,
        args=(d1, d2, origins, dirs, n_rays, alpha),
        method='Nelder-Mead',
        options={'maxiter': 200, 'xatol': 0.01, 'fatol': 0.001}
    )
    return _extract_result(result, f2)


def optimize_powell(d1, d2, origins, dirs, n_rays, alpha=0.7, x0=None):
    f1, f2 = d1['f_mm'], d2['f_mm']
    x0 = x0 or [f1 * 0.8, f1 * 0.8 + f2 * 1.2]
    result = minimize(
        evaluate_config_fast, x0,
        args=(d1, d2, origins, dirs, n_rays, alpha),
        method='Powell',
        options={'maxiter': 200, 'xtol': 0.01, 'ftol': 0.001}
    )
    return _extract_result(result, f2)


def run_optimization(run_date, lenses, name1, name2, 
                     method='differential_evolution',
                     n_rays=1000, alpha=0.7):
    d1, d2 = lenses[name1], lenses[name2]
    origins, dirs = sample_rays(n_rays)
    
    optimizers = {
        'differential_evolution': optimize_differential_evolution,
        'dual_annealing': optimize_dual_annealing,
        'nelder_mead': optimize_nelder_mead,
        'powell': optimize_powell
    }
    
    if method not in optimizers:
        raise ValueError(f"Unknown method: {method}")
    
    z_l1, z_l2, z_fiber, obj_val = optimizers[method](d1, d2, origins, dirs, n_rays, alpha)
    
    origins_final, dirs_final = sample_rays(2000)
    lens1 = PlanoConvex(z_l1, d1['R_mm'], d1['t_mm'], d1['dia']/2.0)
    lens2 = PlanoConvex(z_l2, d2['R_mm'], d2['t_mm'], d2['dia']/2.0)
    
    accepted = trace_system(origins_final, dirs_final, lens1, lens2,
                           z_fiber, C.FIBER_CORE_DIAM_MM/2.0, C.ACCEPTANCE_HALF_RAD)
    
    return {
        'lens1': name1, 'lens2': name2,
        'f1_mm': d1['f_mm'], 'f2_mm': d2['f_mm'],
        'z_l1': z_l1, 'z_l2': z_l2, 'z_fiber': z_fiber,
        'total_len_mm': z_fiber,
        'coupling': np.count_nonzero(accepted) / 2000,
        'origins': origins_final, 'dirs': dirs_final, 'accepted': accepted
    }
