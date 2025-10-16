import numpy as np
from scipy.optimize import differential_evolution
from scripts.PlanoConvex import PlanoConvex
from scripts import consts as C
from scripts.raytrace_helpers import sample_rays, trace_system


def evaluate_config_fast(params, d1, d2, origins, dirs, n_rays, alpha=0.7):
    z_l1, z_l2 = params
    
    if z_l1 < C.SOURCE_TO_LENS_OFFSET or z_l2 <= z_l1 + 0.1:
        return 1e6
    
    z_fiber = z_l2 + d2['f_mm']
    
    lens1 = PlanoConvex(z_l1, d1['R_mm'], d1['tc_mm'], d1['te_mm'], d1['dia']/2.0)
    lens2 = PlanoConvex(z_l2, d2['R_mm'], d2['tc_mm'], d2['te_mm'], d2['dia']/2.0)
    
    accepted = trace_system(origins, dirs, lens1, lens2, z_fiber, 
                           C.FIBER_CORE_DIAM_MM/2.0, C.ACCEPTANCE_HALF_RAD)
    coupling = np.count_nonzero(accepted) / n_rays
    
    return alpha * (1 - coupling) + (1 - alpha) * z_fiber / 80.0


def get_bounds(f1, f2):
    z_l1_max = max(C.SOURCE_TO_LENS_OFFSET + 5.0, f1 * 1.5)
    z_l2_max = max(z_l1_max + 5.0, f1 * 1.5 + f2 * 2.5)
    return [
        (C.SOURCE_TO_LENS_OFFSET, z_l1_max),
        (C.SOURCE_TO_LENS_OFFSET + 1.0, z_l2_max)
    ]


def optimize(lenses, name1, name2, n_rays=1000, alpha=0.7):
    d1, d2 = lenses[name1], lenses[name2]
    origins, dirs = sample_rays(n_rays)
    
    bounds = get_bounds(d1['f_mm'], d2['f_mm'])
    result = differential_evolution(
        evaluate_config_fast, bounds,
        args=(d1, d2, origins, dirs, n_rays, alpha),
        maxiter=50, popsize=10, seed=42, atol=0.001, tol=0.001, workers=1
    )
    
    z_l1, z_l2 = result.x
    z_fiber = z_l2 + d2['f_mm']
    
    origins_final, dirs_final = sample_rays(2000)
    lens1 = PlanoConvex(z_l1, d1['R_mm'], d1['tc_mm'], d1['te_mm'], d1['dia']/2.0)
    lens2 = PlanoConvex(z_l2, d2['R_mm'], d2['tc_mm'], d2['te_mm'], d2['dia']/2.0)
    
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
