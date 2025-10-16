import numpy as np
from scripts.PlanoConvex import PlanoConvex
from scripts import consts as C
from scripts.raytrace_helpers import sample_rays, trace_system

try:
    from skopt import gp_minimize
    from skopt.space import Real
    from skopt.utils import use_named_args
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False


def optimize(lenses, name1, name2, n_calls=100, n_rays=1000, alpha=0.7):
    if not SKOPT_AVAILABLE:
        raise ImportError("scikit-optimize not installed. Run: pip install scikit-optimize")
    
    d1, d2 = lenses[name1], lenses[name2]
    f1, f2 = d1['f_mm'], d2['f_mm']
    
    origins, dirs = sample_rays(n_rays)
    
    z_l1_max = max(C.SOURCE_TO_LENS_OFFSET + 5.0, f1 * 1.5)
    z_l2_min = C.SOURCE_TO_LENS_OFFSET + 0.5
    z_l2_max = z_l1_max + f2 * 0.5
    
    space = [
        Real(C.SOURCE_TO_LENS_OFFSET, z_l1_max, name='z_l1'),
        Real(z_l2_min, z_l2_max, name='z_l2')
    ]
    
    @use_named_args(space)
    def objective(z_l1, z_l2):
        if z_l2 <= z_l1 + 0.1:
            return 1e6
        
        z_fiber = z_l2 + f2
        lens1 = PlanoConvex(z_l1, d1['R_mm'], d1['tc_mm'], d1['te_mm'], d1['dia']/2.0)
        lens2 = PlanoConvex(z_l2, d2['R_mm'], d2['tc_mm'], d2['te_mm'], d2['dia']/2.0)
        
        accepted = trace_system(origins, dirs, lens1, lens2, z_fiber, 
                               C.FIBER_CORE_DIAM_MM/2.0, C.ACCEPTANCE_HALF_RAD)
        coupling = np.count_nonzero(accepted) / n_rays
        normalized_length = z_fiber / 80.0
        return alpha * (1 - coupling) + (1 - alpha) * normalized_length
    
    result = gp_minimize(objective, space, n_calls=n_calls, random_state=42, 
                        verbose=False, n_initial_points=20)
    
    z_l1_opt, z_l2_opt = result.x
    z_fiber_opt = z_l2_opt + f2
    
    origins_final, dirs_final = sample_rays(2000)
    lens1 = PlanoConvex(z_l1_opt, d1['R_mm'], d1['tc_mm'], d1['te_mm'], d1['dia']/2.0)
    lens2 = PlanoConvex(z_l2_opt, d2['R_mm'], d2['tc_mm'], d2['te_mm'], d2['dia']/2.0)
    accepted = trace_system(origins_final, dirs_final, lens1, lens2, z_fiber_opt,
                           C.FIBER_CORE_DIAM_MM/2.0, C.ACCEPTANCE_HALF_RAD)
    
    return {
        'lens1': name1, 'lens2': name2, 'f1_mm': f1, 'f2_mm': f2,
        'z_l1': z_l1_opt, 'z_l2': z_l2_opt, 'z_fiber': z_fiber_opt,
        'total_len_mm': z_fiber_opt, 'coupling': np.count_nonzero(accepted) / 2000,
        'origins': origins_final, 'dirs': dirs_final, 'accepted': accepted
    }
