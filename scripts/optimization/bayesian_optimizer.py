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


def run_bayesian_optimization(run_date, lenses, name1, name2, 
                               n_calls=50, n_rays=1000, alpha=0.7):
    if not SKOPT_AVAILABLE:
        raise ImportError("scikit-optimize not installed. Run: pip install scikit-optimize")
    
    d1, d2 = lenses[name1], lenses[name2]
    f1, f2 = d1['f_mm'], d2['f_mm']
    
    origins, dirs = sample_rays(n_rays)
    
    space = [
        Real(C.SOURCE_TO_LENS_OFFSET, f1 * 1.5, name='z_l1'),
        Real(C.SOURCE_TO_LENS_OFFSET + 1.0, f1 * 1.5 + f2 * 2.5, name='z_l2')
    ]
    
    @use_named_args(space)
    def objective(z_l1, z_l2):
        if z_l2 <= z_l1 + 0.1:
            return 1e6
        
        z_fiber = z_l2 + f2
        lens1 = PlanoConvex(z_l1, d1['R_mm'], d1['t_mm'], d1['dia']/2.0)
        lens2 = PlanoConvex(z_l2, d2['R_mm'], d2['t_mm'], d2['dia']/2.0)
        
        accepted = trace_system(origins, dirs, lens1, lens2, z_fiber, 
                               C.FIBER_CORE_DIAM_MM/2.0, C.ACCEPTANCE_HALF_RAD)
        coupling = np.count_nonzero(accepted) / n_rays
        normalized_length = z_fiber / 200.0
        return alpha * (1 - coupling) + (1 - alpha) * normalized_length
    
    result = gp_minimize(objective, space, n_calls=n_calls, random_state=42, 
                        verbose=False, n_initial_points=10)
    
    z_l1_opt, z_l2_opt = result.x
    z_fiber_opt = z_l2_opt + f2
    
    origins_final, dirs_final = sample_rays(2000)
    lens1 = PlanoConvex(z_l1_opt, d1['R_mm'], d1['t_mm'], d1['dia']/2.0)
    lens2 = PlanoConvex(z_l2_opt, d2['R_mm'], d2['t_mm'], d2['dia']/2.0)
    accepted = trace_system(origins_final, dirs_final, lens1, lens2, z_fiber_opt,
                           C.FIBER_CORE_DIAM_MM/2.0, C.ACCEPTANCE_HALF_RAD)
    
    return {
        'lens1': name1, 'lens2': name2, 'f1_mm': f1, 'f2_mm': f2,
        'z_l1': z_l1_opt, 'z_l2': z_l2_opt, 'z_fiber': z_fiber_opt,
        'total_len_mm': z_fiber_opt, 'coupling': np.count_nonzero(accepted) / 2000,
        'origins': origins_final, 'dirs': dirs_final, 'accepted': accepted
    }


def run_pareto_optimization(run_date, lenses, name1, name2, n_calls=50, n_rays=1000):
    if not SKOPT_AVAILABLE:
        raise ImportError("scikit-optimize not installed. Run: pip install scikit-optimize")
    
    d1, d2 = lenses[name1], lenses[name2]
    f1, f2 = d1['f_mm'], d2['f_mm']
    origins, dirs = sample_rays(n_rays)
    
    space = [
        Real(C.SOURCE_TO_LENS_OFFSET, f1 * 1.5, name='z_l1'),
        Real(C.SOURCE_TO_LENS_OFFSET + 1.0, f1 * 1.5 + f2 * 2.5, name='z_l2')
    ]
    
    @use_named_args(space)
    def objective(z_l1, z_l2):
        if z_l2 <= z_l1 + 0.1:
            return 1e6
        
        z_fiber = z_l2 + f2
        lens1 = PlanoConvex(z_l1, d1['R_mm'], d1['t_mm'], d1['dia']/2.0)
        lens2 = PlanoConvex(z_l2, d2['R_mm'], d2['t_mm'], d2['dia']/2.0)
        
        accepted = trace_system(origins, dirs, lens1, lens2, z_fiber, 
                               C.FIBER_CORE_DIAM_MM/2.0, C.ACCEPTANCE_HALF_RAD)
        coupling = np.count_nonzero(accepted) / n_rays
        normalized_length = z_fiber / 200.0
        return alpha * (1 - coupling) + (1 - alpha) * normalized_length
    
    result = gp_minimize(objective, space, n_calls=n_calls, random_state=42, 
                        verbose=False, n_initial_points=10)
    
    z_l1_opt, z_l2_opt = result.x
    z_fiber_opt = z_l2_opt + f2
    
    origins_final, dirs_final = sample_rays(2000)
    lens1 = PlanoConvex(z_l1_opt, d1['R_mm'], d1['t_mm'], d1['dia']/2.0)
    lens2 = PlanoConvex(z_l2_opt, d2['R_mm'], d2['t_mm'], d2['dia']/2.0)
    accepted = trace_system(origins_final, dirs_final, lens1, lens2, z_fiber_opt,
                           C.FIBER_CORE_DIAM_MM/2.0, C.ACCEPTANCE_HALF_RAD)
    
    return {
        'lens1': name1, 'lens2': name2, 'f1_mm': f1, 'f2_mm': f2,
        'z_l1': z_l1_opt, 'z_l2': z_l2_opt, 'z_fiber': z_fiber_opt,
        'total_len_mm': z_fiber_opt, 'coupling': np.count_nonzero(accepted) / 2000,
        'origins': origins_final, 'dirs': dirs_final, 'accepted': accepted
    }


def run_pareto_optimization(run_date, lenses, name1, name2, n_calls=50, n_rays=1000):
    if not SKOPT_AVAILABLE:
        raise ImportError("scikit-optimize not installed. Run: pip install scikit-optimize")
    
    d1, d2 = lenses[name1], lenses[name2]
    f1, f2 = d1['f_mm'], d2['f_mm']
    origins, dirs = sample_rays(n_rays)
    
    space = [
        Real(C.SOURCE_TO_LENS_OFFSET, f1 * 1.5, name='z_l1'),
        Real(C.SOURCE_TO_LENS_OFFSET + 1.0, f1 * 1.5 + f2 * 2.5, name='z_l2')
    ]
    
    all_results = []
    
    @use_named_args(space)
    def objective(z_l1, z_l2):
        if z_l2 <= z_l1 + 0.1:
            return 1e6
        
        z_fiber = z_l2 + f2
        lens1 = PlanoConvex(z_l1, d1['R_mm'], d1['t_mm'], d1['dia']/2.0)
        lens2 = PlanoConvex(z_l2, d2['R_mm'], d2['t_mm'], d2['dia']/2.0)
        accepted = trace_system(origins, dirs, lens1, lens2, z_fiber,
                               C.FIBER_CORE_DIAM_MM/2.0, C.ACCEPTANCE_HALF_RAD)
        coupling = np.count_nonzero(accepted) / n_rays
        all_results.append({'z_l1': z_l1, 'z_l2': z_l2, 'z_fiber': z_fiber,
                           'coupling': coupling, 'total_len_mm': z_fiber})
        return 1 - coupling
    
    gp_minimize(objective, space, n_calls=n_calls, random_state=42, verbose=False)
    
    pareto_front = []
    for r1 in all_results:
        is_pareto = True
        for r2 in all_results:
            if (r2['coupling'] >= r1['coupling'] and r2['total_len_mm'] <= r1['total_len_mm'] 
                and (r2['coupling'] > r1['coupling'] or r2['total_len_mm'] < r1['total_len_mm'])):
                is_pareto = False
                break
        if is_pareto:
            r1.update({'lens1': name1, 'lens2': name2, 'f1_mm': f1, 'f2_mm': f2})
            pareto_front.append(r1)
    
    return pareto_front
