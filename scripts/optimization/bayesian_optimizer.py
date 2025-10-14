"""
Bayesian Optimization using Gaussian Processes.
This is more sample-efficient than grid search and can model the objective function.

Requires: scikit-optimize (skopt)
Install: pip install scikit-optimize
"""

import numpy as np
from tqdm import tqdm
from .PlanoConvex import PlanoConvex
from . import consts as C
from .raytrace_helpers import sample_rays, trace_system

try:
    from skopt import gp_minimize
    from skopt.space import Real
    from skopt.utils import use_named_args
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False


def run_bayesian_optimization(run_date, lenses, name1, name2, 
                               n_calls=50, n_rays=1000, alpha=0.7):
    """
    Run Bayesian Optimization for a lens pair.
    
    Parameters:
    - n_calls: number of evaluations (much smaller than grid search!)
    - n_rays: rays per evaluation
    - alpha: weight for coupling vs. length
    
    Returns:
    - result dictionary with optimal configuration
    """
    if not SKOPT_AVAILABLE:
        raise ImportError(
            "scikit-optimize not installed. Run: pip install scikit-optimize"
        )
    
    d1 = lenses[name1]
    d2 = lenses[name2]
    f1 = d1['f_mm']
    f2 = d2['f_mm']
    
    # Generate ray set once
    origins, dirs = sample_rays(n_rays)
    
    # Define search space
    space = [
        Real(C.SOURCE_TO_LENS_OFFSET, f1 * 1.5, name='z_l1'),
        Real(C.SOURCE_TO_LENS_OFFSET + f2 * 0.5, f1 * 1.5 + f2 * 2.5, name='z_l2')
    ]
    
    # Objective function
    @use_named_args(space)
    def objective(z_l1, z_l2):
        # Constraints
        if z_l2 <= z_l1 + 0.1:
            return 1e6
        
        z_fiber = z_l2 + f2
        
        lens1 = PlanoConvex(vertex_z_front=z_l1,
                           R_front_mm=d1['R_mm'],
                           thickness_mm=d1['t_mm'],
                           ap_rad_mm=d1['dia']/2.0)
        lens2 = PlanoConvex(vertex_z_front=z_l2,
                           R_front_mm=d2['R_mm'],
                           thickness_mm=d2['t_mm'],
                           ap_rad_mm=d2['dia']/2.0)
        
        accepted = trace_system(origins, dirs, lens1, lens2,
                               z_fiber, C.FIBER_CORE_DIAM_MM/2.0,
                               C.ACCEPTANCE_HALF_RAD)
        coupling = np.count_nonzero(accepted) / n_rays
        
        normalized_length = z_fiber / 200.0
        objective_val = alpha * (1 - coupling) + (1 - alpha) * normalized_length
        
        return objective_val
    
    # Run optimization
    result = gp_minimize(
        objective,
        space,
        n_calls=n_calls,
        random_state=42,
        verbose=False,
        n_initial_points=10
    )
    
    z_l1_opt, z_l2_opt = result.x
    z_fiber_opt = z_l2_opt + f2
    
    # Final evaluation with more rays
    origins_final, dirs_final = sample_rays(2000)
    lens1 = PlanoConvex(vertex_z_front=z_l1_opt,
                        R_front_mm=d1['R_mm'],
                        thickness_mm=d1['t_mm'],
                        ap_rad_mm=d1['dia']/2.0)
    lens2 = PlanoConvex(vertex_z_front=z_l2_opt,
                        R_front_mm=d2['R_mm'],
                        thickness_mm=d2['t_mm'],
                        ap_rad_mm=d2['dia']/2.0)
    
    accepted = trace_system(origins_final, dirs_final, lens1, lens2,
                           z_fiber_opt, C.FIBER_CORE_DIAM_MM/2.0,
                           C.ACCEPTANCE_HALF_RAD)
    coupling = np.count_nonzero(accepted) / 2000
    
    return {
        'lens1': name1,
        'lens2': name2,
        'f1_mm': f1,
        'f2_mm': f2,
        'z_l1': z_l1_opt,
        'z_l2': z_l2_opt,
        'z_fiber': z_fiber_opt,
        'total_len_mm': z_fiber_opt,
        'coupling': coupling,
        'origins': origins_final,
        'dirs': dirs_final,
        'accepted': accepted
    }


def run_pareto_optimization(run_date, lenses, name1, name2, 
                            n_calls=50, n_rays=1000):
    """
    Multi-objective optimization to find Pareto front.
    Returns multiple solutions trading off coupling vs. length.
    """
    if not SKOPT_AVAILABLE:
        raise ImportError(
            "scikit-optimize not installed. Run: pip install scikit-optimize"
        )
    
    d1 = lenses[name1]
    d2 = lenses[name2]
    f1 = d1['f_mm']
    f2 = d2['f_mm']
    
    origins, dirs = sample_rays(n_rays)
    
    space = [
        Real(C.SOURCE_TO_LENS_OFFSET, f1 * 1.5, name='z_l1'),
        Real(C.SOURCE_TO_LENS_OFFSET + f2 * 0.5, f1 * 1.5 + f2 * 2.5, name='z_l2')
    ]
    
    # Store all evaluated points
    all_results = []
    
    @use_named_args(space)
    def objective(z_l1, z_l2):
        if z_l2 <= z_l1 + 0.1:
            return 1e6
        
        z_fiber = z_l2 + f2
        
        lens1 = PlanoConvex(vertex_z_front=z_l1,
                           R_front_mm=d1['R_mm'],
                           thickness_mm=d1['t_mm'],
                           ap_rad_mm=d1['dia']/2.0)
        lens2 = PlanoConvex(vertex_z_front=z_l2,
                           R_front_mm=d2['R_mm'],
                           thickness_mm=d2['t_mm'],
                           ap_rad_mm=d2['dia']/2.0)
        
        accepted = trace_system(origins, dirs, lens1, lens2,
                               z_fiber, C.FIBER_CORE_DIAM_MM/2.0,
                               C.ACCEPTANCE_HALF_RAD)
        coupling = np.count_nonzero(accepted) / n_rays
        
        all_results.append({
            'z_l1': z_l1,
            'z_l2': z_l2,
            'z_fiber': z_fiber,
            'coupling': coupling,
            'total_len_mm': z_fiber
        })
        
        # Optimize for coupling primarily
        return 1 - coupling
    
    # Run optimization
    result = gp_minimize(
        objective,
        space,
        n_calls=n_calls,
        random_state=42,
        verbose=False
    )
    
    # Find Pareto front from all evaluated points
    pareto_front = []
    for i, r1 in enumerate(all_results):
        is_pareto = True
        for r2 in all_results:
            # r2 dominates r1 if it's better in both objectives
            if (r2['coupling'] >= r1['coupling'] and 
                r2['total_len_mm'] <= r1['total_len_mm'] and
                (r2['coupling'] > r1['coupling'] or 
                 r2['total_len_mm'] < r1['total_len_mm'])):
                is_pareto = False
                break
        if is_pareto:
            pareto_front.append(r1)
    
    # Add metadata to Pareto solutions
    for solution in pareto_front:
        solution['lens1'] = name1
        solution['lens2'] = name2
        solution['f1_mm'] = f1
        solution['f2_mm'] = f2
    
    return pareto_front
