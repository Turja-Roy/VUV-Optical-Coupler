"""
Unified runner for all optimization methods.
This replaces the grid search approach with more efficient optimizers.
"""

import numpy as np
from tqdm import tqdm
import logging
from pathlib import Path

from scripts.fetcher import write_temp
from scripts.visualizers import plot_system_rays


def _setup_logger(run_date: str):
    """Setup logger for optimization runs."""
    logger = logging.getLogger("raytrace")
    
    logs_dir = Path.cwd() / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    logfile = logs_dir / f"run_{run_date}.log"
    
    for h in logger.handlers:
        if isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", None) == str(logfile):
            return logger
    
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(logfile, encoding="utf-8")
    fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s",
                           datefmt="%Y-%m-%d %H:%M:%S")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    
    return logger


def run_combos_optimized(lenses, combos, run_date, method='differential_evolution',
                         alpha=0.7, n_rays=1000, batch_num=None):
    """
    Run optimization for all lens combinations.
    
    Parameters:
    - lenses: dictionary of lens data
    - combos: list of (lens1, lens2) tuples
    - run_date: date string for file naming
    - method: optimization method to use
        - 'differential_evolution': robust global optimizer (recommended)
        - 'dual_annealing': simulated annealing + local search
        - 'nelder_mead': fast local search
        - 'powell': another local search method
        - 'bayesian': Bayesian optimization (requires scikit-optimize)
    - alpha: weight for coupling vs. length (0.7 = 70% coupling, 30% length)
    - n_rays: number of rays for optimization
    - batch_num: batch number for file naming
    
    Returns:
    - list of result dictionaries
    """
    logger = _setup_logger(run_date)
    
    # Import the appropriate optimizer
    if method == 'bayesian':
        from .bayesian_optimizer import run_bayesian_optimization
        optimizer_func = lambda rd, l, a, b: run_bayesian_optimization(
            rd, l, a, b, n_calls=50, n_rays=n_rays, alpha=alpha
        )
    else:
        from .scipy_optimizer import run_optimization
        optimizer_func = lambda rd, l, a, b: run_optimization(
            rd, l, a, b, method=method, n_rays=n_rays, alpha=alpha
        )
    
    for (a, b) in tqdm(combos, desc=f"Optimizing with {method}"):
        logger.info(f"\nOptimizing {a} + {b} using {method}...")
        
        try:
            res = optimizer_func(run_date, lenses, a, b)
            
            plot_system_rays(lenses, res, run_date)
            write_temp(res, run_date, batch_num)
            
            logger.info(f"Coupling={res['coupling']:.4f}, "
                       f"Length={res['total_len_mm']:.2f}mm, "
                       f"z_l1={res['z_l1']:.2f}, z_l2={res['z_l2']:.2f}")
            
        except Exception as e:
            logger.error(f"Error optimizing {a} + {b}: {str(e)}")
            continue
    
    # Read results from temp file
    results = []
    filename = 'temp.json' if batch_num is None else f'temp_batch_{batch_num}.json'
    filepath = f'./results/{run_date}/{filename}'
    
    try:
        if Path(filepath).exists():
            import json
            with open(filepath, 'r') as f:
                data = json.load(f)
                for result in data:
                    if 'origins' in result:
                        result['origins'] = np.array(result['origins'])
                    if 'dirs' in result:
                        result['dirs'] = np.array(result['dirs'])
                    if 'accepted' in result:
                        result['accepted'] = np.array(result['accepted'])
                results.extend(data)
            Path(filepath).unlink()
    except Exception as e:
        logger.error(f"Error reading temporary file {filename}: {str(e)}")
    
    return results


def compare_optimizers(lenses, test_combo, run_date, n_rays=1000, alpha=0.7):
    """
    Compare different optimization methods on a single lens combination.
    Useful for determining which method works best.
    
    Parameters:
    - lenses: lens dictionary
    - test_combo: tuple of (lens1, lens2) to test
    - run_date: date string
    - n_rays: rays per evaluation
    - alpha: coupling vs. length weight
    
    Returns:
    - dictionary with results from each method
    """
    from .scipy_optimizer import run_optimization
    
    methods = ['differential_evolution', 'dual_annealing', 'nelder_mead', 'powell']
    results = {}
    
    lens1, lens2 = test_combo
    print(f"\nComparing optimization methods for {lens1} + {lens2}")
    print("="*60)
    
    for method in methods:
        print(f"\nTesting {method}...")
        try:
            import time
            start = time.time()
            
            res = run_optimization(run_date, lenses, lens1, lens2,
                                  method=method, n_rays=n_rays, alpha=alpha)
            
            elapsed = time.time() - start
            
            results[method] = {
                'coupling': res['coupling'],
                'total_len_mm': res['total_len_mm'],
                'z_l1': res['z_l1'],
                'z_l2': res['z_l2'],
                'time_seconds': elapsed
            }
            
            print(f"  Coupling: {res['coupling']:.4f}")
            print(f"  Length: {res['total_len_mm']:.2f} mm")
            print(f"  Time: {elapsed:.2f} seconds")
            
        except Exception as e:
            print(f"  Error: {str(e)}")
            results[method] = None
    
    # Try Bayesian if available
    try:
        from .bayesian_optimizer import run_bayesian_optimization
        print(f"\nTesting bayesian optimization...")
        import time
        start = time.time()
        
        res = run_bayesian_optimization(run_date, lenses, lens1, lens2,
                                       n_calls=30, n_rays=n_rays, alpha=alpha)
        elapsed = time.time() - start
        
        results['bayesian'] = {
            'coupling': res['coupling'],
            'total_len_mm': res['total_len_mm'],
            'z_l1': res['z_l1'],
            'z_l2': res['z_l2'],
            'time_seconds': elapsed
        }
        
        print(f"  Coupling: {res['coupling']:.4f}")
        print(f"  Length: {res['total_len_mm']:.2f} mm")
        print(f"  Time: {elapsed:.2f} seconds")
        
    except ImportError:
        print("\nBayesian optimization not available (install scikit-optimize)")
    except Exception as e:
        print(f"  Error: {str(e)}")
    
    print("\n" + "="*60)
    print("Summary:")
    print("-"*60)
    
    valid_results = {k: v for k, v in results.items() if v is not None}
    if valid_results:
        best_coupling = max(valid_results.items(), key=lambda x: x[1]['coupling'])
        best_length = min(valid_results.items(), key=lambda x: x[1]['total_len_mm'])
        fastest = min(valid_results.items(), key=lambda x: x[1]['time_seconds'])
        
        print(f"Best coupling: {best_coupling[0]} ({best_coupling[1]['coupling']:.4f})")
        print(f"Shortest length: {best_length[0]} ({best_length[1]['total_len_mm']:.2f} mm)")
        print(f"Fastest: {fastest[0]} ({fastest[1]['time_seconds']:.2f} s)")
    
    return results
