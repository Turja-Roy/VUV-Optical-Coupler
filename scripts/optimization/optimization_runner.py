import numpy as np
from tqdm import tqdm
import logging
from pathlib import Path

from scripts.data_io import write_temp
from scripts.visualizers import plot_system_rays


def _setup_logger(run_id):
    logger = logging.getLogger("raytrace")
    
    logs_dir = Path.cwd() / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    logfile = logs_dir / f"{run_id}.log"
    
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


def run_combos(lenses, combos, run_id, method='differential_evolution',
               alpha=0.7, n_rays=1000, batch_num=None):
    logger = _setup_logger(run_id)
    
    if method == 'grid_search':
        from scripts.optimization import grid_search as optimizer
        optimize_func = lambda l, n1, n2: optimizer.run_grid(run_id, l, n1, n2)
    elif method == 'differential_evolution':
        from scripts.optimization import differential_evolution as optimizer
        optimize_func = lambda l, n1, n2: optimizer.optimize(l, n1, n2, n_rays, alpha)
    elif method == 'dual_annealing':
        from scripts.optimization import dual_annealing as optimizer
        optimize_func = lambda l, n1, n2: optimizer.optimize(l, n1, n2, n_rays, alpha)
    elif method == 'nelder_mead':
        from scripts.optimization import nelder_mead as optimizer
        optimize_func = lambda l, n1, n2: optimizer.optimize(l, n1, n2, n_rays, alpha)
    elif method == 'powell':
        from scripts.optimization import powell as optimizer
        optimize_func = lambda l, n1, n2: optimizer.optimize(l, n1, n2, n_rays, alpha)
    elif method == 'bayesian':
        from scripts.optimization import bayesian as optimizer
        optimize_func = lambda l, n1, n2: optimizer.optimize(l, n1, n2, n_calls=50, n_rays=n_rays, alpha=alpha)
    else:
        raise ValueError(f"Unknown optimization method: {method}")
    
    for (a, b) in tqdm(combos, desc=f"Optimizing with {method}"):
        logger.info(f"\nOptimizing {a} + {b} using {method}...")
        
        try:
            res = optimize_func(lenses, a, b)
            
            if res is None:
                logger.warning("Optimization failed or invalid configuration.")
                continue
            
            plot_system_rays(lenses, res, run_id)
            write_temp(res, run_id, batch_num)
            
            logger.info(f"Coupling={res['coupling']:.4f}, "
                       f"Length={res['total_len_mm']:.2f}mm, "
                       f"z_l1={res['z_l1']:.2f}, z_l2={res['z_l2']:.2f}")
            
        except Exception as e:
            logger.error(f"Error optimizing {a} + {b}: {str(e)}")
            continue
    
    results = []
    filename = 'temp.json' if batch_num is None else f'temp_batch_{batch_num}.json'
    filepath = f'./results/{run_id}/{filename}'
    
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


def compare_optimizers(lenses, test_combo, run_id, n_rays=1000, alpha=0.7):
    methods = ['differential_evolution', 'dual_annealing', 'nelder_mead', 'powell', 'grid_search', 'bayesian']
    results = {}
    
    lens1, lens2 = test_combo
    print(f"\nComparing optimization methods for {lens1} + {lens2}")
    print("="*60)
    
    for method in methods:
        print(f"\nTesting {method}...")
        try:
            import time
            start = time.time()
            
            res = None
            if method == 'grid_search':
                from scripts.optimization import grid_search as optimizer
                res = optimizer.run_grid(run_id, lenses, lens1, lens2)
            elif method == 'differential_evolution':
                from scripts.optimization import differential_evolution as optimizer
                res = optimizer.optimize(lenses, lens1, lens2, n_rays, alpha)
            elif method == 'dual_annealing':
                from scripts.optimization import dual_annealing as optimizer
                res = optimizer.optimize(lenses, lens1, lens2, n_rays, alpha)
            elif method == 'nelder_mead':
                from scripts.optimization import nelder_mead as optimizer
                res = optimizer.optimize(lenses, lens1, lens2, n_rays, alpha)
            elif method == 'powell':
                from scripts.optimization import powell as optimizer
                res = optimizer.optimize(lenses, lens1, lens2, n_rays, alpha)
            elif method == 'bayesian':
                from scripts.optimization import bayesian as optimizer
                res = optimizer.optimize(lenses, lens1, lens2, n_calls=30, n_rays=n_rays, alpha=alpha)
            
            elapsed = time.time() - start
            
            if res is not None:
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
            else:
                results[method] = None
            
        except ImportError:
            if method == 'bayesian':
                print("  Bayesian optimization not available (install scikit-optimize)")
            results[method] = None
        except Exception as e:
            print(f"  Error: {str(e)}")
            results[method] = None
    
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
