import pandas as pd
import numpy as np
from pathlib import Path
import logging
import time
from tqdm import tqdm

from scripts.data_io import write_temp
from scripts.visualizers import plot_system_rays


def _setup_logger(run_id):
    logger = logging.getLogger("raytrace_analyze")
    
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


def analyze_combos(results_file, coupling_threshold, lenses, run_id, alpha=0.7, n_rays=1000):
    logger = _setup_logger(run_id)
    
    logger.info(f"Loading results from {results_file}")
    print(f"\nLoading results from {results_file}...")
    
    df = pd.read_csv(results_file)
    
    filtered = df[df['coupling'] >= coupling_threshold]
    
    print(f"Found {len(filtered)} combinations with coupling >= {coupling_threshold}")
    logger.info(f"Found {len(filtered)} combinations with coupling >= {coupling_threshold}")
    
    if len(filtered) == 0:
        print("No combinations meet the threshold. Exiting.")
        return {}
    
    methods = ['differential_evolution', 'dual_annealing', 'nelder_mead', 'powell', 'grid_search', 'bayesian']
    
    try:
        from scripts.optimization import bayesian
    except ImportError:
        print("Warning: Bayesian optimization not available (scikit-optimize not installed)")
        methods.remove('bayesian')
    
    all_results = {method: [] for method in methods}
    
    for method in methods:
        print(f"\n{'='*60}")
        print(f"Running {method} optimization on {len(filtered)} combinations")
        print(f"{'='*60}")
        logger.info(f"\nStarting {method} optimization on {len(filtered)} combinations")
        
        for idx, row in tqdm(filtered.iterrows(), total=len(filtered), desc=f"{method}"):
            lens1, lens2 = row['lens1'], row['lens2']
            
            logger.info(f"Optimizing {lens1} + {lens2} using {method}")
            
            try:
                start_time = time.time()
                
                res = None
                if method == 'differential_evolution':
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
                elif method == 'grid_search':
                    from scripts.optimization import grid_search as optimizer
                    res = optimizer.run_grid(run_id, lenses, lens1, lens2)
                elif method == 'bayesian':
                    from scripts.optimization import bayesian as optimizer
                    res = optimizer.optimize(lenses, lens1, lens2, n_calls=50, n_rays=n_rays, alpha=alpha)
                
                elapsed = time.time() - start_time
                
                if res is None:
                    logger.warning(f"Optimization failed for {lens1} + {lens2}")
                    continue
                
                res['method'] = method
                res['time_seconds'] = elapsed
                
                write_temp(res, run_id, f'{method}_batch')
                plot_system_rays(lenses, res, run_id, method=method)
                
                logger.info(f"Coupling={res['coupling']:.4f}, "
                           f"Length={res['total_len_mm']:.2f}mm, "
                           f"Time={elapsed:.2f}s")
                
            except Exception as e:
                logger.error(f"Error optimizing {lens1} + {lens2} with {method}: {str(e)}")
                continue
        
        method_results = []
        filepath = f'./results/{run_id}/temp_batch_{method}_batch.json'
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
                method_results.extend(data)
            Path(filepath).unlink()
        
        all_results[method] = method_results
        
        print(f"Completed {method}: {len(method_results)} successful optimizations")
        logger.info(f"Completed {method}: {len(method_results)} successful optimizations")
    
    print(f"\n{'='*60}")
    print("Generating combined method comparison plots...")
    print(f"{'='*60}")
    logger.info("Generating combined method comparison plots")
    
    lens_combos = {}
    for method, results in all_results.items():
        for res in results:
            combo_key = (res['lens1'], res['lens2'])
            if combo_key not in lens_combos:
                lens_combos[combo_key] = {}
            lens_combos[combo_key][method] = res
    
    from scripts.visualizers import plot_combined_methods
    for (lens1, lens2), methods_dict in lens_combos.items():
        logger.info(f"Creating combined plot for {lens1} + {lens2} ({len(methods_dict)} methods)")
        plot_combined_methods(lenses, methods_dict, lens1, lens2, run_id)
    
    print(f"Generated {len(lens_combos)} combined plots")
    logger.info(f"Generated {len(lens_combos)} combined plots")
    
    print(f"\n{'='*60}")
    print("Writing results CSV files by lens combination...")
    print(f"{'='*60}")
    logger.info("Writing results CSV files by lens combination")
    
    results_dir = Path('./results') / run_id
    results_dir.mkdir(parents=True, exist_ok=True)
    
    for (lens1, lens2), methods_dict in lens_combos.items():
        rows = []
        for method, res in methods_dict.items():
            row = {
                'lens1': res['lens1'],
                'lens2': res['lens2'],
                'method': method,
                'coupling': res['coupling'],
                'total_len_mm': res['total_len_mm'],
                'z_l1': res['z_l1'],
                'z_l2': res['z_l2'],
                'z_fiber': res['z_fiber'],
                'f1_mm': res['f1_mm'],
                'f2_mm': res['f2_mm'],
                'time_seconds': res.get('time_seconds', 0.0)
            }
            rows.append(row)
        
        df = pd.DataFrame(rows).sort_values('coupling', ascending=False)
        filename = f"{lens1}+{lens2}.csv"
        df.to_csv(results_dir / filename, index=False)
        logger.info(f"Wrote {filename} with {len(rows)} methods")
    
    print(f"Wrote {len(lens_combos)} CSV files")
    logger.info(f"Wrote {len(lens_combos)} CSV files")
    
    return all_results
