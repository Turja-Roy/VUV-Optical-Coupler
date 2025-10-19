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


def analyze_combos(results_file, coupling_threshold, lenses, run_id, alpha=0.7, n_rays=1000, medium='air'):
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
                    res = optimizer.optimize(lenses, lens1, lens2, n_rays, alpha, medium)
                elif method == 'dual_annealing':
                    from scripts.optimization import dual_annealing as optimizer
                    res = optimizer.optimize(lenses, lens1, lens2, n_rays, alpha, medium)
                elif method == 'nelder_mead':
                    from scripts.optimization import nelder_mead as optimizer
                    res = optimizer.optimize(lenses, lens1, lens2, n_rays, alpha, medium)
                elif method == 'powell':
                    from scripts.optimization import powell as optimizer
                    res = optimizer.optimize(lenses, lens1, lens2, n_rays, alpha, medium)
                elif method == 'grid_search':
                    from scripts.optimization import grid_search as optimizer
                    res = optimizer.run_grid(run_id, lenses, lens1, lens2, medium=medium)
                elif method == 'bayesian':
                    from scripts.optimization import bayesian as optimizer
                    res = optimizer.optimize(lenses, lens1, lens2, n_calls=50, n_rays=n_rays, alpha=alpha, medium=medium)
                
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


def wavelength_analysis(results_file, run_id, wl_start=180, wl_end=300, wl_step=10, n_rays=1000, alpha=0.7, medium='air'):
    logger = _setup_logger(run_id)
    
    logger.info(f"Loading lens combinations from {results_file}")
    print(f"\nLoading lens combinations from {results_file}...")
    
    df = pd.read_csv(results_file)
    
    required_cols = ['lens1', 'lens2']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        error_msg = f"Missing required columns: {', '.join(missing_cols)}"
        logger.error(error_msg)
        print(f"Error: {error_msg}")
        return
    
    lens_combos = df[['lens1', 'lens2']].drop_duplicates()
    print(f"Found {len(lens_combos)} lens combinations")
    logger.info(f"Found {len(lens_combos)} lens combinations")
    
    from scripts.data_io import find_combos
    _, lenses = find_combos('combine')
    
    wavelengths = np.arange(wl_start, wl_end + 1, wl_step)
    methods = ['differential_evolution', 'dual_annealing', 'nelder_mead', 'powell', 'grid_search', 'bayesian']
    
    print(f"Parameters: wavelengths={wl_start}-{wl_end}nm (step={wl_step}nm), n_rays={n_rays}, alpha={alpha}, medium={medium}")
    logger.info(f"Parameters: wavelengths={wl_start}-{wl_end}nm (step={wl_step}nm), n_rays={n_rays}, alpha={alpha}, medium={medium}")
    
    try:
        from scripts.optimization import bayesian
    except ImportError:
        print("Warning: Bayesian optimization not available (scikit-optimize not installed)")
        methods.remove('bayesian')
    
    results_dir = Path('./results') / run_id
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nRunning wavelength analysis from {wavelengths[0]} to {wavelengths[-1]} nm...")
    print(f"Total tasks: {len(lens_combos)} combos × {len(methods)} methods × {len(wavelengths)} wavelengths = {len(lens_combos) * len(methods) * len(wavelengths)} optimizations")
    logger.info(f"Wavelength range: {wavelengths[0]}-{wavelengths[-1]} nm in {len(wavelengths)} steps")
    logger.info(f"Total optimizations: {len(lens_combos) * len(methods) * len(wavelengths)}")
    
    for idx, row in tqdm(lens_combos.iterrows(), total=len(lens_combos), desc="Lens combinations"):
        lens1, lens2 = row['lens1'], row['lens2']
        combo_id = f"{lens1}+{lens2}_{medium}"
        
        print(f"\nAnalyzing {combo_id}")
        logger.info(f"Starting wavelength analysis for {combo_id}")
        
        temp_file = results_dir / f"temp_batch_wavelength_{combo_id}.json"
        completed_runs = set()
        
        if temp_file.exists():
            print(f"  Found existing progress, resuming...")
            logger.info(f"Resuming from temp file: {temp_file}")
            import json
            try:
                with open(temp_file, 'r') as f:
                    existing_data = json.load(f)
                    for entry in existing_data:
                        key = (entry['method'], entry['wavelength_nm'])
                        completed_runs.add(key)
                print(f"  Skipping {len(completed_runs)} already-completed runs")
                logger.info(f"Found {len(completed_runs)} completed runs")
            except json.JSONDecodeError:
                logger.warning("Could not parse temp file, starting fresh")
        
        for method in methods:
            print(f"  Running {method}...")
            logger.info(f"  Method: {method}")
            
            for wavelength in tqdm(wavelengths, desc=f"  {method}", leave=False):
                if (method, float(wavelength)) in completed_runs:
                    continue
                
                import scripts.consts as C
                original_wavelength = C.WAVELENGTH_NM
                C.WAVELENGTH_NM = wavelength
                
                try:
                    res = None
                    if method == 'differential_evolution':
                        from scripts.optimization import differential_evolution as optimizer
                        res = optimizer.optimize(lenses, lens1, lens2, n_rays=n_rays, alpha=alpha, medium=medium)
                    elif method == 'dual_annealing':
                        from scripts.optimization import dual_annealing as optimizer
                        res = optimizer.optimize(lenses, lens1, lens2, n_rays=n_rays, alpha=alpha, medium=medium)
                    elif method == 'nelder_mead':
                        from scripts.optimization import nelder_mead as optimizer
                        res = optimizer.optimize(lenses, lens1, lens2, n_rays=n_rays, alpha=alpha, medium=medium)
                    elif method == 'powell':
                        from scripts.optimization import powell as optimizer
                        res = optimizer.optimize(lenses, lens1, lens2, n_rays=n_rays, alpha=alpha, medium=medium)
                    elif method == 'grid_search':
                        from scripts.optimization import grid_search as optimizer
                        res = optimizer.run_grid(run_id, lenses, lens1, lens2, medium=medium)
                    elif method == 'bayesian':
                        from scripts.optimization import bayesian as optimizer
                        res = optimizer.optimize(lenses, lens1, lens2, n_calls=50, n_rays=n_rays, alpha=alpha, medium=medium)
                    
                    if res and res['coupling'] > 0:
                        result_entry = {
                            'lens1': lens1,
                            'lens2': lens2,
                            'method': method,
                            'wavelength_nm': float(wavelength),
                            'coupling': float(res['coupling']),
                            'total_len_mm': float(res['total_len_mm']),
                            'z_l1': float(res['z_l1']),
                            'z_l2': float(res['z_l2']),
                            'z_fiber': float(res['z_fiber']),
                            'medium': medium
                        }
                        
                        write_temp(result_entry, run_id, f'wavelength_{combo_id}')
                        logger.info(f"    {wavelength}nm: coupling={res['coupling']:.4f}")
                    else:
                        logger.warning(f"    {wavelength}nm: optimization failed")
                    
                except Exception as e:
                    logger.error(f"    {wavelength}nm: Error - {str(e)}")
                
                finally:
                    C.WAVELENGTH_NM = original_wavelength
        
        if temp_file.exists():
            import json
            with open(temp_file, 'r') as f:
                all_wavelength_data = json.load(f)
            
            if all_wavelength_data:
                combo_df = pd.DataFrame(all_wavelength_data)
                combo_filename = f"{combo_id}_wavelength.csv"
                combo_df.to_csv(results_dir / combo_filename, index=False)
                logger.info(f"Saved data to {combo_filename}")
                print(f"  Saved {len(all_wavelength_data)} results to {combo_filename}")
                
                temp_file.unlink()
                logger.info(f"Deleted temp file: {temp_file}")
    
    print(f"\n{'='*60}")
    print(f"Wavelength analysis complete!")
    print(f"Results saved to: results/{run_id}/")
    print(f"{'='*60}")
    
    logger.info("Wavelength analysis complete")

