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


def analyze_combos(results_file, coupling_threshold, lenses, run_id, alpha=0.7, n_rays=1000, medium='air', methods=None, plot_style='3d', orientation_mode='both'):
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
    
    # Use provided methods list or default to all methods
    if methods is None:
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
                    res = optimizer.optimize(lenses, lens1, lens2, n_rays, alpha, medium, orientation_mode=orientation_mode)
                elif method == 'dual_annealing':
                    from scripts.optimization import dual_annealing as optimizer
                    res = optimizer.optimize(lenses, lens1, lens2, n_rays, alpha, medium, orientation_mode=orientation_mode)
                elif method == 'nelder_mead':
                    from scripts.optimization import nelder_mead as optimizer
                    res = optimizer.optimize(lenses, lens1, lens2, n_rays, alpha, medium, orientation_mode=orientation_mode)
                elif method == 'powell':
                    from scripts.optimization import powell as optimizer
                    res = optimizer.optimize(lenses, lens1, lens2, n_rays, alpha, medium, orientation_mode=orientation_mode)
                elif method == 'grid_search':
                    from scripts.optimization import grid_search as optimizer
                    res = optimizer.run_grid(run_id, lenses, lens1, lens2, medium=medium)
                elif method == 'bayesian':
                    from scripts.optimization import bayesian as optimizer
                    res = optimizer.optimize(lenses, lens1, lens2, n_calls=50, n_rays=n_rays, alpha=alpha, medium=medium, orientation_mode=orientation_mode)
                
                elapsed = time.time() - start_time
                
                if res is None:
                    logger.warning(f"Optimization failed for {lens1} + {lens2}")
                    continue
                
                # Handle both list (orientation_mode='both') and single dict returns
                results_to_process = res if isinstance(res, list) else [res]
                
                for result in results_to_process:
                    result['method'] = method
                    result['time_seconds'] = elapsed
                    
                    write_temp(result, run_id, f'{method}_batch')
                    plot_system_rays(lenses, result, run_id, method=method, plot_style=plot_style)
                    
                    logger.info(f"Coupling={result['coupling']:.4f}, "
                               f"Length={result['total_len_mm']:.2f}mm, "
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
        plot_combined_methods(lenses, methods_dict, lens1, lens2, run_id, plot_style=plot_style)
    
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
                'orientation': res.get('orientation', 'ScffcF'),
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


def evaluate_fixed_config_at_wavelength(lenses, lens1, lens2, z_l1, z_l2, z_fiber, wavelength, n_rays=2000, medium='air', orientation='ScffcF'):
    from scripts.PlanoConvex import PlanoConvex
    from scripts.raytrace_helpers import sample_rays, trace_system
    from scripts import consts as C
    
    d1, d2 = lenses[lens1], lenses[lens2]
    
    # Parse orientation to determine flipped flags
    # ScffcF: lens1 curved-first (False), lens2 flat-first (True)
    # SfccfF: lens1 flat-first (True), lens2 curved-first (False)
    if orientation == 'SfccfF':
        flipped1, flipped2 = True, False
    else:  # Default to 'ScffcF'
        flipped1, flipped2 = False, True
    
    original_wavelength = C.WAVELENGTH_NM
    C.WAVELENGTH_NM = wavelength
    
    try:
        origins, dirs = sample_rays(n_rays)
        lens1_obj = PlanoConvex(z_l1, d1['R_mm'], d1['tc_mm'], d1['te_mm'], d1['dia']/2.0, flipped=flipped1)
        lens2_obj = PlanoConvex(z_l2, d2['R_mm'], d2['tc_mm'], d2['te_mm'], d2['dia']/2.0, flipped=flipped2)
        
        accepted, transmission = trace_system(origins, dirs, lens1_obj, lens2_obj,
                                z_fiber, C.FIBER_CORE_DIAM_MM/2.0, C.ACCEPTANCE_HALF_RAD,
                                medium, C.PRESSURE_ATM, C.TEMPERATURE_K, C.HUMIDITY_FRACTION)
        
        avg_transmission = np.mean(transmission[accepted]) if np.any(accepted) else 0.0
        coupling = (np.count_nonzero(accepted) / n_rays) * avg_transmission
        
        return coupling
    finally:
        C.WAVELENGTH_NM = original_wavelength


def wavelength_analysis(results_file, run_id, wl_start=180, wl_end=300, wl_step=10, n_rays=2000, alpha=0.7, medium='air', methods=None, orientation_mode='both'):
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
    
    # Use provided methods list or default to all methods
    if methods is None:
        methods = ['differential_evolution', 'dual_annealing', 'nelder_mead', 'powell', 'grid_search', 'bayesian']
    
    wavelengths = np.arange(wl_start, wl_end + 1, wl_step)
    
    print(f"Parameters: wavelengths={wl_start}-{wl_end}nm (step={wl_step}nm), n_rays={n_rays}, medium={medium}")
    print(f"Strategy: Calibrate at 200nm, then test fixed geometry across wavelengths")
    logger.info(f"Parameters: wavelengths={wl_start}-{wl_end}nm (step={wl_step}nm), n_rays={n_rays}, medium={medium}")
    logger.info(f"Strategy: Using fixed 200nm calibration across all wavelengths")
    
    try:
        from scripts.optimization import bayesian
    except ImportError:
        print("Warning: Bayesian optimization not available (scikit-optimize not installed)")
        methods.remove('bayesian')
    
    results_dir = Path('./results') / run_id
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nRunning wavelength analysis from {wavelengths[0]} to {wavelengths[-1]} nm...")
    print(f"Total tasks: {len(lens_combos)} combos × {len(methods)} methods = {len(lens_combos) * len(methods)} calibrations")
    print(f"             + {len(lens_combos)} combos × {len(methods)} methods × {len(wavelengths)} wavelengths = {len(lens_combos) * len(methods) * len(wavelengths)} evaluations")
    logger.info(f"Wavelength range: {wavelengths[0]}-{wavelengths[-1]} nm in {len(wavelengths)} steps")
    logger.info(f"Total calibrations: {len(lens_combos) * len(methods)}")
    logger.info(f"Total evaluations: {len(lens_combos) * len(methods) * len(wavelengths)}")
    
    for idx, row in tqdm(lens_combos.iterrows(), total=len(lens_combos), desc="Lens combinations"):
        lens1, lens2 = row['lens1'], row['lens2']
        combo_id = f"{lens1}+{lens2}_{medium}"
        
        print(f"\nAnalyzing {combo_id}")
        logger.info(f"Starting wavelength analysis for {combo_id}")
        
        temp_file = results_dir / f"temp_batch_wavelength_{combo_id}.json"
        calibrations = {}
        completed_wavelengths = {}
        
        if temp_file.exists():
            print(f"  Found existing progress, loading...")
            logger.info(f"Resuming from temp file: {temp_file}")
            import json
            try:
                with open(temp_file, 'r') as f:
                    existing_data = json.load(f)
                    for entry in existing_data:
                        method = entry['method']
                        wl = entry['wavelength_nm']
                        
                        if method not in calibrations:
                            calibrations[method] = {
                                'z_l1': entry['z_l1'],
                                'z_l2': entry['z_l2'],
                                'z_fiber': entry['z_fiber'],
                                'total_len_mm': entry['total_len_mm'],
                                'orientation': entry.get('orientation', 'ScffcF')
                            }
                        
                        if method not in completed_wavelengths:
                            completed_wavelengths[method] = set()
                        completed_wavelengths[method].add(wl)
                
                total_completed = sum(len(wls) for wls in completed_wavelengths.values())
                print(f"  Loaded {len(calibrations)} calibrations, {total_completed} completed evaluations")
                logger.info(f"Loaded {len(calibrations)} calibrations, {total_completed} completed evaluations")
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Could not parse temp file: {e}, starting fresh")
                calibrations = {}
                completed_wavelengths = {}
        
        for method in methods:
            if method in calibrations:
                print(f"  Skipping calibration for {method} (already done)")
                logger.info(f"  Skipping calibration for {method} (already done)")
                continue
            
            print(f"  Calibrating with {method} at 200nm...")
            logger.info(f"  Calibrating with {method} at 200nm")
            
            import scripts.consts as C
            original_wavelength = C.WAVELENGTH_NM
            C.WAVELENGTH_NM = 200.0
            
            try:
                res = None
                if method == 'differential_evolution':
                    from scripts.optimization import differential_evolution as optimizer
                    res = optimizer.optimize(lenses, lens1, lens2, n_rays=n_rays, alpha=alpha, medium=medium, orientation_mode=orientation_mode)
                elif method == 'dual_annealing':
                    from scripts.optimization import dual_annealing as optimizer
                    res = optimizer.optimize(lenses, lens1, lens2, n_rays=n_rays, alpha=alpha, medium=medium, orientation_mode=orientation_mode)
                elif method == 'nelder_mead':
                    from scripts.optimization import nelder_mead as optimizer
                    res = optimizer.optimize(lenses, lens1, lens2, n_rays=n_rays, alpha=alpha, medium=medium, orientation_mode=orientation_mode)
                elif method == 'powell':
                    from scripts.optimization import powell as optimizer
                    res = optimizer.optimize(lenses, lens1, lens2, n_rays=n_rays, alpha=alpha, medium=medium, orientation_mode=orientation_mode)
                elif method == 'grid_search':
                    from scripts.optimization import grid_search as optimizer
                    res = optimizer.run_grid(run_id, lenses, lens1, lens2, medium=medium)
                elif method == 'bayesian':
                    from scripts.optimization import bayesian as optimizer
                    res = optimizer.optimize(lenses, lens1, lens2, n_calls=50, n_rays=n_rays, alpha=alpha, medium=medium, orientation_mode=orientation_mode)
                
                # Handle both list (orientation_mode='both') and single dict returns
                # For wavelength analysis, we only use the first/best result
                if isinstance(res, list):
                    res = res[0] if res else None
                
                if res and res['coupling'] > 0:
                    calibrations[method] = {
                        'z_l1': res['z_l1'],
                        'z_l2': res['z_l2'],
                        'z_fiber': res['z_fiber'],
                        'total_len_mm': res['total_len_mm'],
                        'orientation': res.get('orientation', 'ScffcF')
                    }
                    logger.info(f"    Calibrated: z_l1={res['z_l1']:.2f}, z_l2={res['z_l2']:.2f}, z_fiber={res['z_fiber']:.2f}, coupling@200nm={res['coupling']:.4f}, orientation={res.get('orientation', 'ScffcF')}")
                else:
                    logger.warning(f"    Calibration failed for {method}")
                
            except Exception as e:
                logger.error(f"    Calibration error with {method}: {str(e)}")
            
            finally:
                C.WAVELENGTH_NM = original_wavelength
        
        print(f"  Calibrations ready: {len(calibrations)}/{len(methods)} methods")
        logger.info(f"  Calibrations ready: {len(calibrations)}/{len(methods)} methods")
        
        for method, calib in calibrations.items():
            completed_for_method = completed_wavelengths.get(method, set())
            remaining = [wl for wl in wavelengths if float(wl) not in completed_for_method]
            
            if len(remaining) == 0:
                print(f"  Skipping {method} (all wavelengths done)")
                logger.info(f"  Skipping {method} (all wavelengths done)")
                continue
            
            if len(remaining) < len(wavelengths):
                print(f"  Testing {method} ({len(remaining)}/{len(wavelengths)} remaining)...")
            else:
                print(f"  Testing {method} calibration across wavelengths...")
            logger.info(f"  Testing {method} calibration: {len(remaining)} wavelengths remaining")
            
            for wavelength in tqdm(remaining, desc=f"  {method}", leave=False):
                try:
                    coupling = evaluate_fixed_config_at_wavelength(
                        lenses, lens1, lens2,
                        calib['z_l1'], calib['z_l2'], calib['z_fiber'],
                        wavelength, n_rays=n_rays, medium=medium,
                        orientation=calib.get('orientation', 'ScffcF')
                    )
                    
                    result_entry = {
                        'lens1': lens1,
                        'lens2': lens2,
                        'method': method,
                        'wavelength_nm': float(wavelength),
                        'coupling': float(coupling),
                        'total_len_mm': float(calib['total_len_mm']),
                        'z_l1': float(calib['z_l1']),
                        'z_l2': float(calib['z_l2']),
                        'z_fiber': float(calib['z_fiber']),
                        'orientation': calib.get('orientation', 'ScffcF'),
                        'medium': medium
                    }
                    
                    write_temp(result_entry, run_id, f'wavelength_{combo_id}')
                    logger.info(f"    {wavelength}nm: coupling={coupling:.4f}")
                    
                except Exception as e:
                    logger.error(f"    {wavelength}nm: Error - {str(e)}")
        
        temp_file = results_dir / f"temp_batch_wavelength_{combo_id}.json"
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
