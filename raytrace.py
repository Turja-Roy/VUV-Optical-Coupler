import numpy as np
import pandas as pd
from pathlib import Path

from scripts import consts as C
from scripts.data_io import find_combos, particular_combo, write_results
from scripts.visualizers import plot_spot_diagram
from scripts.runner import run_batches, run_batches_continue
from scripts.cli import parse_arguments, print_usage


def main():
    args = parse_arguments()
    
    # Get plot style from config if available
    plot_style = '3d'  # Default
    orientation_mode = 'both'  # Default
    if '_config' in args:
        from scripts.config_loader import ConfigLoader
        loader = ConfigLoader()
        plot_style = loader.get_plot_style(args['_config'])
        orientation_mode = loader.get_orientation_mode(args['_config'])
    
    # Initialize database connection if enabled
    db = None
    if C.USE_DATABASE:
        from scripts.database import OptimizationDatabase
        db = OptimizationDatabase(C.DATABASE_PATH)
        print(f"Database enabled: {C.DATABASE_PATH}")

    # Build run_id based on mode and method
    if args['mode'] == 'compare':
        run_id = f"compare_{args['date']}_{args['lens1']}+{args['lens2']}_{args['medium']}"
    elif args['mode'] == 'particular':
        run_id = f"particular_{args['date']}_{args['optimizer']}_{args['medium']}"
    elif args['mode'] == 'analyze':
        threshold_str = f"{args['coupling_threshold']:.2f}".replace('.', '_')
        run_id = f"analyze_{args['date']}_coupling_{threshold_str}_{args['medium']}"
    elif args['mode'] == 'wavelength-analyze':
        run_id = f"wavelength_analyze_{args['date']}"
    elif args['mode'] == 'wavelength-analyze-plot':
        run_id = Path(args['results_dir']).name
    else:
        run_id = f"{args['date']}_{args['method']}_{args['optimizer']}_{args['medium']}"

    # Handle dashboard mode
    if args['mode'] == 'dashboard':
        from scripts.web_dashboard import start_dashboard
        
        # Load dashboard params from config if available
        dashboard_port = args['port']
        dashboard_db = args['db_path']
        
        if '_config' in args:
            from scripts.config_loader import ConfigLoader
            loader = ConfigLoader()
            dash_params = loader.get_dashboard_params(args['_config'])
            
            # CLI args override config values
            import sys
            if '--port' not in sys.argv:
                dashboard_port = dash_params['port']
            if '--db' not in sys.argv:
                dashboard_db = dash_params['db_path']
        
        print("\n" + "="*60)
        print("Starting Web Dashboard")
        print("="*60)
        print(f"Port: {dashboard_port}")
        if dashboard_db:
            print(f"Database: {dashboard_db}")
        else:
            print(f"Database: Auto-detect")
        print(f"Results directory: ./results")
        print("="*60)
        
        start_dashboard(
            port=dashboard_port,
            db_path=dashboard_db,
            results_dir='./results'
        )
        return

    # Handle wavelength-analyze-plot mode
    if args['mode'] == 'wavelength-analyze-plot':
        import glob
        import os
        
        print("\n" + "="*60)
        print("Wavelength Analysis Plotting")
        print("="*60)
        print(f"Results directory: {args['results_dir']}")
        print(f"Plot directory: plots/{run_id}/")
        if args['fit_types']:
            # Convert None to 'none' for display purposes
            fit_display = [ft if ft is not None else 'none' for ft in args['fit_types']]
            print(f"Curve fitting: {', '.join(fit_display)}")
        print("="*60 + "\n")
        
        results_dir = Path(args['results_dir'])
        if not results_dir.exists():
            print(f"Error: Results directory '{args['results_dir']}' does not exist")
            return
        
        csv_files = list(results_dir.glob('*_wavelength.csv'))
        if not csv_files:
            print(f"Error: No wavelength CSV files found in '{args['results_dir']}'")
            return
        
        print(f"Found {len(csv_files)} wavelength CSV file(s)")
        
        from scripts.visualizers import plot_wavelength_per_lens, plot_wavelength_per_method
        
        all_data = {}
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            # df = df[df['method'] != 'nelder_mead']  # Exclude 'nelder_mead' method (temporary)
            lens1 = df['lens1'].iloc[0]
            lens2 = df['lens2'].iloc[0]
            
            has_medium = 'medium' in df.columns
            
            if has_medium:
                for medium in df['medium'].unique():
                    medium_filtered = df[df['medium'] == medium]
                    combo_name = f"{lens1}+{lens2}_{medium}"
                    all_data[combo_name] = {}
                    
                    for method in medium_filtered['method'].unique():
                        method_filtered = medium_filtered[medium_filtered['method'] == method]
                        wavelengths = np.array(method_filtered['wavelength_nm'])
                        couplings = np.array(method_filtered['coupling'])
                        sorted_indices = np.argsort(wavelengths)
                        all_data[combo_name][method] = {
                            'wavelengths': wavelengths[sorted_indices],
                            'couplings': couplings[sorted_indices]
                        }
            else:
                combo_name = f"{lens1}+{lens2}"
                all_data[combo_name] = {}
                
                for method in df['method'].unique():
                    method_df = df[df['method'] == method]
                    wavelengths = np.array(method_df['wavelength_nm'])
                    couplings = np.array(method_df['coupling'])
                    sorted_indices = np.argsort(wavelengths)
                    all_data[combo_name][method] = {
                        'wavelengths': wavelengths[sorted_indices],
                        'couplings': couplings[sorted_indices]
                    }
        
        plot_base_dir = Path(f'./plots/{run_id}')
        per_lens_dir = plot_base_dir / 'per_lens'
        per_method_dir = plot_base_dir / 'per_method'
        
        # If no fit_types specified, generate plots without fits (None)
        fit_types_to_process = args['fit_types'] if args['fit_types'] else [None]
        
        if args['aggregate']:
            from scripts.visualizers import plot_wavelength_per_lens_aggregated, plot_wavelength_per_method_aggregated
            
            per_lens_agg_dir = plot_base_dir / 'per_lens_aggregated'
            per_method_agg_dir = plot_base_dir / 'per_method_aggregated'
            
            for fit_type in fit_types_to_process:
                fit_suffix = f" ({fit_type})" if fit_type else ""
                print(f"\nGenerating aggregated per-lens-combination plots{fit_suffix}...")
                for combo_name, methods_data in all_data.items():
                    lens1, lens2 = combo_name.split('+')
                    plot_wavelength_per_lens_aggregated(lens1, lens2, methods_data, 
                                                       str(per_lens_agg_dir), fit_type)
                    print(f"  Created aggregated plot for {combo_name}{fit_suffix}")
                
                print(f"\nGenerating aggregated per-method plots{fit_suffix}...")
                all_methods = set()
                for combo_data in all_data.values():
                    all_methods.update(combo_data.keys())
                
                for method in sorted(all_methods):
                    lens_combos_data = {}
                    for combo_name, methods_data in all_data.items():
                        if method in methods_data:
                            lens_combos_data[combo_name] = methods_data[method]
                    
                    if lens_combos_data:
                        plot_wavelength_per_method_aggregated(method, lens_combos_data, 
                                                             str(per_method_agg_dir), fit_type)
                        print(f"  Created aggregated plot for {method}{fit_suffix}")
            
            print("\n" + "="*60)
            print("Aggregated Plotting Complete!")
            print("="*60)
            print(f"Per-lens aggregated plots: {per_lens_agg_dir}/")
            print(f"Per-method aggregated plots: {per_method_agg_dir}/")
            print()
        else:
            from scripts.visualizers import plot_wavelength_per_lens, plot_wavelength_per_method
            
            for fit_type in fit_types_to_process:
                fit_suffix = f" ({fit_type})" if fit_type else ""
                print(f"\nGenerating per-lens-combination plots{fit_suffix}...")
                for combo_name, methods_data in all_data.items():
                    lens1, lens2 = combo_name.split('+')
                    plot_wavelength_per_lens(lens1, lens2, methods_data, 
                                            str(per_lens_dir), fit_type)
                    print(f"  Created plot for {combo_name}{fit_suffix}")
                
                print(f"\nGenerating per-method plots{fit_suffix}...")
                all_methods = set()
                for combo_data in all_data.values():
                    all_methods.update(combo_data.keys())
                
                for method in sorted(all_methods):
                    lens_combos_data = {}
                    for combo_name, methods_data in all_data.items():
                        if method in methods_data:
                            lens_combos_data[combo_name] = methods_data[method]
                    
                    if lens_combos_data:
                        plot_wavelength_per_method(method, lens_combos_data, 
                                                  str(per_method_dir), fit_type)
                        print(f"  Created plot for {method}{fit_suffix}")
            
            print("\n" + "="*60)
            print("Plotting Complete!")
            print("="*60)
            print(f"Per-lens plots saved to: {per_lens_dir}/")
            print(f"Per-method plots saved to: {per_method_dir}/")
            print()
        
        return

    print("\n" + "="*60)
    print("Lens Configuration Optimizer")
    print("="*60)
    print(f"Run ID: {run_id}")
    if args['mode'] == 'wavelength-analyze':
        print(f"Mode: Wavelength Analysis")
        print(f"Input CSV: {args['results_file']}")
        print(f"Wavelength range: {args['wl_start']}-{args['wl_end']} nm (step: {args['wl_step']} nm)")
        print(f"Rays per trace: {args['n_rays']}")
        print(f"Medium: {args['medium']}")
        print(f"Alpha (coupling weight): {args['alpha']}")
    elif args['mode'] != 'analyze':
        print(f"Optimizer: {args['optimizer']}")
        print(f"Medium: {args['medium']}")
        if args['optimizer'] != 'grid_search':
            print(f"Alpha (coupling weight): {args['alpha']}")
    else:
        print(f"Mode: Analyze high-coupling results")
        print(f"Medium: {args['medium']}")
        print(f"Coupling threshold: {args['coupling_threshold']}")
        print(f"Results file: {args['results_file']}")
    print("="*60 + "\n")

    # Handle wavelength-analyze mode
    if args['mode'] == 'wavelength-analyze':
        from scripts.analysis import wavelength_analysis
        
        # Load wavelength params from config if available
        wl_start = args['wl_start']
        wl_end = args['wl_end']
        wl_step = args['wl_step']
        wl_n_rays = args['n_rays']
        wl_methods = None
        wl_orientation_mode = 'both'  # Default
        
        if '_config' in args:
            from scripts.config_loader import ConfigLoader
            loader = ConfigLoader()
            wl_params = loader.get_wavelength_params(args['_config'])
            wl_orientation_mode = loader.get_orientation_mode(args['_config'])
            
            # CLI args override config values
            import sys
            if '--wl-start' not in sys.argv:
                wl_start = wl_params['wl_start']
            if '--wl-end' not in sys.argv:
                wl_end = wl_params['wl_end']
            if '--wl-step' not in sys.argv:
                wl_step = wl_params['wl_step']
            if '--n-rays' not in sys.argv:
                wl_n_rays = wl_params['n_rays']
            wl_methods = wl_params['methods']
        
        Path(f'./results/{run_id}').mkdir(parents=True, exist_ok=True)
        
        wavelength_analysis(
            args['results_file'], 
            run_id,
            wl_start=wl_start,
            wl_end=wl_end,
            wl_step=wl_step,
            n_rays=wl_n_rays,
            alpha=args['alpha'],
            medium=args['medium'],
            methods=wl_methods,
            orientation_mode=wl_orientation_mode
        )
        
        return

    # Handle analyze mode
    if args['mode'] == 'analyze':
        from scripts.analysis import analyze_combos
        
        _, lenses = find_combos('combine')
        
        # Load analyze params from config if available
        analyze_n_rays = args['n_rays']
        analyze_threshold = args['coupling_threshold']
        analyze_methods = None
        analyze_plot_style = '3d'  # Default
        analyze_orientation_mode = 'both'  # Default
        
        if '_config' in args:
            from scripts.config_loader import ConfigLoader
            loader = ConfigLoader()
            analyze_params = loader.get_analyze_params(args['_config'])
            analyze_plot_style = loader.get_plot_style(args['_config'])
            analyze_orientation_mode = loader.get_orientation_mode(args['_config'])
            
            # CLI args override config values
            import sys
            if '--n-rays' not in sys.argv:
                analyze_n_rays = analyze_params['n_rays']
            if '--coupling-threshold' not in sys.argv:
                analyze_threshold = analyze_params['coupling_threshold']
            analyze_methods = analyze_params['methods']
        
        Path(f'./results/{run_id}').mkdir(parents=True, exist_ok=True)
        Path(f'./plots/{run_id}').mkdir(parents=True, exist_ok=True)
        
        all_results = analyze_combos(
            args['results_file'],
            analyze_threshold,
            lenses,
            run_id,
            alpha=args['alpha'],
            medium=args['medium'],
            n_rays=analyze_n_rays,
            methods=analyze_methods,
            plot_style=analyze_plot_style,
            orientation_mode=analyze_orientation_mode
        )
        
        for method, results in all_results.items():
            if results:
                print(f"\nSaving {method} results...")
                write_results(f'analyze_{method}', results, run_id, db=db, alpha=args['alpha'])
        
        combined_results = []
        for method, results in all_results.items():
            combined_results.extend(results)
        
        if combined_results:
            print(f"\nSaving combined results...")
            write_results('analyze_combined', combined_results, run_id, db=db, alpha=args['alpha'])
            
            print("\n" + "="*60)
            print("Analysis Complete!")
            print("="*60)
            
            best_by_method = {}
            for method, results in all_results.items():
                if results:
                    best = max(results, key=lambda x: x['coupling'])
                    best_by_method[method] = best
                    print(f"\nBest {method}:")
                    print(f"  Coupling: {best['coupling']:.4f}")
                    print(f"  Lenses: {best['lens1']} + {best['lens2']}")
                    print(f"  Length: {best['total_len_mm']:.2f} mm")
                    print(f"  Time: {best.get('time_seconds', 0):.2f}s")
            
            print(f"\nResults saved to: results/{run_id}/")
            print(f"Plots saved to: plots/{run_id}/")
        
        return

    # Handle compare mode
    if args['mode'] == 'compare':
        from scripts.optimization.optimization_runner import compare_optimizers
        combos, lenses = particular_combo(args['lens1'], args['lens2'])
        compare_optimizers(lenses, (args['lens1'], args['lens2']),
                           run_id, alpha=args['alpha'], medium=args['medium'])
        return
    
    # Handle tolerance analysis mode
    if args['mode'] == 'tolerance':
        from scripts.tolerance_analysis import analyze_tolerance, save_tolerance_results
        from scripts.visualizers import plot_tolerance_results
        
        # Load tolerance parameters from config if available, otherwise use CLI args
        tolerance_plot_style = '3d'  # Default
        tolerance_orientation_mode = 'both'  # Default
        if '_config' in args:
            from scripts.config_loader import ConfigLoader
            loader = ConfigLoader()
            tol_params = loader.get_tolerance_params(args['_config'])
            tolerance_plot_style = loader.get_plot_style(args['_config'])
            tolerance_orientation_mode = loader.get_orientation_mode(args['_config'])
            
            # CLI args override config values (only if explicitly provided)
            # Check if CLI args were explicitly set by looking at sys.argv
            import sys
            if '--z-range' not in sys.argv:
                args['z_range'] = tol_params['z_range_mm']
            if '--n-samples' not in sys.argv:
                args['n_samples'] = tol_params['n_samples']
            # n_rays for tolerance is separate from optimization n_rays
            # Use config tolerance n_rays if available and --n-rays not specified
            if '--n-rays' not in sys.argv:
                tolerance_n_rays = tol_params['n_rays']
            else:
                tolerance_n_rays = args['n_rays']
        else:
            tolerance_n_rays = args['n_rays']
        
        print("\n" + "="*70)
        print("TOLERANCE ANALYSIS MODE")
        print("="*70)
        print(f"Lens pair: {args['lens1']} + {args['lens2']}")
        print(f"Optimizer: {args['optimizer']}")
        print(f"Medium: {args['medium']}")
        print(f"Tolerance parameters:")
        print(f"  Z-displacement range: ±{args['z_range']:.2f} mm")
        print(f"  Number of samples: {args['n_samples']}")
        print(f"  Rays per test: {tolerance_n_rays}")
        print("="*70)
        
        # First, run optimization to get baseline configuration
        combos, lenses = particular_combo(args['lens1'], args['lens2'])
        
        print("\nStep 1: Finding optimal configuration...")
        from scripts.optimization.optimization_runner import run_combos
        
        results = run_combos(
            lenses, combos, run_id, method=args['optimizer'],
            alpha=args['alpha'], n_rays=args['n_rays'],
            batch_num=None, medium=args['medium'], db=db,
            plot_style=tolerance_plot_style, orientation_mode=tolerance_orientation_mode
        )
        
        if not results or len(results) == 0:
            print("Error: Optimization failed, no results returned")
            return
        
        # Get best result
        best_result = max(results, key=lambda x: x['coupling'])
        
        print("\nStep 2: Running tolerance analysis...")
        tolerance_results = analyze_tolerance(
            lenses, best_result,
            n_rays=tolerance_n_rays,
            n_samples=args['n_samples'],
            z_range_mm=args['z_range'],
            medium=args['medium']
        )
        
        # Save results
        print("\nStep 3: Saving results...")
        save_tolerance_results(tolerance_results, run_id)
        
        # Generate plots
        print("\nStep 4: Generating plots...")
        plot_tolerance_results(tolerance_results, run_id)
        
        print("\n" + "="*70)
        print("TOLERANCE ANALYSIS COMPLETE")
        print("="*70)
        print(f"Results saved to: results/{run_id}/")
        print(f"Plots saved to: plots/{run_id}/")
        
        return

    # Get lens combinations
    if args['mode'] == 'particular':
        combos, lenses = particular_combo(args['lens1'], args['lens2'])
        batch_run = False
    else:
        combos, lenses = find_combos(args['method'])
        batch_run = len(combos) > 100

    print(f"Total lens combinations: {len(combos)}")

    # Select runner based on optimizer
    print(f"Using {args['optimizer']} optimization...")
    from scripts.optimization.optimization_runner import run_combos

    def runner_func(lenses, combos, run_id, batch_num): return run_combos(
        lenses, combos, run_id, method=args['optimizer'],
        alpha=args['alpha'], n_rays=1000, batch_num=batch_num,
        medium=args['medium'], db=db, plot_style=plot_style, orientation_mode=orientation_mode
    )

    # Run optimization
    if args['continue'] and batch_run:
        results = run_batches_continue(combos, lenses, run_id,
                                       args['method'], runner_func, db=db, alpha=args['alpha'])
    elif batch_run:
        results = run_batches(combos, lenses, run_id,
                              args['method'], runner_func, db=db, alpha=args['alpha'])
    else:
        results = runner_func(lenses, combos, run_id, None)
        lens_pair = (args['lens1'], args['lens2']) if args['mode'] == 'particular' else None
        # For particular mode, method is in 'optimizer', otherwise it's in 'method'
        method_name = args.get('optimizer') or args.get('method')
        write_results(method_name, results, run_id, lens_pair=lens_pair, db=db, alpha=args['alpha'])

    # Analyze results
    if results:
        print("\n" + "="*60)
        print("Optimization Complete!")
        print("="*60)

        # Find best by coupling
        best_coupling = max(results, key=lambda x: x['coupling'])
        print(f"\nBest coupling: {best_coupling['coupling']:.4f}")
        print(f"  Lenses: {best_coupling['lens1']} + {best_coupling['lens2']}")
        print(f"  Length: {best_coupling['total_len_mm']:.2f} mm")
        print(f"  Config: z_l1={best_coupling['z_l1']:.2f}, "
              f"z_l2={best_coupling['z_l2']:.2f}, "
              f"z_fiber={best_coupling['z_fiber']:.2f}")

        # Find best by length (with acceptable coupling)
        acceptable_coupling = 0.5 * best_coupling['coupling']  # 50% of best
        good_results = [r for r in results if r['coupling']
                        >= acceptable_coupling]
        if good_results:
            best_compact = min(good_results, key=lambda x: x['total_len_mm'])
            print(f"\nBest compact design (coupling >= {
                  acceptable_coupling:.4f}):")
            print(f"  Coupling: {best_compact['coupling']:.4f}")
            print(f"  Lenses: {
                  best_compact['lens1']} + {best_compact['lens2']}")
            print(f"  Length: {best_compact['total_len_mm']:.2f} mm")
            print(f"  Config: z_l1={best_compact['z_l1']:.2f}, "
                  f"z_l2={best_compact['z_l2']:.2f}, "
                  f"z_fiber={best_compact['z_fiber']:.2f}")

        # Generate spot diagram for best
        print("\nGenerating spot diagram for best configuration...")
        # Re-run best to get ray data if needed
        if 'origins' not in best_coupling or best_coupling['origins'] is None:
            from scripts.optimization.grid_search import run_grid
            best_coupling = run_grid(run_id, lenses,
                                     best_coupling['lens1'],
                                     best_coupling['lens2'],
                                     medium=args['medium'])

        plot_spot_diagram(best_coupling, lenses, run_id)

        print(f"\nResults saved to: results/{run_id}/")
        print(f"Plots saved to: plots/{run_id}/")


if __name__ == "__main__":
    main()
