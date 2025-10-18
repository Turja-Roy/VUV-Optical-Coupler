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

    # Handle wavelength-analyze-plot mode
    if args['mode'] == 'wavelength-analyze-plot':
        import glob
        import os
        
        print("\n" + "="*60)
        print("Wavelength Analysis Plotting")
        print("="*60)
        print(f"Results directory: {args['results_dir']}")
        print(f"Plot directory: plots/{run_id}/")
        if args['fit_type']:
            print(f"Curve fitting: {args['fit_type']}")
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
            lens1 = df['lens1'].iloc[0]
            lens2 = df['lens2'].iloc[0]
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
        
        if args['aggregate']:
            from scripts.visualizers import plot_wavelength_per_lens_aggregated, plot_wavelength_per_method_aggregated
            
            per_lens_agg_dir = plot_base_dir / 'per_lens_aggregated'
            per_method_agg_dir = plot_base_dir / 'per_method_aggregated'
            
            print("\nGenerating aggregated per-lens-combination plots (with error bars)...")
            for combo_name, methods_data in all_data.items():
                lens1, lens2 = combo_name.split('+')
                plot_wavelength_per_lens_aggregated(lens1, lens2, methods_data, 
                                                   str(per_lens_agg_dir), args['fit_type'])
                print(f"  Created aggregated plot for {combo_name}")
            
            print("\nGenerating aggregated per-method plots (with error bars)...")
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
                                                         str(per_method_agg_dir), args['fit_type'])
                    print(f"  Created aggregated plot for {method}")
            
            print("\n" + "="*60)
            print("Aggregated Plotting Complete!")
            print("="*60)
            print(f"Per-lens aggregated plots: {per_lens_agg_dir}/")
            print(f"Per-method aggregated plots: {per_method_agg_dir}/")
            print()
        else:
            from scripts.visualizers import plot_wavelength_per_lens, plot_wavelength_per_method
            
            print("\nGenerating per-lens-combination plots...")
            for combo_name, methods_data in all_data.items():
                lens1, lens2 = combo_name.split('+')
                plot_wavelength_per_lens(lens1, lens2, methods_data, 
                                        str(per_lens_dir), args['fit_type'])
                print(f"  Created plot for {combo_name}")
            
            print("\nGenerating per-method plots...")
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
                                              str(per_method_dir), args['fit_type'])
                    print(f"  Created plot for {method}")
            
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
        
        Path(f'./results/{run_id}').mkdir(parents=True, exist_ok=True)
        
        wavelength_analysis(
            args['results_file'], 
            run_id,
            wl_start=args['wl_start'],
            wl_end=args['wl_end'],
            wl_step=args['wl_step'],
            n_rays=args['n_rays'],
            alpha=args['alpha'],
            medium=args['medium']
        )
        
        return

    # Handle analyze mode
    if args['mode'] == 'analyze':
        from scripts.analysis import analyze_combos
        
        _, lenses = find_combos('combine')
        
        Path(f'./results/{run_id}').mkdir(parents=True, exist_ok=True)
        Path(f'./plots/{run_id}').mkdir(parents=True, exist_ok=True)
        
        all_results = analyze_combos(
            args['results_file'],
            args['coupling_threshold'],
            lenses,
            run_id,
            alpha=args['alpha'],
            medium=args['medium']
        )
        
        for method, results in all_results.items():
            if results:
                print(f"\nSaving {method} results...")
                write_results(f'analyze_{method}', results, run_id)
        
        combined_results = []
        for method, results in all_results.items():
            combined_results.extend(results)
        
        if combined_results:
            print(f"\nSaving combined results...")
            write_results('analyze_combined', combined_results, run_id)
            
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
        medium=args['medium']
    )

    # Run optimization
    if args['continue'] and batch_run:
        results = run_batches_continue(combos, lenses, run_id,
                                       args['method'], runner_func)
    elif batch_run:
        results = run_batches(combos, lenses, run_id,
                              args['method'], runner_func)
    else:
        results = runner_func(lenses, combos, run_id, None)
        lens_pair = (args['lens1'], args['lens2']) if args['mode'] == 'particular' else None
        write_results(args['method'], results, run_id, lens_pair=lens_pair)

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
