import sys
import os
import numpy as np
import pandas as pd
import regex as re
from pathlib import Path

from scripts import consts as C
from scripts.fetcher import find_combos, particular_combo, write_results
from scripts.visualizers import plot_spot_diagram


def parse_arguments():
    """Parse command line arguments."""
    args = {
        'mode': None,
        'lens1': None,
        'lens2': None,
        'method': None,
        'optimizer': 'differential_evolution',  # default
        'alpha': 0.7,  # default: 70% coupling, 30% length
        'continue': False,
        'date': C.DATE_STR
    }

    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)

    # Parse main command
    if sys.argv[1] == 'particular':
        if len(sys.argv) < 4:
            print("Error: particular mode requires two lens names")
            print_usage()
            sys.exit(1)
        args['mode'] = 'particular'
        args['lens1'] = sys.argv[2]
        args['lens2'] = sys.argv[3]

    elif sys.argv[1] == 'compare':
        if len(sys.argv) < 4:
            print("Error: compare mode requires two lens names")
            print_usage()
            sys.exit(1)
        args['mode'] = 'compare'
        args['lens1'] = sys.argv[2]
        args['lens2'] = sys.argv[3]

    elif sys.argv[1] in ['select', 'combine']:
        args['mode'] = 'method'
        args['method'] = sys.argv[1]

    else:
        print(f"Error: Unknown command '{sys.argv[1]}'")
        print_usage()
        sys.exit(1)

    # Parse optional arguments
    i = 2 if args['mode'] == 'method' else 4
    while i < len(sys.argv):
        arg = sys.argv[i]

        if arg == '--opt':
            if i + 1 < len(sys.argv):
                args['optimizer'] = sys.argv[i + 1]
                i += 2
            else:
                print("Error: --opt requires an optimizer name")
                sys.exit(1)

        elif arg == '--alpha':
            if i + 1 < len(sys.argv):
                try:
                    args['alpha'] = float(sys.argv[i + 1])
                    if not 0 <= args['alpha'] <= 1:
                        raise ValueError
                except ValueError:
                    print("Error: --alpha must be between 0 and 1")
                    sys.exit(1)
                i += 2
            else:
                print("Error: --alpha requires a value")
                sys.exit(1)

        elif arg == 'continue':
            args['continue'] = True
            i += 1

        elif re.match(r'\d{4}-\d{2}-\d{2}', arg):
            args['date'] = arg
            i += 1

        else:
            print(f"Error: Unknown argument '{arg}'")
            print_usage()
            sys.exit(1)

    return args


def print_usage():
    """Print usage information."""
    print("""
Usage:
    python raytrace.py <command> [options]

Commands:
    particular <lens1> <lens2>    Run optimization for specific lens pair
    compare <lens1> <lens2>       Compare all optimization methods
    select                        Run all L1 x L2 combinations
    combine                       Run all combinations from Combined_Lenses.csv

Options:
    --opt <method>                Optimization method (default: differential_evolution)
                                  Options: differential_evolution, dual_annealing,
                                           nelder_mead, powell, bayesian, grid_search
    --alpha <value>               Weight for coupling vs. length (0-1, default: 0.7)
                                  Higher = prioritize coupling more
    continue                      Continue incomplete batch run
    <YYYY-MM-DD>                  Specify run date

Examples:
    # Fast global optimization (recommended)
    python raytrace.py combine --opt differential_evolution
    
    # Prioritize coupling more (90% coupling, 10% length)
    python raytrace.py combine --opt differential_evolution --alpha 0.9
    
    # Bayesian optimization (install: pip install scikit-optimize)
    python raytrace.py combine --opt bayesian
    
    # Compare methods on a test case
    python raytrace.py compare LA4001 LA4647
    
    # Use legacy grid search
    python raytrace.py combine --opt grid_search
    
    # Continue from specific date
    python raytrace.py combine continue 2025-10-14
""")


def main():
    args = parse_arguments()

    print("\n" + "="*60)
    print("Lens Configuration Optimizer")
    print("="*60)
    print(f"Run date: {args['date']}")
    print(f"Optimizer: {args['optimizer']}")
    if args['optimizer'] != 'grid_search':
        print(f"Alpha (coupling weight): {args['alpha']}")
    print("="*60 + "\n")

    # Handle compare mode
    if args['mode'] == 'compare':
        from scripts.optimization.optimization_runner import compare_optimizers
        combos, lenses = particular_combo(args['lens1'], args['lens2'])
        compare_optimizers(lenses, (args['lens1'], args['lens2']),
                           args['date'], alpha=args['alpha'])
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
    if args['optimizer'] == 'grid_search':
        print("Using legacy grid search method...")
        from scripts.optimization.legacy.grid_search import run_combos

        def runner_func(lenses, combos, date, batch_num): return run_combos(
            lenses, combos, date, batch_num
        )
    else:
        print(f"Using {args['optimizer']} optimization...")
        from scripts.optimization.optimization_runner import run_combos_optimized

        def runner_func(lenses, combos, date, batch_num): return run_combos_optimized(
            lenses, combos, date, method=args['optimizer'],
            alpha=args['alpha'], n_rays=1000, batch_num=batch_num
        )

    # Run optimization
    if args['continue'] and batch_run:
        results = run_batches_continue(combos, lenses, args['date'],
                                       args['method'], runner_func)
    elif batch_run:
        results = run_batches(combos, lenses, args['date'],
                              args['method'], runner_func)
    else:
        results = runner_func(lenses, combos, args['date'], None)
        write_results(args['method'], results, args['date'])

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
            from scripts.optimization.legacy.grid_search import run_grid
            best_coupling = run_grid(args['date'], lenses,
                                     best_coupling['lens1'],
                                     best_coupling['lens2'])

        plot_spot_diagram(best_coupling, lenses, args['date'])

        print(f"\nResults saved to: results/{args['date']}/")
        print(f"Plots saved to: plots/{args['date']}/")


def run_batches(combos, lenses, run_date, method, runner_func):
    """Run optimization in batches."""
    results = []
    n_batches = int(np.ceil(len(combos) / 100))

    for i in range(n_batches):
        batch_combos = combos[i*100:(i+1)*100]
        print(f"\n{'='*60}")
        print(f"Batch {i+1}/{n_batches}")
        print(f"{'='*60}")

        batch_results = runner_func(lenses, batch_combos, run_date, i+1)
        write_results(method, batch_results, run_date,
                      batch=True, batch_num=i+1)
        results.extend(batch_results)

    # Write combined results
    write_results(method, results, run_date)
    return results


def run_batches_continue(combos, lenses, run_date, method, runner_func):
    """Continue incomplete batch run."""
    results_dir = Path(f'./results/{run_date}')

    # Find completed batches
    completed = len([f for f in results_dir.glob('batch_*.csv')])

    print(f"Found {completed} completed batches")

    # Load existing results
    results = []
    for i in range(completed):
        df = pd.read_csv(results_dir / f'batch_{method}_{i+1}.csv')
        results.extend(df.to_dict('records'))

    # Check for incomplete batch
    temp_files = list(results_dir.glob('temp*.json'))
    completed_combos = 0
    if temp_files:
        import json
        with open(temp_files[0], 'r') as f:
            try:
                completed_data = json.load(f)
                completed_combos = len(completed_data)
                print(f"Resuming incomplete batch {completed+1} "
                      f"({completed_combos} combos done)")
            except json.JSONDecodeError:
                completed_combos = 0

    # Start from where we left off
    start_index = completed * 100 + completed_combos
    remaining_combos = combos[start_index:]
    
    n_total_batches = int(np.ceil(len(combos) / 100))
    starting_batch = completed + 1 if completed_combos > 0 else completed + 1

    print(f"Starting from combo {start_index} (batch {starting_batch})")
    
    # Process remaining batches
    current_combo_idx = start_index
    for batch_num in range(starting_batch, n_total_batches + 1):
        batch_start = (batch_num - 1) * 100
        batch_end = min(batch_num * 100, len(combos))
        
        # Skip combos already done in this batch
        if current_combo_idx > batch_start:
            batch_start = current_combo_idx
        
        if batch_start >= batch_end:
            continue
            
        batch_combos = combos[batch_start:batch_end]
        n_remaining_batches = n_total_batches - batch_num + 1

        print(f"\n{'='*60}")
        print(f"Batch {batch_num}/{n_total_batches} ({len(batch_combos)} combos, {n_remaining_batches} batches remaining)")
        print(f"{'='*60}")

        batch_results = runner_func(lenses, batch_combos, run_date, batch_num)
        write_results(method, batch_results, run_date,
                      batch=True, batch_num=batch_num)
        results.extend(batch_results)
        
        current_combo_idx = batch_end

    # Write combined results
    write_results(method, results, run_date)
    return results


if __name__ == "__main__":
    main()
