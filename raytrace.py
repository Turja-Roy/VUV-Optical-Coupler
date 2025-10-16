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
        run_id = f"{args['date']}_compare"
    elif args['mode'] == 'particular':
        run_id = f"{args['date']}_particular_{args['optimizer']}"
    else:
        run_id = f"{args['date']}_{args['method']}_{args['optimizer']}"

    print("\n" + "="*60)
    print("Lens Configuration Optimizer")
    print("="*60)
    print(f"Run ID: {run_id}")
    print(f"Optimizer: {args['optimizer']}")
    if args['optimizer'] != 'grid_search':
        print(f"Alpha (coupling weight): {args['alpha']}")
    print("="*60 + "\n")

    # Handle compare mode
    if args['mode'] == 'compare':
        from scripts.optimization.optimization_runner import compare_optimizers
        combos, lenses = particular_combo(args['lens1'], args['lens2'])
        compare_optimizers(lenses, (args['lens1'], args['lens2']),
                           run_id, alpha=args['alpha'])
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
        alpha=args['alpha'], n_rays=1000, batch_num=batch_num
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
        write_results(args['method'], results, run_id)

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
                                     best_coupling['lens2'])

        plot_spot_diagram(best_coupling, lenses, run_id)

        print(f"\nResults saved to: results/{run_id}/")
        print(f"Plots saved to: plots/{run_id}/")


if __name__ == "__main__":
    main()
