import sys
import os
import numpy as np
import pandas as pd
import regex as re
from scripts import consts as C
from scripts.fetcher import find_combos, particular_combo, write_results
from scripts.runner import run_combos
from scripts.visualizers import plot_spot_diagram


def main():
    batch_run = False
    batch_continue = False
    method = 'combine'  # default
    run_date = C.DATE_STR

    """
        Parse command line arguments
        Usage:
        python raytrace.py [particular <lens1> <lens2> | select [batch] [continue] [YYYY-MM-DD] | combine [batch] [continue]]
    """
    if sys.argv[1] == 'particular':                                     # Particular lenses
        method = 'particular'
        combos, lenses = particular_combo(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == 'select':                                       # Select candidates
        method = 'select'
        if 'batch' in sys.argv and 'continue' not in sys.argv:
            batch_run = True
        if 'continue' in sys.argv:
            batch_continue = True
        date_args = [arg for arg in sys.argv if re.match(r'\d{4}-\d{2}-\d{2}', arg)]
        if date_args != []:
            run_date = date_args[0]
        print('Run date: ' + run_date)
        combos, lenses = find_combos('select')
    else:                                                               # All lens combos
        method = 'combine'
        if 'batch' in sys.argv: batch_run = True
        if 'continue' in sys.argv: batch_continue = True
        combos, lenses = find_combos('combine')


    """
        Run sweep across all combos (coarse+refine)
    """
    results = []
    print("Total number of lens combos to evaluate:", len(combos))

    if batch_continue:  # Continue incomplete run
        existing_files = [f for f in os.listdir(f'./results/{run_date}') if f.startswith('batch_') and f.endswith('.csv')]
        completed = len(existing_files)
        remaining = int(np.ceil((len(combos) - completed * 100) / 100))

        # Check if the last batch is completed
        check_df = pd.read_csv(f'./results/{run_date}/batch_{completed}.csv')
        if len(check_df) < 100:
            combos = combos[completed * 100 - len(check_df):]
            print(f"Resuming from batch {completed} with {len(combos)} combos remaining...")

            # Run for only the remaining combos in the last batch
            incomplete_batch_results = run_combos(lenses, combos[:100 - len(check_df)], run_date)
            write_results(method, incomplete_batch_results, run_date, batch=True, batch_num=completed, contd_batch=True)

            results.extend(incomplete_batch_results)
            combos = combos[100 - len(check_df):]

        else:
            combos = combos[completed * 100:]
            print(f"Resuming from batch {completed + 1} with {len(combos)} combos remaining...")

        # Run the remaining batches
        for i in range(remaining):
            batch_combos = combos[:100]
            print(f"\nStarting batch {completed+i+1}/{completed + remaining}")

            batch_results = run_combos(lenses, batch_combos, run_date)
            write_results(method, batch_results, run_date, batch=True, batch_num=completed+i+1)

            combos = combos[100:]
            results.extend(batch_results)

    elif batch_run:  # Start batches from beginning
        n_batches = int(np.ceil(len(combos) / 100))

        for i in range(n_batches):
            batch_combos = combos[:100]
            print(f"\nStarting batch {i+1}/{n_batches}")

            batch_results = run_combos(lenses, batch_combos, run_date)
            write_results(method, batch_results, run_date, batch=True, batch_num=i+1)

            combos = combos[100:]
            results.extend(batch_results)

    else:  # No batches
        print(f"Running coarse+refined grid sweep for {
              len(combos)} combos (this may take from a few minutes to hours)...")

        results = run_combos(lenses, combos, run_date)
        write_results(method, results, run_date)

    # pick best overall
    best = results[np.argmax([r['coupling'] for r in results])]
    print('\nBest combo overall:', best['lens1'], best['lens2'], 'coupling =', best['coupling'])
    print(f"Configuration: z_l1 = {best['z_l1']}, z_l2 = {best['z_l2']}, z_fiber = {best['z_fiber']}")

    # Generate spot diagram for best combo
    plot_spot_diagram(best, lenses)


if __name__ == "__main__":
    main()
