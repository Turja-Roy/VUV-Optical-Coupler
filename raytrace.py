import sys
import os
import numpy as np
import pandas as pd
import regex as re
from scripts import consts as C
from scripts.fetcher import find_combos, particular_combo, write_results
from scripts.runner import run_combos, run_grid
from scripts.visualizers import plot_spot_diagram


def main():
    batch_run = False
    batch_continue = False
    method = 'combine'  # default
    run_date = C.DATE_STR

    """
        Parse command line arguments
        Usage:
        python raytrace.py [particular <lens1> <lens2> | select | combine] [continue] [<YYYY-MM-DD>]
    """
    if sys.argv[1] == 'particular':                                     # Particular lenses
        method = 'particular'
        combos, lenses = particular_combo(sys.argv[2], sys.argv[3])

    elif sys.argv[1] in ['select', 'combine']:                          # Select or combine
        method = sys.argv[1]

        if 'continue' in sys.argv:
            batch_continue = True

        date_args = [arg for arg in sys.argv if re.match(r'\d{4}-\d{2}-\d{2}', arg)]
        if date_args != []:
            run_date = date_args[0]
        print('Run date: ' + run_date)

        combos, lenses = find_combos(method)
        if len(combos) > 100:
            batch_run = True

    else:                                                               # Invalid arguments
        print("Invalid arguments. Usage:\n"
              "python raytrace.py [particular <lens1> <lens2> | select | combine] [continue] [<YYYY-MM-DD>]")
        return

    """
        Run sweep across all combos (coarse+refine)
    """
    print("Total number of lens combos to evaluate:", len(combos))

    if batch_continue:  # Continue incomplete run
        completed = len([f for f in os.listdir(
            f'./results/{run_date}') if f.startswith('batch_') and f.endswith('.csv')])
        remaining = int(np.ceil((len(combos) - completed * 100) / 100))

        # Check if a temp file (.json) exists for the last incomplete batch
        temp_files = [f for f in os.listdir(
            f'./results/{run_date}') if f.startswith('temp') and f.endswith('.json')]

        if len(temp_files) > 0:
            # Read the JSON file to see how many combos were completed
            with open(f'./results/{run_date}/{temp_files[0]}', 'r') as f:
                try:
                    completed_data = __import__("json").load(f)
                    completed_combos = len(completed_data)
                except __import__("json").JSONDecodeError:
                    completed_combos = 0

            combos = combos[:len(combos) - (completed-1)*100 - len(completed_combos)]
            print(f"Resuming from batch {completed} with {
                  len(combos)} combos remaining...")

            batch_results = run_combos(lenses, combos, run_date, batch_num=completed)
            write_results(method, batch_results, run_date,
                          batch=True, batch_num=completed)

        else:
            combos = combos[completed * 100:]
            print(f"Resuming from batch {
                  completed + 1} with {len(combos)} combos remaining...")

        # Run the remaining batches
        for i in range(remaining):
            batch_combos = combos[:100]
            print(f"\nStarting batch {completed+i+1}/{completed + remaining}")

            batch_results = run_combos(lenses, batch_combos, run_date, batch_num=completed+i+1)
            write_results(method, batch_results, run_date,
                          batch=True, batch_num=completed+i+1)
            combos = combos[100:]

    elif batch_run:  # Start batches from beginning (override existing)
        n_batches = int(np.ceil(len(combos) / 100))

        for i in range(n_batches):
            batch_combos = combos[:100]
            print(f"\nStarting batch {i+1}/{n_batches}")

            batch_results = run_combos(lenses, batch_combos, run_date, batch_num=i+1)
            write_results(method, batch_results, run_date,
                          batch=True, batch_num=i+1)

            combos = combos[100:]

    # Update final result with completed batches
    results = []

    completed = len([f for f in os.listdir(
        f'./results/{run_date}') if f.startswith('batch_') and f.endswith('.csv')])

    for i in range(completed):
        df = pd.read_csv(f'./results/{run_date}/batch_{method}_{i+1}.csv')
        results.extend(df.to_dict('records'))  # get list of row dicts

    write_results(method, results, run_date)

    # pick best overall
    best = results[np.argmax([r['coupling'] for r in results])]
    print('\nBest combo overall:', best['lens1'],
          best['lens2'], 'coupling =', best['coupling'])
    print(f"Configuration: z_l1 = {best['z_l1']}, z_l2 = {
          best['z_l2']}, z_fiber = {best['z_fiber']}")

    # Generate spot diagram for best combo
    # Running the best combo again before plotting to get landing points
    best = run_grid(run_date, lenses, best['lens1'], best['lens2'])
    plot_spot_diagram(best, lenses, run_date)


if __name__ == "__main__":
    main()
