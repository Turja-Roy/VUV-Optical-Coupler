import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

from scripts.data_io import write_results
from scripts import consts as C


def run_batches(combos, lenses, run_id, method, runner_func, db=None):
    """Run optimization in batches."""
    results = []
    n_batches = int(np.ceil(len(combos) / 100))

    for i in range(n_batches):
        batch_combos = combos[i*100:(i+1)*100]
        print(f"\n{'='*60}")
        print(f"Batch {i+1}/{n_batches}")
        print(f"{'='*60}")

        batch_results = runner_func(lenses, batch_combos, run_id, i+1)
        write_results(method, batch_results, run_id,
                      batch=True, batch_num=i+1, db=db)
        results.extend(batch_results)

    write_results(method, results, run_id, db=db)
    return results


def run_batches_continue(combos, lenses, run_id, method, runner_func, db=None):
    """Continue incomplete batch run."""
    results_dir = Path(f'./results/{run_id}')

    completed = len([f for f in results_dir.glob('batch_*.csv')])

    print(f"Found {completed} completed batches")

    results = []
    for i in range(completed):
        df = pd.read_csv(results_dir / f'batch_{method}_{i+1}.csv')
        results.extend(df.to_dict('records'))

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

    start_index = completed * 100 + completed_combos
    remaining_combos = combos[start_index:]
    
    n_total_batches = int(np.ceil(len(combos) / 100))
    starting_batch = completed + 1 if completed_combos > 0 else completed + 1

    print(f"Starting from combo {start_index} (batch {starting_batch})")
    
    current_combo_idx = start_index
    for batch_num in range(starting_batch, n_total_batches + 1):
        batch_start = (batch_num - 1) * 100
        batch_end = min(batch_num * 100, len(combos))
        
        if current_combo_idx > batch_start:
            batch_start = current_combo_idx
        
        if batch_start >= batch_end:
            continue
            
        batch_combos = combos[batch_start:batch_end]
        n_remaining_batches = n_total_batches - batch_num + 1

        print(f"\n{'='*60}")
        print(f"Batch {batch_num}/{n_total_batches} ({len(batch_combos)} combos, {n_remaining_batches} batches remaining)")
        print(f"{'='*60}")

        batch_results = runner_func(lenses, batch_combos, run_id, batch_num)
        write_results(method, batch_results, run_id,
                      batch=True, batch_num=batch_num, db=db)
        results.extend(batch_results)
        
        current_combo_idx = batch_end

    write_results(method, results, run_id, db=db)
    return results
