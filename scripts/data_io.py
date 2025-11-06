import pandas as pd
from scripts import consts as C
from typing import Optional


def fetch_lens_data(method):
    """Fetch lens data from CSV files and return as dictionaries."""
    lens1, lens2, lenses = {}, {}, {}

    if method == 'select' or method == 'select_ext':
        if method == 'select_ext':
            l1_candidates = pd.read_csv('./data/l1_candidates_ext.csv')
            l2_candidates = pd.read_csv('./data/l2_candidates_ext.csv')
        else:
            l1_candidates = pd.read_csv('./data/l1_candidates.csv')
            l2_candidates = pd.read_csv('./data/l2_candidates.csv')

        for _, row in l1_candidates.iterrows():
            lens1[row['Item #']] = {'dia': row['Diameter (mm)'],
                                    'f_mm': row['Focal Length (mm)'],
                                    'R_mm': row['Radius of Curvature (mm)'],
                                    'tc_mm': row['Center Thickness (mm)'],
                                    'te_mm': row['Edge Thickness (mm)'],
                                    'BFL_mm': row['Back Focal Length (mm)']}
        for _, row in l2_candidates.iterrows():
            lens2[row['Item #']] = {'dia': row['Diameter (mm)'],
                                    'f_mm': row['Focal Length (mm)'],
                                    'R_mm': row['Radius of Curvature (mm)'],
                                    'tc_mm': row['Center Thickness (mm)'],
                                    'te_mm': row['Edge Thickness (mm)'],
                                    'BFL_mm': row['Back Focal Length (mm)']}
        lenses = lens1 | lens2
        return lens1, lens2, lenses

    else:  # method == 'combine'
        lenses_candidates = pd.read_csv('./data/Combined_Lenses.csv')

        for _, row in lenses_candidates.iterrows():
            lenses[row['Item #']] = {'dia': row['Diameter (mm)'],
                                     'f_mm': row['Focal Length (mm)'],
                                     'R_mm': row['Radius of Curvature (mm)'],
                                     'tc_mm': row['Center Thickness (mm)'],
                                     'te_mm': row['Edge Thickness (mm)'],
                                     'BFL_mm': row['Back Focal Length (mm)']}
        return lenses


def find_combos(method):
    """Generate all lens combinations to evaluate (based on method)"""
    if method == 'select' or method == 'select_ext':
        lens1, lens2, lenses = fetch_lens_data(method)
        combos = []
        for a in lens1:
            for b in lens2:
                combos.append((a, b))
        return combos, lenses

    else:  # method == 'combine'
        lenses = fetch_lens_data(method)
        combos = []
        for a in lenses:
            for b in lenses:
                combos.append((a, b))
        return combos, lenses


def particular_combo(name1, name2):
    lens_candidates = pd.read_csv('./data/Combined_Lenses.csv')

    lens1, lens2, lenses = {}, {}, {}

    lens1_data = lens_candidates[lens_candidates['Item #'] == name1].iloc[0]
    lens2_data = lens_candidates[lens_candidates['Item #'] == name2].iloc[0]

    lens1[name1] = {'dia': lens1_data['Diameter (mm)'],
                    'f_mm': lens1_data['Focal Length (mm)'],
                    'R_mm': lens1_data['Radius of Curvature (mm)'],
                    'tc_mm': lens1_data['Center Thickness (mm)'],
                    'te_mm': lens1_data['Edge Thickness (mm)'],
                    'BFL_mm': lens1_data['Back Focal Length (mm)']}
    lens2[name2] = {'dia': lens2_data['Diameter (mm)'],
                    'f_mm': lens2_data['Focal Length (mm)'],
                    'R_mm': lens2_data['Radius of Curvature (mm)'],
                    'tc_mm': lens2_data['Center Thickness (mm)'],
                    'te_mm': lens2_data['Edge Thickness (mm)'],
                    'BFL_mm': lens2_data['Back Focal Length (mm)']}
    lenses = lens1 | lens2

    # Create the specific combination requested
    combos = [(name1, name2)]

    return combos, lenses


def write_results(method, results, run_id, batch=False, batch_num=None, lens_pair=None, 
                 use_database=None, db=None, alpha=None):
    """
        Write results to CSV file and optionally to database. Each result must be a 
        dictionary with scalar values for lens parameters, positions, and coupling efficiency.
        
        Note: If results are being saved per-test in optimization functions, this will
        skip database insertion to avoid duplicates and only write CSV files.
    """
    import os
    from datetime import datetime

    # Validate results is a sequence of dictionaries
    if not results:
        return
    if not hasattr(results, '__iter__'):
        raise ValueError("results must be a sequence")
    
    # Check if database should be used
    if use_database is None:
        use_database = getattr(C, 'USE_DATABASE', False)
    
    scalar_keys = ['lens1', 'lens2', 'f1_mm', 'f2_mm', 'z_l1', 'z_l2',
                  'z_fiber', 'total_len_mm', 'coupling', 'orientation']
    rows = []
    for i, r in enumerate(results):
        if not isinstance(r, dict):
            raise ValueError(f"result {i} is {type(r)}, expected dict")
        rows.append({k: v for k, v in r.items() if k in scalar_keys})
    df = pd.DataFrame(rows).sort_values(
        ['coupling', 'total_len_mm'],
        ascending=[False, True]).reset_index(drop=True)

    # Write to database if enabled AND it's a final aggregation (not a batch)
    # Per-test writes are now handled in the optimization functions themselves
    # This only writes to DB for analyze mode or final aggregations
    if use_database and db is not None and not batch:
        try:
            # Ensure run exists in database first
            # Check if run already exists
            existing_run = db.get_run(run_id)
            if not existing_run:
                # Insert run metadata (use alpha if provided, otherwise use default 0.7)
                alpha_value = alpha if alpha is not None else 0.7
                config = {'alpha': alpha_value}
                db.insert_run(
                    run_id=run_id,
                    method=method,
                    medium=C.MEDIUM,
                    n_rays=C.N_RAYS,
                    wavelength_nm=C.WAVELENGTH_NM,
                    pressure_atm=C.PRESSURE_ATM,
                    temperature_k=C.TEMPERATURE_K,
                    humidity_fraction=C.HUMIDITY_FRACTION,
                    config=config
                )
            
            # For analyze mode, check if results need to be written to database
            # (per-test writes don't happen in analyze mode)
            if 'analyze' in method:
                # Add method to each result dict for database storage
                results_with_method = [dict(r, method=method) for r in rows]
                db.insert_results_batch(run_id, results_with_method)
        except Exception as e:
            print(f"Warning: Failed to write to database: {e}")
    
    # Always write CSV files as backup
    if batch and batch_num is not None:
        if not os.path.exists('./results/' + run_id):
            os.makedirs('./results/' + run_id)
        df.to_csv(f"results/{run_id}/batch_{method}_{batch_num}.csv", index=False)
    else:
        if not os.path.exists('./results/' + run_id):
            os.makedirs('./results/' + run_id)
        
        if lens_pair is not None:
            timestamp = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
            filename = f"results_{timestamp}_{lens_pair[0]}+{lens_pair[1]}.csv"
        else:
            filename = f"results_{method}_{run_id}.csv"
        
        df.to_csv(f"results/{run_id}/{filename}", index=False)


def write_temp(result, run_id, batch_num):
    """Append a single result (dict) to a temporary json file."""
    import os
    import json
    import numpy as np

    if not os.path.exists('./results/' + run_id):
        os.makedirs('./results/' + run_id)

    filename = 'temp.json' if batch_num is None else f'temp_batch_{batch_num}.json'
    filepath = f'./results/{run_id}/{filename}'

    # Convert numpy arrays to lists for JSON serialization
    serializable_result = {}
    for k, v in result.items():
        if isinstance(v, np.ndarray):
            serializable_result[k] = v.tolist()
        elif isinstance(v, np.bool_):
            serializable_result[k] = bool(v)
        else:
            serializable_result[k] = v

    # Load existing data if file exists
    existing_data = []
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r') as f:
                existing_data = json.load(f)
        except json.JSONDecodeError:
            pass  # Start fresh if file is corrupted

    # Append new result
    existing_data.append(serializable_result)

    # Write back all data
    with open(filepath, 'w') as f:
        json.dump(existing_data, f, indent=2)
