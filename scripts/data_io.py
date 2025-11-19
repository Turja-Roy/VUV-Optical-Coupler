import pandas as pd
from scripts import consts as C
from typing import Optional, Dict, List
from scripts.database import OptimizationDatabase
from scripts.lens_database import LensDatabase
from scripts.lens_factory import convert_db_lens_to_dict


def fetch_lens_data_from_db(db: LensDatabase, 
                           lens_type: Optional[str] = None,
                           min_focal_length: Optional[float] = None,
                           max_focal_length: Optional[float] = None,
                           vendor: Optional[str] = None,
                           sql_query: Optional[str] = None) -> Dict:
    """
    Fetch lenses from database with optional filtering.
    
    Parameters:
    -----------
    db : LensDatabase
        Database connection
    lens_type : str, optional
        Filter by lens type ('Plano-Convex', 'Bi-Convex', 'Aspheric')
    min_focal_length : float, optional
        Minimum focal length in mm
    max_focal_length : float, optional
        Maximum focal length in mm
    vendor : str, optional
        Filter by vendor name
    sql_query : str, optional
        Custom SQL query to filter lenses. When provided, other filter parameters are ignored.
        Example: "SELECT * FROM lenses WHERE focal_length_mm BETWEEN 15 AND 30"
        
    Returns:
    --------
    dict : Dictionary mapping item_number to lens data dict (in optimization format)
    """
    # Use custom SQL query if provided
    if sql_query:
        db_lenses = db.execute_custom_query(sql_query)
    else:
        # Use standard filtering
        db_lenses = db.get_all_lenses(
            lens_type=lens_type,
            min_focal_length=min_focal_length,
            max_focal_length=max_focal_length,
            vendor=vendor
        )
    
    lenses = {}
    for db_lens in db_lenses:
        item_number = db_lens['item_number']
        lenses[item_number] = convert_db_lens_to_dict(db_lens)
    
    return lenses


def fetch_lens_data(method, use_database=False, db=None, **db_filters):
    """
    Fetch lens data from CSV files or database and return as dictionaries.
    
    Parameters:
    -----------
    method : str
        Selection method ('select', 'select_ext', 'combine')
    use_database : bool, optional
        If True, fetch from database instead of CSV (default: False)
    db : LensDatabase, optional
        Database connection (required if use_database=True)
    **db_filters : optional
        Additional filters for database queries (lens_type, min_focal_length, sql_query, etc.)
        
    Returns:
    --------
    For 'select' or 'select_ext': (lens1, lens2, lenses) tuple
    For 'combine': lenses dict only
    
    Notes:
    ------
    All returned lens dictionaries now include 'lens_type' field for compatibility
    with lens_factory.create_lens(). Legacy CSV data defaults to 'Plano-Convex'.
    """
    # Database mode
    if use_database:
        if db is None:
            raise ValueError("Database connection required when use_database=True")
        
        lenses = fetch_lens_data_from_db(db, **db_filters)
        
        if method == 'select' or method == 'select_ext':
            # For select mode, split into L1 and L2 based on focal length
            # This is a heuristic - could be improved with explicit L1/L2 designation
            all_focal_lengths = sorted([l['f_mm'] for l in lenses.values()])
            median_f = all_focal_lengths[len(all_focal_lengths) // 2] if all_focal_lengths else 20.0
            
            lens1 = {k: v for k, v in lenses.items() if v['f_mm'] <= median_f}
            lens2 = {k: v for k, v in lenses.items() if v['f_mm'] > median_f}
            return lens1, lens2, lenses
        else:
            return lenses
    
    # CSV mode (legacy)
    lens1, lens2, lenses = {}, {}, {}

    if method == 'select' or method == 'select_ext':
        if method == 'select_ext':
            l1_candidates = pd.read_csv('./data/l1_candidates_ext.csv')
            l2_candidates = pd.read_csv('./data/l2_candidates_ext.csv')
        else:
            l1_candidates = pd.read_csv('./data/l1_candidates.csv')
            l2_candidates = pd.read_csv('./data/l2_candidates.csv')

        for _, row in l1_candidates.iterrows():
            lens1[row['Item #']] = {
                'dia': row['Diameter (mm)'],
                'f_mm': row['Focal Length (mm)'],
                'R_mm': row['Radius of Curvature (mm)'],
                'tc_mm': row['Center Thickness (mm)'],
                'te_mm': row['Edge Thickness (mm)'],
                'BFL_mm': row['Back Focal Length (mm)'],
                'lens_type': 'Plano-Convex'  # CSV data is all plano-convex
            }
        for _, row in l2_candidates.iterrows():
            lens2[row['Item #']] = {
                'dia': row['Diameter (mm)'],
                'f_mm': row['Focal Length (mm)'],
                'R_mm': row['Radius of Curvature (mm)'],
                'tc_mm': row['Center Thickness (mm)'],
                'te_mm': row['Edge Thickness (mm)'],
                'BFL_mm': row['Back Focal Length (mm)'],
                'lens_type': 'Plano-Convex'  # CSV data is all plano-convex
            }
        lenses = lens1 | lens2
        return lens1, lens2, lenses

    else:  # method == 'combine'
        lenses_candidates = pd.read_csv('./data/Combined_Lenses.csv')

        for _, row in lenses_candidates.iterrows():
            lenses[row['Item #']] = {
                'dia': row['Diameter (mm)'],
                'f_mm': row['Focal Length (mm)'],
                'R_mm': row['Radius of Curvature (mm)'],
                'tc_mm': row['Center Thickness (mm)'],
                'te_mm': row['Edge Thickness (mm)'],
                'BFL_mm': row['Back Focal Length (mm)'],
                'lens_type': 'Plano-Convex'  # CSV data is all plano-convex
            }
        return lenses


def find_combos(method, use_database=False, db=None, **db_filters):
    """
    Generate all lens combinations to evaluate (based on method).
    
    Parameters:
    -----------
    method : str
        'select', 'select_ext', or 'combine'
    use_database : bool, optional
        If True, load lenses from database instead of CSV
    db : LensDatabase, optional
        Database instance to use (required if use_database=True)
    **db_filters : dict, optional
        Additional filters for database queries (e.g., lens_type, vendor)
    """
    if method == 'select' or method == 'select_ext':
        lens1, lens2, lenses = fetch_lens_data(method, use_database=use_database, db=db, **db_filters)
        combos = []
        for a in lens1:
            for b in lens2:
                combos.append((a, b))
        return combos, lenses

    else:  # method == 'combine'
        lenses = fetch_lens_data(method, use_database=use_database, db=db, **db_filters)
        combos = []
        # Add two-lens combinations
        for a in lenses:
            for b in lenses:
                combos.append((a, b))
        # Add single-lens combinations (test each lens individually)
        for a in lenses:
            combos.append((a, None))
        return combos, lenses


def particular_combo(name1, name2, use_database=False, db=None):
    """
    Get a specific lens pair for testing.
    
    Parameters:
    -----------
    name1 : str
        First lens item number
    name2 : str
        Second lens item number
    use_database : bool, optional
        If True, fetch from database instead of CSV (default: False)
    db : LensDatabase, optional
        Database connection (required if use_database=True)
        
    Returns:
    --------
    tuple : (combos, lenses) where combos is [(name1, name2)] and lenses is dict
    """
    lens1, lens2, lenses = {}, {}, {}
    
    if use_database:
        if db is None:
            raise ValueError("Database connection required when use_database=True")
        
        # Fetch both lenses from database
        db_lens1 = db.get_lens(name1)
        db_lens2 = db.get_lens(name2)
        
        if db_lens1 is None:
            raise ValueError(f"Lens '{name1}' not found in database")
        if db_lens2 is None:
            raise ValueError(f"Lens '{name2}' not found in database")
        
        lens1[name1] = convert_db_lens_to_dict(db_lens1)
        lens2[name2] = convert_db_lens_to_dict(db_lens2)
        lenses = lens1 | lens2
    else:
        # CSV mode (legacy)
        lens_candidates = pd.read_csv('./data/Combined_Lenses.csv')

        lens1_data = lens_candidates[lens_candidates['Item #'] == name1].iloc[0]
        lens2_data = lens_candidates[lens_candidates['Item #'] == name2].iloc[0]

        lens1[name1] = {
            'dia': lens1_data['Diameter (mm)'],
            'f_mm': lens1_data['Focal Length (mm)'],
            'R_mm': lens1_data['Radius of Curvature (mm)'],
            'tc_mm': lens1_data['Center Thickness (mm)'],
            'te_mm': lens1_data['Edge Thickness (mm)'],
            'BFL_mm': lens1_data['Back Focal Length (mm)'],
            'lens_type': 'Plano-Convex'  # CSV data is all plano-convex
        }
        lens2[name2] = {
            'dia': lens2_data['Diameter (mm)'],
            'f_mm': lens2_data['Focal Length (mm)'],
            'R_mm': lens2_data['Radius of Curvature (mm)'],
            'tc_mm': lens2_data['Center Thickness (mm)'],
            'te_mm': lens2_data['Edge Thickness (mm)'],
            'BFL_mm': lens2_data['Back Focal Length (mm)'],
            'lens_type': 'Plano-Convex'  # CSV data is all plano-convex
        }
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
                  'z_fiber', 'total_len_mm', 'coupling', 'orientation', 'fiber_position_method']
    rows = []
    for i, r in enumerate(results):
        if not isinstance(r, dict):
            raise ValueError(f"result {i} is {type(r)}, expected dict")
        rows.append({k: v for k, v in r.items() if k in scalar_keys})
    df = pd.DataFrame(rows).sort_values(
        ['coupling', 'total_len_mm'],
        ascending=[False, True]).reset_index(drop=True)

    # Write to database if enabled
    # For modes with per-test writes (grid_search), this writes final aggregation
    # For batch modes (select with powell), this writes batch results
    # For analyze modes, this writes all results
    if use_database and db is not None:
        try:
            # Ensure run exists in database first
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
            
            # Write results to database (batch or final)
            # For batch writes, only write if this is a batch chunk (not final aggregation)
            # For non-batch, write the final results
            if batch:
                # This is a batch chunk - write these results
                results_with_method = [dict(r, method=method) for r in rows]
                db.insert_results_batch(run_id, results_with_method)
            elif not batch and 'analyze' in method:
                # Analyze mode - write all results
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
