import pandas as pd


def fetch_lens_data(method):
    """Fetch lens data from CSV files and return as dictionaries."""
    lens1, lens2, lenses = {}, {}, {}

    if method == 'select':
        l1_candidates = pd.read_csv('./data/l1_candidates.csv')
        l2_candidates = pd.read_csv('./data/l2_candidates.csv')

        for _, row in l1_candidates.iterrows():
            lens1[row['Item #']] = {'dia': row['Diameter (mm)'],
                                    'f_mm': row['Focal Length (mm)'],
                                    'R_mm': row['Radius of Curvature (mm)'],
                                    't_mm': row['Center Thickness (mm)'],
                                    'BFL_mm': row['Back Focal Length (mm)']}
        for _, row in l2_candidates.iterrows():
            lens2[row['Item #']] = {'dia': row['Diameter (mm)'],
                                    'f_mm': row['Focal Length (mm)'],
                                    'R_mm': row['Radius of Curvature (mm)'],
                                    't_mm': row['Center Thickness (mm)'],
                                    'BFL_mm': row['Back Focal Length (mm)']}
        lenses = lens1 | lens2
        return lens1, lens2, lenses

    else:  # method == 'combine'
        lenses_candidates = pd.read_csv('./data/Combined_Lenses.csv')

        for _, row in lenses_candidates.iterrows():
            lenses[row['Item #']] = {'dia': row['Diameter (mm)'],
                                     'f_mm': row['Focal Length (mm)'],
                                     'R_mm': row['Radius of Curvature (mm)'],
                                     't_mm': row['Center Thickness (mm)'],
                                     'BFL_mm': row['Back Focal Length (mm)']}
        return lenses


def find_combos(method):
    """Generate all lens combinations to evaluate (based on method)"""
    if method == 'select':
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
                    't_mm': lens1_data['Center Thickness (mm)'],
                    'BFL_mm': lens1_data['Back Focal Length (mm)']}
    lens2[name2] = {'dia': lens2_data['Diameter (mm)'],
                    'f_mm': lens2_data['Focal Length (mm)'],
                    'R_mm': lens2_data['Radius of Curvature (mm)'],
                    't_mm': lens2_data['Center Thickness (mm)'],
                    'BFL_mm': lens2_data['Back Focal Length (mm)']}
    lenses = lens1 | lens2

    combos = []
    for a in lenses:
        for b in lenses:
            combos.append((a, b))
    combos = [combos[1]]

    return combos, lenses


def write_results(method, results, run_date, batch=False, batch_num=None, contd_batch=False):
    """Write results to CSV file."""
    import os
    from . import consts as C

    rows = [{k: v for k, v in r.items()
             if k in ['lens1', 'lens2', 'f1_mm', 'f2_mm',
                      'z_l1', 'z_l2', 'z_fiber',
                      'total_len_mm', 'coupling', 'accepted']}
            for r in results]
    df = pd.DataFrame(rows).sort_values(
        ['coupling', 'total_len_mm'],
        ascending=[False, True]).reset_index(drop=True)

    if batch and batch_num is not None:
        if not os.path.exists('./results/' + run_date):
            os.makedirs('./results/' + run_date)

        if contd_batch:  # Append to existing batch file
            existing_df = pd.read_csv(f"results/{run_date}/batch_{batch_num}.csv")
            df = pd.concat([existing_df, df]).drop_duplicates().reset_index(drop=True)
            df.to_csv(f"results/{run_date}/batch_{method}_{batch_num}.csv", index=False)
        else:
            df.to_csv(f"results/{run_date}/batch_{method}_{batch_num}.csv", index=False)
    else:
        if not os.path.exists('./results/' + run_date):
            os.makedirs('./results/' + run_date)
        df.to_csv(f"results/{run_date}/results_{method}_{run_date}.csv", index=False)
