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

    return combos, lenses
