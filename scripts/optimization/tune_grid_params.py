import numpy as np
from pathlib import Path

from scripts.optimization.grid_search import run_grid
from scripts.data_io import fetch_lens_data

def test_param_sensitivity():
    np.random.seed(42)
    
    lenses = fetch_lens_data('combine')
    
    lens1 = 'LA4022'
    lens2 = 'LA4034'
    
    param_sets = [
        (7, 9, 500, 1000),
        (7, 9, 2000, 6000),
        (9, 11, 5000, 8000),
        (9, 11, 5000, 10000),
    ]
    
    print("Testing different search strategies:")
    print("coarse_steps, refine_steps, n_coarse, n_refine, coupling")
    
    for params in param_sets:
        result = run_grid("2025-10-13", lenses, lens1, lens2, 
                         params[0], params[1], params[2], params[3])
        if result:
            print(f"{params[0]}, {params[1]}, {params[2]}, {params[3]}, {result['coupling']:.6f}")

if __name__ == '__main__':
    test_param_sensitivity()
