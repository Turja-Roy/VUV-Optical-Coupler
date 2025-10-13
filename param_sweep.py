import numpy as np
from pathlib import Path
import sys
import os

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from scripts.runner import run_grid
from scripts.fetcher import fetch_lens_data

def test_param_sensitivity():
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Load lenses
    lenses = fetch_lens_data('combine')
    
    # Choose a representative lens pair
    lens1 = 'LA4001'  # Example lens, replace with your common test case
    lens2 = 'LA4966'
    
    # Test configurations with different strategies
    param_sets = [
        # Format: (coarse_steps, refine_steps, n_coarse, n_refine, jitter)
        # Standard grid search (baseline)
        (11, 15, 5000, 10000, 0.0),    # Moderate parameters, no jitter
        
        # Fine grid with moderate rays
        (21, 31, 5000, 10000, 0.0),    # More grid points, same ray count
        
        # Moderate grid with jitter
        (11, 15, 5000, 10000, 0.1),    # Add 10% random jitter to positions
        (11, 15, 5000, 10000, 0.2),    # Add 20% random jitter to positions
        
        # Combined approach
        (15, 21, 5000, 10000, 0.1),    # Balanced approach with jitter
    ]
    
    print("Testing different search strategies:")
    print("coarse_steps, refine_steps, n_coarse, n_refine, jitter, coupling")
    
    for params in param_sets:
        # Add jitter to the original run_grid function by modifying z positions
        def run_grid_with_jitter(date, lenses, l1, l2, cs, rs, nc, nr, jitter):
            result = run_grid(date, lenses, l1, l2, cs, rs, nc, nr)
            if result and jitter > 0:
                # Try positions with random jitter around the best found
                z1, z2 = result['z_l1'], result['z_l2']
                for _ in range(10):  # Try 10 jittered positions
                    z1_new = z1 * (1 + np.random.uniform(-jitter, jitter))
                    z2_new = z2 * (1 + np.random.uniform(-jitter, jitter))
                    jittered = run_grid(date, lenses, l1, l2, cs, rs, nc, nr,
                                      z1_init=z1_new, z2_init=z2_new)
                    if jittered and jittered['coupling'] > result['coupling']:
                        result = jittered
            return result
        
        result = run_grid_with_jitter("2025-10-13", lenses, lens1, lens2,
                                    params[0], params[1], params[2], params[3], params[4])
        
        if result:
            print(f"{params[0]}, {params[1]}, {params[2]}, {params[3]}, {params[4]}, {result['coupling']:.6f}")
            
        # Validate the best result with high ray count
        if result and result['coupling'] > 0.3:  # Only validate promising results
            print(f"\nValidating best result (coupling={result['coupling']:.6f}) with 50000 rays...")
            validation = run_grid("2025-10-13", lenses, lens1, lens2,
                                coarse_steps=1, refine_steps=1,  # Just evaluate the best point
                                n_coarse=50000, n_refine=50000,
                                z1_init=result['z_l1'], z2_init=result['z_l2'])
            if validation:
                print(f"Validated coupling: {validation['coupling']:.6f}")

if __name__ == '__main__':
    test_param_sensitivity()
