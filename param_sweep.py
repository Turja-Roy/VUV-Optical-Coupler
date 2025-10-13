import numpy as np
from pathlib import Path
import sys

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
    lens1 = 'LA4001'
    lens2 = 'LA4966'
    
    # Test configurations with different parameter combinations
    param_sets = [
        # Test different ray counts with same grid size
        (7, 9, 500, 1000),     # Baseline grid, lower rays
        (7, 9, 2000, 6000),    # Baseline grid, medium rays
        # (7, 9, 5000, 8000),    # Baseline grid, higher rays
        
        # Test different grid sizes with medium rays
        # (5, 7, 5000, 10000),     # Coarser grid
        # (9, 13, 5000, 10000),    # Medium grid
        # (11, 15, 5000, 10000),   # Finer grid
        
        # Test balanced combinations
        # (9, 11, 3000, 8000),     # Balanced moderate
        # (11, 13, 4000, 9000),    # Balanced medium
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
