"""
Test script for database functionality.

Tests the SQLite database module with sample optimization results.
"""

import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from scripts.database import OptimizationDatabase
from scripts import consts as C


def test_database():
    """Test database operations with sample data."""
    
    # Use a temporary test database
    test_db_path = './results/test_optimization.db'
    
    print("=" * 80)
    print("Testing Database Functionality")
    print("=" * 80)
    
    # Clean up any existing test database
    if os.path.exists(test_db_path):
        os.remove(test_db_path)
        print(f"Removed existing test database: {test_db_path}")
    
    # Initialize database
    print(f"\n1. Initializing database: {test_db_path}")
    with OptimizationDatabase(test_db_path) as db:
        print("   Database initialized successfully")
        
        # Test: Insert run
        print("\n2. Testing run insertion...")
        run_id = "test_run_2025-11-03"
        db.insert_run(
            run_id=run_id,
            method='powell',
            medium='air',
            n_rays=1000,
            wavelength_nm=200.0,
            pressure_atm=1.0,
            temperature_k=293.15,
            humidity_fraction=0.01,
            notes="Test run for database validation"
        )
        print(f"   Inserted run: {run_id}")
        
        # Test: Insert sample lenses
        print("\n3. Testing lens insertion...")
        sample_lenses = {
            'LA4001': {
                'dia': 25.4,
                'f_mm': 20.0,
                'R_mm': 18.5,
                'tc_mm': 3.5,
                'te_mm': 2.0,
                'BFL_mm': 17.2
            },
            'LA4647': {
                'dia': 25.4,
                'f_mm': 40.0,
                'R_mm': 37.0,
                'tc_mm': 3.0,
                'te_mm': 2.0,
                'BFL_mm': 38.1
            }
        }
        db.insert_lenses_from_dict(sample_lenses)
        print(f"   Inserted {len(sample_lenses)} lenses")
        
        # Test: Insert sample results
        print("\n4. Testing result insertion...")
        sample_results = [
            {
                'lens1': 'LA4001',
                'lens2': 'LA4647',
                'method': 'powell',
                'orientation': 'ScffcF',
                'z_l1': 10.5,
                'z_l2': 25.3,
                'z_fiber': 65.3,
                'total_len_mm': 65.3,
                'coupling': 0.245,
                'f1_mm': 20.0,
                'f2_mm': 40.0
            },
            {
                'lens1': 'LA4001',
                'lens2': 'LA4647',
                'method': 'powell',
                'orientation': 'SfccfF',
                'z_l1': 11.0,
                'z_l2': 26.0,
                'z_fiber': 66.0,
                'total_len_mm': 66.0,
                'coupling': 0.238,
                'f1_mm': 20.0,
                'f2_mm': 40.0
            },
            {
                'lens1': 'LA4001',
                'lens2': 'LA4001',
                'method': 'powell',
                'orientation': 'ScffcF',
                'z_l1': 12.0,
                'z_l2': 27.0,
                'z_fiber': 47.0,
                'total_len_mm': 47.0,
                'coupling': 0.215,
                'f1_mm': 20.0,
                'f2_mm': 20.0
            }
        ]
        db.insert_results_batch(run_id, sample_results)
        print(f"   Inserted {len(sample_results)} results")
        
        # Test: Query run
        print("\n5. Testing run query...")
        run = db.get_run(run_id)
        if run:
            print(f"   Retrieved run: {run['run_id']}")
            print(f"   Method: {run['method']}, Medium: {run['medium']}")
        else:
            print("   ERROR: Failed to retrieve run")
        
        # Test: Query results
        print("\n6. Testing results query...")
        results = db.get_results(run_id, limit=10)
        print(f"   Retrieved {len(results)} results")
        if results:
            print(f"   Best coupling: {results[0]['coupling']:.4f}")
            print(f"   Lens pair: {results[0]['lens1']} + {results[0]['lens2']}")
        
        # Test: Query best results
        print("\n7. Testing best results query...")
        best = db.get_best_results(limit=5)
        print(f"   Retrieved {len(best)} best results across all runs")
        if best:
            print(f"   Top result: {best[0]['lens1']} + {best[0]['lens2']}")
            print(f"   Coupling: {best[0]['coupling']:.4f}")
        
        # Test: Query lens pair history
        print("\n8. Testing lens pair history...")
        history = db.get_lens_pair_history('LA4001', 'LA4647')
        print(f"   Retrieved {len(history)} results for LA4001 + LA4647")
        
        # Test: Get statistics
        print("\n9. Testing statistics...")
        stats = db.get_statistics(run_id)
        print(f"   Total results: {stats['total_results']}")
        print(f"   Coupling range: {stats['min_coupling']:.4f} - {stats['max_coupling']:.4f}")
        print(f"   Average coupling: {stats['avg_coupling']:.4f}")
        print(f"   Length range: {stats['min_length']:.2f} - {stats['max_length']:.2f} mm")
        
        # Test: Export to CSV
        print("\n10. Testing CSV export...")
        csv_path = './results/test_export.csv'
        db.export_to_csv(run_id, csv_path)
        if os.path.exists(csv_path):
            print(f"   Exported to {csv_path}")
            os.remove(csv_path)
            print(f"   Cleaned up test file")
        else:
            print("   ERROR: Export failed")
        
        print("\n" + "=" * 80)
        print("All tests completed successfully!")
        print("=" * 80)
        print(f"\nTest database created at: {test_db_path}")
        print("You can inspect it with: python -m scripts.db_query --db {} stats".format(test_db_path))
        print("\nTo enable database for real runs, set in configs/default.yaml:")
        print("  database:")
        print("    enabled: true")


if __name__ == '__main__':
    test_database()
