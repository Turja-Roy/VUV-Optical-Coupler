#!/usr/bin/env python3
"""
Migration script to import historical CSV results into the optimization database.

This script scans the results directory for CSV files and imports them into the
database, preserving run metadata and result data.
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
import argparse
import sys
import re
from typing import Dict, List, Optional

from scripts.database import OptimizationDatabase
from scripts import consts as C


def parse_run_id(dirname: str) -> Dict[str, str]:
    """
    Parse run_id directory name to extract metadata.
    
    Examples:
        2025-10-20_select_powell_air -> {date: 2025-10-20, mode: select, method: powell, medium: air}
        analyze_2025-10-21_coupling_0_24_argon -> {date: 2025-10-21, mode: analyze, medium: argon}
        particular_2025-11-07_powell_air -> {date: 2025-11-07, mode: particular, method: powell, medium: air}
        wavelength_analyze_2025-10-21 -> {date: 2025-10-21, mode: wavelength_analyze}
    """
    metadata = {}
    
    # Extract date (YYYY-MM-DD pattern)
    date_match = re.search(r'(\d{4}-\d{2}-\d{2})', dirname)
    if date_match:
        metadata['date'] = date_match.group(1)
    
    # Extract medium (air or argon)
    if '_air' in dirname:
        metadata['medium'] = 'air'
    elif '_argon' in dirname:
        metadata['medium'] = 'argon'
    else:
        metadata['medium'] = 'air'  # default
    
    # Extract mode and method
    if dirname.startswith('analyze_'):
        metadata['mode'] = 'analyze'
        metadata['method'] = 'analyze'
    elif dirname.startswith('wavelength_analyze_'):
        metadata['mode'] = 'wavelength_analyze'
        metadata['method'] = 'wavelength_analyze'
    elif dirname.startswith('particular_'):
        metadata['mode'] = 'particular'
        # Extract method from particular_DATE_METHOD_MEDIUM
        parts = dirname.split('_')
        if len(parts) >= 4:
            metadata['method'] = parts[3]
        else:
            metadata['method'] = 'unknown'
    elif '_select_' in dirname:
        metadata['mode'] = 'select'
        # Extract method from DATE_select_METHOD_MEDIUM
        parts = dirname.split('_')
        if len(parts) >= 3:
            metadata['method'] = parts[2]
        else:
            metadata['method'] = 'unknown'
    else:
        metadata['mode'] = 'unknown'
        metadata['method'] = 'unknown'
    
    return metadata


def import_csv_file(csv_path: Path, run_id: str, db: OptimizationDatabase, 
                   metadata: Dict, verbose: bool = False) -> int:
    """
    Import a single CSV file into the database.
    
    Returns:
        Number of results imported
    """
    try:
        df = pd.read_csv(csv_path)
        
        if df.empty:
            if verbose:
                print(f"  Skipping empty file: {csv_path.name}")
            return 0
        
        # Ensure run exists
        existing_run = db.get_run(run_id)
        if not existing_run:
            # Create run metadata
            timestamp = metadata.get('date', datetime.now().strftime('%Y-%m-%d'))
            config = {'migrated_from_csv': True}
            
            db.insert_run(
                run_id=run_id,
                method=metadata.get('method', 'unknown'),
                medium=metadata.get('medium', 'air'),
                n_rays=C.N_RAYS,
                wavelength_nm=C.WAVELENGTH_NM,
                pressure_atm=C.PRESSURE_ATM,
                temperature_k=C.TEMPERATURE_K,
                humidity_fraction=C.HUMIDITY_FRACTION,
                config=config
            )
            if verbose:
                print(f"  Created run: {run_id}")
        
        # Convert DataFrame to list of dicts
        results = []
        for _, row in df.iterrows():
            result = {
                'lens1': str(row.get('lens1', '')),
                'lens2': str(row.get('lens2', '')),
                'method': str(row.get('method', metadata.get('method', 'unknown'))),
                'orientation': str(row.get('orientation', '')),
                'z_l1': float(row.get('z_l1') if pd.notna(row.get('z_l1')) else 0),
                'z_l2': float(row.get('z_l2') if pd.notna(row.get('z_l2')) else 0),
                'z_fiber': float(row.get('z_fiber') if pd.notna(row.get('z_fiber')) else 0),
                'total_len_mm': float(row.get('total_len_mm') if pd.notna(row.get('total_len_mm')) else 0),
                'coupling': float(row.get('coupling') if pd.notna(row.get('coupling')) else 0),
                'f1_mm': float(row.get('f1_mm') if pd.notna(row.get('f1_mm')) else 0),
                'f2_mm': float(row.get('f2_mm') if pd.notna(row.get('f2_mm')) else 0)
            }
            results.append(result)
        
        # Insert batch
        db.insert_results_batch(run_id, results)
        
        if verbose:
            print(f"  Imported {len(results)} results from {csv_path.name}")
        
        return len(results)
        
    except Exception as e:
        print(f"  ERROR importing {csv_path}: {e}")
        return 0


def import_run_directory(run_dir: Path, db: Optional[OptimizationDatabase], 
                        verbose: bool = False, dry_run: bool = False) -> tuple[int, int]:
    """
    Import all CSV files from a run directory.
    
    Returns:
        (total_files, total_results)
    """
    run_id = run_dir.name
    metadata = parse_run_id(run_id)
    
    print(f"\n{'='*70}")
    print(f"Processing: {run_id}")
    print(f"  Mode: {metadata.get('mode', 'unknown')}")
    print(f"  Method: {metadata.get('method', 'unknown')}")
    print(f"  Medium: {metadata.get('medium', 'unknown')}")
    print(f"  Date: {metadata.get('date', 'unknown')}")
    print(f"{'='*70}")
    
    if dry_run:
        print("  [DRY RUN - no database changes will be made]")
    
    csv_files = list(run_dir.glob('*.csv'))
    
    if not csv_files:
        print(f"  No CSV files found")
        return 0, 0
    
    total_results = 0
    for csv_file in csv_files:
        if not dry_run and db is not None:
            results_count = import_csv_file(csv_file, run_id, db, metadata, verbose)
            total_results += results_count
        else:
            # In dry run, just count rows
            df = pd.read_csv(csv_file)
            total_results += len(df)
            if verbose:
                print(f"  Would import {len(df)} results from {csv_file.name}")
    
    print(f"  Total: {len(csv_files)} files, {total_results} results")
    
    return len(csv_files), total_results


def main():
    parser = argparse.ArgumentParser(
        description='Migrate CSV results to optimization database',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run to see what would be imported
  python scripts/migrate_csv_to_db.py --dry-run
  
  # Import all CSV files
  python scripts/migrate_csv_to_db.py
  
  # Import specific run directory
  python scripts/migrate_csv_to_db.py --run 2025-11-05_select_powell_argon
  
  # Use custom database path
  python scripts/migrate_csv_to_db.py --db results/my_database.db
  
  # Verbose output
  python scripts/migrate_csv_to_db.py -v
        """
    )
    
    parser.add_argument(
        '--db',
        type=str,
        default='./results/optimization.db',
        help='Path to optimization database (default: ./results/optimization.db)'
    )
    
    parser.add_argument(
        '--results-dir',
        type=str,
        default='./results',
        help='Path to results directory (default: ./results)'
    )
    
    parser.add_argument(
        '--run',
        type=str,
        help='Import specific run directory only (e.g., "2025-11-05_select_powell_argon")'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be imported without making changes'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Verbose output'
    )
    
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        help='Skip runs that already exist in database'
    )
    
    args = parser.parse_args()
    
    # Validate paths
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"ERROR: Results directory not found: {results_dir}")
        sys.exit(1)
    
    # Connect to database
    if not args.dry_run:
        print(f"\nConnecting to database: {args.db}")
        db = OptimizationDatabase(args.db)
    else:
        print(f"\nDRY RUN MODE - Database: {args.db}")
        db = None
    
    # Get list of run directories to process
    if args.run:
        run_dirs = [results_dir / args.run]
        if not run_dirs[0].exists():
            print(f"ERROR: Run directory not found: {run_dirs[0]}")
            sys.exit(1)
    else:
        # Get all subdirectories (skip database files and other non-run items)
        run_dirs = [
            d for d in results_dir.iterdir() 
            if d.is_dir() and not d.name.startswith('.')
        ]
    
    if not run_dirs:
        print("No run directories found to process")
        sys.exit(0)
    
    print(f"\nFound {len(run_dirs)} run directories to process")
    
    # Process each run directory
    total_files = 0
    total_results = 0
    skipped = 0
    
    for run_dir in sorted(run_dirs):
        if args.skip_existing and db is not None:
            existing_run = db.get_run(run_dir.name)
            if existing_run:
                print(f"\nSkipping existing run: {run_dir.name}")
                skipped += 1
                continue
        
        files, results = import_run_directory(run_dir, db, args.verbose, args.dry_run)
        total_files += files
        total_results += results
    
    # Summary
    print(f"\n{'='*70}")
    print("MIGRATION SUMMARY")
    print(f"{'='*70}")
    print(f"Processed directories: {len(run_dirs) - skipped}")
    if skipped > 0:
        print(f"Skipped (already in DB): {skipped}")
    print(f"Total CSV files: {total_files}")
    print(f"Total results: {total_results}")
    
    if args.dry_run:
        print("\n[DRY RUN COMPLETE - No changes made]")
        print("Run without --dry-run to actually import data")
    else:
        print("\n[MIGRATION COMPLETE]")
    
    if db is not None:
        db.close()


if __name__ == '__main__':
    main()
