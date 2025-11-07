#!/usr/bin/env python3
"""
Add lenses from CSV files to the lens database.

This script provides a simple CLI to import lenses from various CSV formats
into the lens database (data/lenses.db).

Supported formats:
  - ThorLabs Combined Lenses CSV
  - Edmund Optics Condenser Lenses CSV
  - Custom format (requires column mapping)

Usage:
    python -m scripts.add_lenses_from_csv <csv_file> --format <format> [--vendor <vendor_name>]
    
Examples:
    # Import ThorLabs lenses
    python -m scripts.add_lenses_from_csv data/new_thorlabs.csv --format thorlabs --vendor "ThorLabs"
    
    # Import Edmund Optics lenses
    python -m scripts.add_lenses_from_csv data/edmund.csv --format edmund --vendor "Edmund Optics"
    
    # Show what would be imported (dry run)
    python -m scripts.add_lenses_from_csv data/new_lenses.csv --format thorlabs --dry-run
"""

import argparse
import pandas as pd
from pathlib import Path
from scripts.lens_database import LensDatabase


def import_thorlabs_format(csv_path: str, db: LensDatabase, vendor: str = 'ThorLabs', 
                           dry_run: bool = False) -> int:
    """
    Import lenses from ThorLabs-format CSV.
    
    Expected columns:
        - Item #
        - Diameter (mm)
        - Focal Length (mm)
        - Radius of Curvature (mm)
        - Center Thickness (mm)
        - Edge Thickness (mm)
        - Back Focal Length (mm)
    
    Returns:
        Number of lenses imported
    """
    df = pd.read_csv(csv_path)
    
    required_cols = ['Item #', 'Diameter (mm)', 'Focal Length (mm)', 
                    'Radius of Curvature (mm)', 'Center Thickness (mm)', 
                    'Edge Thickness (mm)', 'Back Focal Length (mm)']
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    count = 0
    lenses_data = []
    
    for _, row in df.iterrows():
        lens_data = {
            'item_number': str(row['Item #']),
            'lens_type': 'Plano-Convex',
            'diameter_mm': float(row['Diameter (mm)']),
            'focal_length_mm': float(row['Focal Length (mm)']),
            'radius_r1_mm': float(row['Radius of Curvature (mm)']),
            'radius_r2_mm': None,
            'center_thickness_mm': float(row['Center Thickness (mm)']),
            'edge_thickness_mm': float(row['Edge Thickness (mm)']),
            'back_focal_length_mm': float(row['Back Focal Length (mm)']),
            'vendor': vendor,
            'notes': f'Imported from {Path(csv_path).name}'
        }
        lenses_data.append(lens_data)
        count += 1
    
    if not dry_run:
        for lens_data in lenses_data:
            db.insert_lens(**lens_data)
    
    return count


def import_edmund_format(csv_path: str, db: LensDatabase, vendor: str = 'Edmund Optics',
                        dry_run: bool = False) -> int:
    """
    Import lenses from Edmund Optics-format CSV.
    
    Expected columns:
        - Item #
        - Lens Type (Plano/Convex/Aspheric)
        - Diameter (mm)
        - Focal Length (mm)
        - Radius R1 (mm)
        - Radius R2 (mm) [optional]
        - Center Thickness (mm)
        - Edge Thickness (mm)
        - Back Focal Length (mm)
        - Numerical Aperture [optional]
        - Substrate [optional]
        - Coating [optional]
        - Wavelength Range (nm) [optional]
        - Asphere Diameter (mm) [optional]
    
    Returns:
        Number of lenses imported
    """
    df = pd.read_csv(csv_path)
    
    required_cols = ['Item #', 'Lens Type', 'Diameter (mm)', 'Focal Length (mm)',
                    'Radius R1 (mm)', 'Center Thickness (mm)', 'Edge Thickness (mm)',
                    'Back Focal Length (mm)']
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    type_mapping = {
        'Plano': 'Plano-Convex',
        'Convex': 'Bi-Convex',
        'Aspheric': 'Aspheric'
    }
    
    count = 0
    lenses_data = []
    
    for _, row in df.iterrows():
        # Skip if R1 is missing (essential)
        if pd.isna(row['Radius R1 (mm)']):
            continue
        
        # Skip if edge thickness is missing (required for ray tracing)
        if pd.isna(row['Edge Thickness (mm)']):
            continue
        
        csv_lens_type = str(row['Lens Type']).strip()
        lens_type = type_mapping.get(csv_lens_type, 'Plano-Convex')
        
        # Parse optional R2
        r2_mm = None
        if 'Radius R2 (mm)' in df.columns and not pd.isna(row['Radius R2 (mm)']):
            r2_mm = float(row['Radius R2 (mm)'])
        
        # Parse optional fields
        def get_optional_field(field_name):
            if field_name in df.columns and not pd.isna(row[field_name]):
                val = str(row[field_name]).strip()
                if val and val != 'nan' and val != '':
                    return val
            return None
        
        def get_optional_float(field_name):
            if field_name in df.columns and not pd.isna(row[field_name]):
                try:
                    return float(row[field_name])
                except (ValueError, TypeError):
                    pass
            return None
        
        lens_data = {
            'item_number': str(row['Item #']),
            'lens_type': lens_type,
            'diameter_mm': float(row['Diameter (mm)']),
            'focal_length_mm': float(row['Focal Length (mm)']),
            'radius_r1_mm': float(row['Radius R1 (mm)']),
            'radius_r2_mm': r2_mm,
            'center_thickness_mm': float(row['Center Thickness (mm)']),
            'edge_thickness_mm': float(row['Edge Thickness (mm)']),
            'back_focal_length_mm': float(row['Back Focal Length (mm)']),
            'numerical_aperture': get_optional_float('Numerical Aperture'),
            'substrate': get_optional_field('Substrate'),
            'coating': get_optional_field('Coating'),
            'wavelength_range_nm': get_optional_field('Wavelength Range (nm)'),
            'asphere_diameter_mm': get_optional_float('Asphere Diameter (mm)'),
            'vendor': vendor,
            'notes': f'Imported from {Path(csv_path).name} - Type: {csv_lens_type}'
        }
        lenses_data.append(lens_data)
        count += 1
    
    if not dry_run:
        for lens_data in lenses_data:
            db.insert_lens(**lens_data)
    
    return count


def show_csv_info(csv_path: str):
    """Show information about the CSV file."""
    df = pd.read_csv(csv_path, nrows=5)
    
    print("\nCSV File Information:")
    print(f"  Path: {csv_path}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Total rows: {len(pd.read_csv(csv_path))}")
    print(f"\nFirst few rows:")
    print(df.to_string(index=False))
    print()


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description='Import lenses from CSV files into the lens database',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('csv_file', type=str, 
                       help='Path to CSV file containing lens data')
    parser.add_argument('--format', type=str, choices=['thorlabs', 'edmund'], 
                       required=True,
                       help='CSV format (thorlabs or edmund)')
    parser.add_argument('--vendor', type=str, default=None,
                       help='Vendor name (defaults: ThorLabs or Edmund Optics)')
    parser.add_argument('--db-path', type=str, default='./data/lenses.db',
                       help='Path to lens database (default: ./data/lenses.db)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be imported without making changes')
    parser.add_argument('--show-info', action='store_true',
                       help='Show CSV file information and exit')
    
    args = parser.parse_args()
    
    # Check if CSV file exists
    if not Path(args.csv_file).exists():
        print(f"Error: CSV file not found: {args.csv_file}")
        return 1
    
    # Show CSV info if requested
    if args.show_info:
        show_csv_info(args.csv_file)
        return 0
    
    # Set default vendor based on format
    vendor = args.vendor
    if vendor is None:
        vendor = 'ThorLabs' if args.format == 'thorlabs' else 'Edmund Optics'
    
    print("=" * 70)
    print("LENS IMPORT FROM CSV")
    print("=" * 70)
    print(f"CSV File: {args.csv_file}")
    print(f"Format: {args.format}")
    print(f"Vendor: {vendor}")
    print(f"Database: {args.db_path}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'LIVE IMPORT'}")
    print("=" * 70)
    print()
    
    try:
        # Open database connection
        db = LensDatabase(args.db_path) if not args.dry_run else None
        
        # Get initial count
        initial_count = 0
        final_count = 0
        if db:
            initial_count = db.get_lens_count()
            print(f"Current lens count in database: {initial_count}")
            print()
        
        # Import based on format
        if args.format == 'thorlabs':
            count = import_thorlabs_format(args.csv_file, db, vendor, args.dry_run)
        elif args.format == 'edmund':
            count = import_edmund_format(args.csv_file, db, vendor, args.dry_run)
        else:
            print(f"Error: Unknown format '{args.format}'")
            if db:
                db.close()
            return 1
        
        # Close database
        if db:
            final_count = db.get_lens_count()
            db.close()
        
        # Report results
        print("=" * 70)
        if args.dry_run:
            print(f"DRY RUN: Would import {count} lenses")
        else:
            print(f"✓ Successfully imported {count} lenses")
            print(f"  Previous count: {initial_count}")
            print(f"  New count: {final_count}")
            print(f"  Added: {final_count - initial_count}")
        print("=" * 70)
        
        return 0
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
