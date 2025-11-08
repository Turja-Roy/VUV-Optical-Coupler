"""
Import lens data from CSV files into the unified SQLite database.

This script handles importing lenses from multiple sources:
- ThorLabs plano-convex lenses (Combined_Lenses.csv)
- Edmund Optics condenser lenses (plano-convex, bi-convex, aspheric)
"""

import pandas as pd
from pathlib import Path
from typing import Optional
from scripts.lens_database import LensDatabase


def import_thorlabs_plano_convex(db: LensDatabase, 
                                 csv_path: str = './data/Combined_Lenses.csv') -> int:
    """
    Import ThorLabs plano-convex lenses from Combined_Lenses.csv.
    
    Parameters:
    -----------
    db : LensDatabase
        Database connection
    csv_path : str
        Path to the CSV file
        
    Returns:
    --------
    int : Number of lenses imported
    """
    if not Path(csv_path).exists():
        print(f"Warning: {csv_path} not found, skipping ThorLabs import")
        return 0
    
    df = pd.read_csv(csv_path)
    count = 0
    
    for _, row in df.iterrows():
        db.insert_lens(
            item_number=str(row['Item #']),
            lens_type='Plano-Convex',
            diameter_mm=float(row['Diameter (mm)']),
            focal_length_mm=float(row['Focal Length (mm)']),
            radius_r1_mm=float(row['Radius of Curvature (mm)']),
            radius_r2_mm=None,  # Plano-convex has no R2
            center_thickness_mm=float(row['Center Thickness (mm)']),
            edge_thickness_mm=float(row['Edge Thickness (mm)']),
            back_focal_length_mm=float(row['Back Focal Length (mm)']),
            vendor='ThorLabs'
        )
        count += 1
    
    print(f"Imported {count} ThorLabs plano-convex lenses")
    return count


def import_edmund_optics_condenser(db: LensDatabase,
                                   csv_path: str = './data/Edmund-Optics_condenser-lenses.csv') -> int:
    """
    Import Edmund Optics condenser lenses (plano-convex, bi-convex, aspheric).
    
    The CSV contains a 'Lens Type' column with values:
    - "Plano" → mapped to "Plano-Convex"
    - "Convex" → mapped to "Bi-Convex"
    - "Aspheric" → mapped to "Aspheric"
    
    Parameters:
    -----------
    db : LensDatabase
        Database connection
    csv_path : str
        Path to the CSV file
        
    Returns:
    --------
    int : Number of lenses imported
    """
    if not Path(csv_path).exists():
        print(f"Warning: {csv_path} not found, skipping Edmund Optics import")
        return 0
    
    df = pd.read_csv(csv_path)
    count = 0
    
    # Mapping from CSV "Lens Type" to our standard types
    type_mapping = {
        'Plano': 'Plano-Convex',
        'Convex': 'Bi-Convex',
        'Aspheric': 'Aspheric'
    }
    
    for _, row in df.iterrows():
        # Map lens type
        csv_lens_type = str(row['Lens Type']).strip()
        lens_type = type_mapping.get(csv_lens_type, 'Plano-Convex')
        
        # Parse radii - skip if R1 is missing (essential for lens)
        if pd.isna(row['Radius R1 (mm)']):
            continue  # Skip lenses without R1
        r1_mm = float(row['Radius R1 (mm)'])
        
        # R2 may be empty or contain a value
        r2_mm = None
        if not pd.isna(row['Radius R2 (mm)']):
            r2_mm = float(row['Radius R2 (mm)'])
        
        # Parse optional fields
        wavelength_range = str(row['Wavelength Range (nm)']).strip()
        if wavelength_range == '' or wavelength_range == 'nan':
            wavelength_range = None
        
        coating = str(row['Coating']).strip()
        if coating == '' or coating == 'nan':
            coating = None
        
        substrate = str(row['Substrate']).strip()
        if substrate == '' or substrate == 'nan':
            substrate = None
        
        asphere_dia_str = str(row['Asphere Diameter (mm)']).strip()
        asphere_diameter_mm = None
        if asphere_dia_str and asphere_dia_str != '' and asphere_dia_str != 'nan':
            try:
                asphere_diameter_mm = float(asphere_dia_str)
            except (ValueError, TypeError):
                asphere_diameter_mm = None
        
        numerical_aperture_str = str(row['Numerical Aperture']).strip()
        numerical_aperture = None
        if numerical_aperture_str and numerical_aperture_str != '' and numerical_aperture_str != 'nan':
            try:
                numerical_aperture = float(numerical_aperture_str)
            except (ValueError, TypeError):
                numerical_aperture = None
        
        # Parse edge thickness (may be missing) - skip if missing (required for ray tracing)
        if pd.isna(row['Edge Thickness (mm)']):
            continue
        edge_thickness_mm = float(row['Edge Thickness (mm)'])
        
        db.insert_lens(
            item_number=str(row['Item #']),
            lens_type=lens_type,
            diameter_mm=float(row['Diameter (mm)']),
            focal_length_mm=float(row['Focal Length (mm)']),
            radius_r1_mm=r1_mm,
            radius_r2_mm=r2_mm,
            center_thickness_mm=float(row['Center Thickness (mm)']),
            edge_thickness_mm=edge_thickness_mm,
            back_focal_length_mm=float(row['Back Focal Length (mm)']),
            numerical_aperture=numerical_aperture,
            substrate=substrate,
            coating=coating,
            wavelength_range_nm=wavelength_range,
            asphere_diameter_mm=asphere_diameter_mm,
            conic_constant=None,  # Not in this CSV
            vendor='Edmund Optics'
        )
        count += 1
    
    print(f"Imported {count} Edmund Optics lenses")
    return count


def get_lens_statistics(db: LensDatabase) -> dict:
    """
    Get statistics about lenses in the database.
    
    Returns:
    --------
    dict : Statistics by lens type and vendor
    """
    all_lenses = db.get_all_lenses()
    
    stats = {
        'total': len(all_lenses),
        'by_type': {},
        'by_vendor': {}
    }
    
    for lens in all_lenses:
        lens_type = lens.get('lens_type', 'Unknown')
        vendor = lens.get('vendor', 'Unknown')
        
        stats['by_type'][lens_type] = stats['by_type'].get(lens_type, 0) + 1
        stats['by_vendor'][vendor] = stats['by_vendor'].get(vendor, 0) + 1
    
    return stats


def import_all_lenses(db: LensDatabase, verbose: bool = True) -> dict:
    """
    Import all lens data from all known CSV sources.
    
    Parameters:
    -----------
    db : LensDatabase
        Database connection
    verbose : bool
        Print detailed statistics
        
    Returns:
    --------
    dict : Import statistics
    """
    print("=" * 60)
    print("Importing lens data to database...")
    print("=" * 60)
    
    thorlabs_count = import_thorlabs_plano_convex(db)
    edmund_count = import_edmund_optics_condenser(db)
    
    total_imported = thorlabs_count + edmund_count
    
    print("\n" + "=" * 60)
    print(f"Import complete: {total_imported} lenses imported")
    print("=" * 60)
    
    if verbose:
        stats = get_lens_statistics(db)
        print(f"\nDatabase Statistics:")
        print(f"  Total lenses: {stats['total']}")
        print(f"\n  By Type:")
        for lens_type, count in sorted(stats['by_type'].items()):
            print(f"    {lens_type}: {count}")
        print(f"\n  By Vendor:")
        for vendor, count in sorted(stats['by_vendor'].items()):
            print(f"    {vendor}: {count}")
    
    return {
        'thorlabs': thorlabs_count,
        'edmund_optics': edmund_count,
        'total': total_imported,
        'statistics': get_lens_statistics(db) if verbose else None
    }


def main():
    """Main function for standalone execution."""
    db = LensDatabase()
    result = import_all_lenses(db, verbose=True)
    db.close()
    
    print("\n" + "=" * 60)
    print("Lens import completed successfully!")
    print("=" * 60)
    
    return result


if __name__ == '__main__':
    main()
