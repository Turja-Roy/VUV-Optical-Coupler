"""
Migration script to move lens data from optimization.db to lenses.db.

This script:
1. Exports all lens data from results/optimization.db (lenses table)
2. Creates data/lenses.db and imports all lens data
3. Deletes the lenses table from results/optimization.db
4. Verifies migration success

Usage:
    python -m scripts.migrate_lens_database [--dry-run]
"""

import sqlite3
import shutil
from pathlib import Path
from datetime import datetime
import argparse


def create_backup(db_path: str) -> str:
    """Create a backup of the database before migration."""
    backup_path = db_path.replace('.db', f'_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.db')
    shutil.copy2(db_path, backup_path)
    print(f"✓ Created backup: {backup_path}")
    return backup_path


def get_lens_count(conn: sqlite3.Connection) -> int:
    """Get count of lenses in the lenses table."""
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM lenses")
    count = cursor.fetchone()[0]
    return count


def export_lenses_from_optimization_db(db_path: str) -> list:
    """Export all lenses from the optimization database."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Check if lenses table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='lenses'")
    if not cursor.fetchone():
        print("✗ No 'lenses' table found in optimization.db")
        conn.close()
        return []
    
    # Export all lenses
    cursor.execute("SELECT * FROM lenses")
    lenses = [dict(row) for row in cursor.fetchall()]
    
    conn.close()
    return lenses


def import_lenses_to_lens_db(db_path: str, lenses: list) -> bool:
    """Import lenses into the lens database."""
    # Create parent directory if needed
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create lenses table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS lenses (
            item_number TEXT PRIMARY KEY,
            lens_type TEXT NOT NULL DEFAULT 'Plano-Convex',
            diameter_mm REAL NOT NULL,
            focal_length_mm REAL NOT NULL,
            radius_r1_mm REAL NOT NULL,
            radius_r2_mm REAL,
            center_thickness_mm REAL NOT NULL,
            edge_thickness_mm REAL NOT NULL,
            back_focal_length_mm REAL NOT NULL,
            numerical_aperture REAL,
            substrate TEXT,
            coating TEXT,
            wavelength_range_nm TEXT,
            asphere_diameter_mm REAL,
            conic_constant REAL,
            vendor TEXT
        )
    """)
    
    # Create indexes
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_lenses_type 
        ON lenses(lens_type)
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_lenses_focal_length 
        ON lenses(focal_length_mm)
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_lenses_diameter 
        ON lenses(diameter_mm)
    """)
    
    # Insert lenses
    for lens in lenses:
        cursor.execute("""
            INSERT OR REPLACE INTO lenses 
                (item_number, lens_type, diameter_mm, focal_length_mm, 
                 radius_r1_mm, radius_r2_mm, center_thickness_mm, edge_thickness_mm, 
                 back_focal_length_mm, numerical_aperture, substrate, coating,
                 wavelength_range_nm, asphere_diameter_mm, conic_constant, vendor)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            lens['item_number'],
            lens['lens_type'],
            lens['diameter_mm'],
            lens['focal_length_mm'],
            lens['radius_r1_mm'],
            lens['radius_r2_mm'],
            lens['center_thickness_mm'],
            lens['edge_thickness_mm'],
            lens['back_focal_length_mm'],
            lens['numerical_aperture'],
            lens['substrate'],
            lens['coating'],
            lens['wavelength_range_nm'],
            lens['asphere_diameter_mm'],
            lens['conic_constant'],
            lens['vendor']
        ))
    
    conn.commit()
    
    # Verify count
    imported_count = get_lens_count(conn)
    conn.close()
    
    return imported_count == len(lenses)


def delete_lenses_table_from_optimization_db(db_path: str) -> bool:
    """Delete the lenses table from the optimization database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Drop lenses table
    cursor.execute("DROP TABLE IF EXISTS lenses")
    
    # Drop lenses indexes
    cursor.execute("DROP INDEX IF EXISTS idx_lenses_type")
    cursor.execute("DROP INDEX IF EXISTS idx_lenses_focal_length")
    cursor.execute("DROP INDEX IF EXISTS idx_lenses_diameter")
    
    conn.commit()
    
    # Verify table is gone
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='lenses'")
    table_exists = cursor.fetchone() is not None
    
    conn.close()
    
    return not table_exists


def verify_migration(optimization_db_path: str, lens_db_path: str, expected_count: int) -> bool:
    """Verify that migration was successful."""
    success = True
    
    # Check that lenses table is gone from optimization.db
    opt_conn = sqlite3.connect(optimization_db_path)
    cursor = opt_conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='lenses'")
    if cursor.fetchone():
        print("✗ Lenses table still exists in optimization.db")
        success = False
    else:
        print("✓ Lenses table removed from optimization.db")
    opt_conn.close()
    
    # Check that lenses exist in lenses.db
    if not Path(lens_db_path).exists():
        print(f"✗ Lens database not found: {lens_db_path}")
        return False
    
    lens_conn = sqlite3.connect(lens_db_path)
    actual_count = get_lens_count(lens_conn)
    lens_conn.close()
    
    if actual_count == expected_count:
        print(f"✓ All {expected_count} lenses successfully migrated to lenses.db")
    else:
        print(f"✗ Lens count mismatch: expected {expected_count}, found {actual_count}")
        success = False
    
    return success


def main():
    """Main migration function."""
    parser = argparse.ArgumentParser(description='Migrate lens data from optimization.db to lenses.db')
    parser.add_argument('--dry-run', action='store_true', 
                       help='Show what would be done without making changes')
    args = parser.parse_args()
    
    optimization_db_path = './results/optimization.db'
    lens_db_path = './data/lenses.db'
    
    print("=" * 70)
    print("LENS DATABASE MIGRATION")
    print("=" * 70)
    print(f"Source: {optimization_db_path}")
    print(f"Target: {lens_db_path}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'LIVE MIGRATION'}")
    print("=" * 70)
    print()
    
    # Check if source database exists
    if not Path(optimization_db_path).exists():
        print(f"✗ Source database not found: {optimization_db_path}")
        print("Migration aborted.")
        return 1
    
    # Export lenses from optimization.db
    print("Step 1: Exporting lenses from optimization.db...")
    lenses = export_lenses_from_optimization_db(optimization_db_path)
    
    if not lenses:
        print("✗ No lenses found to migrate")
        print("Migration aborted.")
        return 1
    
    print(f"✓ Found {len(lenses)} lenses to migrate")
    
    # Show lens statistics
    type_counts = {}
    vendor_counts = {}
    for lens in lenses:
        lens_type = lens.get('lens_type', 'Unknown')
        vendor = lens.get('vendor', 'Unknown')
        type_counts[lens_type] = type_counts.get(lens_type, 0) + 1
        vendor_counts[vendor] = vendor_counts.get(vendor, 0) + 1
    
    print("\n  Lens Types:")
    for lens_type, count in sorted(type_counts.items()):
        print(f"    - {lens_type}: {count}")
    
    print("\n  Vendors:")
    for vendor, count in sorted(vendor_counts.items()):
        print(f"    - {vendor}: {count}")
    print()
    
    if args.dry_run:
        print("DRY RUN: No changes made")
        print(f"Would migrate {len(lenses)} lenses from {optimization_db_path} to {lens_db_path}")
        print(f"Would delete lenses table from {optimization_db_path}")
        return 0
    
    # Create backup
    print("Step 2: Creating backup of optimization.db...")
    backup_path = create_backup(optimization_db_path)
    print()
    
    # Import lenses into lenses.db
    print("Step 3: Importing lenses into lenses.db...")
    if import_lenses_to_lens_db(lens_db_path, lenses):
        print(f"✓ Successfully imported {len(lenses)} lenses to {lens_db_path}")
    else:
        print("✗ Import verification failed - count mismatch")
        print("Migration aborted. Backup available at:", backup_path)
        return 1
    print()
    
    # Delete lenses table from optimization.db
    print("Step 4: Deleting lenses table from optimization.db...")
    if delete_lenses_table_from_optimization_db(optimization_db_path):
        print("✓ Successfully deleted lenses table from optimization.db")
    else:
        print("✗ Failed to delete lenses table")
        print("Migration aborted. Backup available at:", backup_path)
        return 1
    print()
    
    # Verify migration
    print("Step 5: Verifying migration...")
    if verify_migration(optimization_db_path, lens_db_path, len(lenses)):
        print()
        print("=" * 70)
        print("✓ MIGRATION COMPLETED SUCCESSFULLY")
        print("=" * 70)
        print(f"Migrated {len(lenses)} lenses from optimization.db to lenses.db")
        print(f"Backup created: {backup_path}")
        print()
        print("Next steps:")
        print("  1. Update your scripts to use LensDatabase for lens operations")
        print("  2. Test the new architecture")
        print("  3. If everything works, you can delete the backup file")
        return 0
    else:
        print()
        print("=" * 70)
        print("✗ MIGRATION VERIFICATION FAILED")
        print("=" * 70)
        print("Please restore from backup:", backup_path)
        return 1


if __name__ == '__main__':
    exit(main())
