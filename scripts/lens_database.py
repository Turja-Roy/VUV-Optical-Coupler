import sqlite3
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any


class LensDatabase:
    """SQLite database for storing lens catalog data."""
    
    def __init__(self, db_path: str = './data/lenses.db'):
        """Initialize database connection and create tables if needed."""
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()
    
    def _create_tables(self):
        """Create database tables if they don't exist."""
        cursor = self.conn.cursor()
        
        # Lenses table - stores lens specifications
        # Supports Plano-Convex, Bi-Convex, and Aspheric lens types
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
                vendor TEXT,
                notes TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create indexes for faster queries
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
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_lenses_vendor 
            ON lenses(vendor)
        """)
        
        self.conn.commit()
    
    def insert_lens(self, item_number: str, diameter_mm: float, 
                   focal_length_mm: float, radius_r1_mm: float,
                   center_thickness_mm: float, edge_thickness_mm: float,
                   back_focal_length_mm: float, 
                   lens_type: str = 'Plano-Convex',
                   radius_r2_mm: Optional[float] = None,
                   numerical_aperture: Optional[float] = None,
                   substrate: Optional[str] = None,
                   coating: Optional[str] = None,
                   wavelength_range_nm: Optional[str] = None,
                   asphere_diameter_mm: Optional[float] = None,
                   conic_constant: Optional[float] = None,
                   vendor: Optional[str] = None,
                   notes: Optional[str] = None) -> None:
        """Insert or update lens specification."""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO lenses 
                (item_number, lens_type, diameter_mm, focal_length_mm, 
                 radius_r1_mm, radius_r2_mm, center_thickness_mm, edge_thickness_mm, 
                 back_focal_length_mm, numerical_aperture, substrate, coating,
                 wavelength_range_nm, asphere_diameter_mm, conic_constant, vendor, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (item_number, lens_type, diameter_mm, focal_length_mm, 
              radius_r1_mm, radius_r2_mm, center_thickness_mm, edge_thickness_mm, 
              back_focal_length_mm, numerical_aperture, substrate, coating,
              wavelength_range_nm, asphere_diameter_mm, conic_constant, vendor, notes))
        
        self.conn.commit()
    
    def insert_lenses_from_dict(self, lenses: Dict[str, Dict], vendor: Optional[str] = None) -> None:
        """
        Bulk insert lenses from dictionary (as returned by data_io).
        
        Parameters:
        -----------
        lenses : dict
            Dictionary mapping item_number to lens data dict containing:
            - dia: diameter in mm
            - f_mm: focal length in mm
            - R_mm or R1_mm: front surface radius in mm
            - R2_mm: back surface radius (optional, for bi-convex)
            - tc_mm: center thickness in mm
            - te_mm: edge thickness in mm
            - BFL_mm: back focal length in mm
            - lens_type: 'Plano-Convex', 'Bi-Convex', or 'Aspheric' (optional, defaults to 'Plano-Convex')
        vendor : str, optional
            Vendor name to assign to all lenses
        """
        cursor = self.conn.cursor()
        
        data = []
        for item_num, lens_data in lenses.items():
            # Handle both old format (R_mm) and new format (R1_mm)
            r1_mm = lens_data.get('R1_mm', lens_data.get('R_mm'))
            r2_mm = lens_data.get('R2_mm', None)
            lens_type = lens_data.get('lens_type', 'Plano-Convex')
            
            data.append((
                item_num,
                lens_type,
                lens_data['dia'],
                lens_data['f_mm'],
                r1_mm,
                r2_mm,
                lens_data['tc_mm'],
                lens_data['te_mm'],
                lens_data['BFL_mm'],
                lens_data.get('numerical_aperture'),
                lens_data.get('substrate'),
                lens_data.get('coating'),
                lens_data.get('wavelength_range_nm'),
                lens_data.get('asphere_diameter_mm'),
                lens_data.get('conic_constant'),
                vendor,
                None
            ))
        
        cursor.executemany("""
            INSERT OR REPLACE INTO lenses 
                (item_number, lens_type, diameter_mm, focal_length_mm, 
                 radius_r1_mm, radius_r2_mm, center_thickness_mm, edge_thickness_mm, 
                 back_focal_length_mm, numerical_aperture, substrate, coating,
                 wavelength_range_nm, asphere_diameter_mm, conic_constant, vendor, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, data)
        
        self.conn.commit()
    
    def get_lens(self, item_number: str) -> Optional[Dict]:
        """Retrieve lens specification."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM lenses WHERE item_number = ?", (item_number,))
        row = cursor.fetchone()
        return dict(row) if row else None
    
    def get_all_lenses(self, lens_type: Optional[str] = None,
                      min_focal_length: Optional[float] = None,
                      max_focal_length: Optional[float] = None,
                      min_diameter: Optional[float] = None,
                      max_diameter: Optional[float] = None,
                      vendor: Optional[str] = None) -> List[Dict]:
        """
        Retrieve lens specifications with optional filtering.
        
        Parameters:
        -----------
        lens_type : str, optional
            Filter by lens type ('Plano-Convex', 'Bi-Convex', 'Aspheric')
        min_focal_length : float, optional
            Minimum focal length in mm
        max_focal_length : float, optional
            Maximum focal length in mm
        min_diameter : float, optional
            Minimum diameter in mm
        max_diameter : float, optional
            Maximum diameter in mm
        vendor : str, optional
            Filter by vendor name
            
        Returns:
        --------
        List of lens dictionaries
        """
        cursor = self.conn.cursor()
        
        query = "SELECT * FROM lenses WHERE 1=1"
        params: List[Any] = []
        
        if lens_type:
            query += " AND lens_type = ?"
            params.append(lens_type)
        
        if min_focal_length:
            query += " AND focal_length_mm >= ?"
            params.append(min_focal_length)
        
        if max_focal_length:
            query += " AND focal_length_mm <= ?"
            params.append(max_focal_length)
        
        if min_diameter:
            query += " AND diameter_mm >= ?"
            params.append(min_diameter)
        
        if max_diameter:
            query += " AND diameter_mm <= ?"
            params.append(max_diameter)
        
        if vendor:
            query += " AND vendor = ?"
            params.append(vendor)
        
        query += " ORDER BY item_number"
        
        cursor.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]
    
    def execute_custom_query(self, sql_query: str) -> List[Dict]:
        """
        Execute a custom SQL query on the lenses table.
        
        Parameters:
        -----------
        sql_query : str
            SQL query string. Must be a SELECT statement for safety.
            Example: "SELECT * FROM lenses WHERE focal_length_mm BETWEEN 15 AND 30"
            
        Returns:
        --------
        List of lens dictionaries matching the query
        
        Raises:
        -------
        ValueError
            If the query is not a SELECT statement or contains unsafe operations
        """
        # Validate query is read-only (SELECT only)
        query_normalized = sql_query.strip().upper()
        
        # Check if it starts with SELECT
        if not query_normalized.startswith('SELECT'):
            raise ValueError("Only SELECT queries are allowed for safety. Query must start with SELECT.")
        
        # Check for dangerous operations (basic SQL injection protection)
        dangerous_keywords = ['DROP', 'DELETE', 'INSERT', 'UPDATE', 'ALTER', 'CREATE', 
                             'TRUNCATE', 'REPLACE', 'EXEC', 'EXECUTE', '--', ';--']
        
        for keyword in dangerous_keywords:
            if keyword in query_normalized:
                raise ValueError(f"Query contains unsafe operation '{keyword}'. Only SELECT queries are allowed.")
        
        # Check that query references the lenses table
        if 'LENSES' not in query_normalized and 'FROM' in query_normalized:
            raise ValueError("Query must reference the 'lenses' table.")
        
        # Execute the query
        try:
            cursor = self.conn.cursor()
            cursor.execute(sql_query)
            results = [dict(row) for row in cursor.fetchall()]
            
            # Validate that results have the expected lens fields
            if results:
                required_fields = ['item_number', 'focal_length_mm', 'diameter_mm']
                for field in required_fields:
                    if field not in results[0]:
                        raise ValueError(f"Query results missing required field: {field}. "
                                       f"Use 'SELECT * FROM lenses WHERE ...' to ensure all fields are included.")
            
            return results
        except Exception as e:
            if isinstance(e, ValueError):
                raise
            raise ValueError(f"Error executing SQL query: {str(e)}")
    
    def get_lens_count(self) -> int:
        """Get total number of lenses in database."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) as count FROM lenses")
        return cursor.fetchone()['count']
    
    def get_statistics(self) -> Dict:
        """Get summary statistics about lenses in the database."""
        cursor = self.conn.cursor()
        
        # Overall statistics
        cursor.execute("""
            SELECT 
                COUNT(*) as total_lenses,
                COUNT(DISTINCT vendor) as total_vendors,
                COUNT(DISTINCT lens_type) as total_types,
                AVG(focal_length_mm) as avg_focal_length,
                MIN(focal_length_mm) as min_focal_length,
                MAX(focal_length_mm) as max_focal_length,
                AVG(diameter_mm) as avg_diameter,
                MIN(diameter_mm) as min_diameter,
                MAX(diameter_mm) as max_diameter
            FROM lenses
        """)
        overall = dict(cursor.fetchone())
        
        # By type
        cursor.execute("""
            SELECT lens_type, COUNT(*) as count
            FROM lenses
            GROUP BY lens_type
            ORDER BY count DESC
        """)
        by_type = {row['lens_type']: row['count'] for row in cursor.fetchall()}
        
        # By vendor
        cursor.execute("""
            SELECT vendor, COUNT(*) as count
            FROM lenses
            WHERE vendor IS NOT NULL
            GROUP BY vendor
            ORDER BY count DESC
        """)
        by_vendor = {row['vendor']: row['count'] for row in cursor.fetchall()}
        
        return {
            'overall': overall,
            'by_type': by_type,
            'by_vendor': by_vendor
        }
    
    def delete_lens(self, item_number: str) -> None:
        """Delete a lens from the database."""
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM lenses WHERE item_number = ?", (item_number,))
        self.conn.commit()
    
    def export_to_csv(self, output_path: str, lens_type: Optional[str] = None) -> None:
        """Export lenses to CSV file."""
        lenses = self.get_all_lenses(lens_type=lens_type)
        df = pd.DataFrame(lenses)
        
        # Drop database-specific columns
        cols_to_drop = ['created_at']
        df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
        
        df.to_csv(output_path, index=False)
    
    def close(self):
        """Close database connection."""
        self.conn.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
