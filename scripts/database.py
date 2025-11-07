import sqlite3
import json
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any


class OptimizationDatabase:
    """SQLite database for storing optimization results and metadata."""
    
    def __init__(self, db_path: str = './results/optimization.db'):
        """Initialize database connection and create tables if needed."""
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()
    
    def _create_tables(self):
        """Create database tables if they don't exist."""
        cursor = self.conn.cursor()
        
        # Runs table - stores metadata about each optimization run
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                method TEXT NOT NULL,
                medium TEXT NOT NULL,
                n_rays INTEGER NOT NULL,
                wavelength_nm REAL NOT NULL,
                pressure_atm REAL NOT NULL,
                temperature_k REAL NOT NULL,
                humidity_fraction REAL NOT NULL,
                config_json TEXT,
                notes TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Results table - stores individual optimization results
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                lens1 TEXT NOT NULL,
                lens2 TEXT NOT NULL,
                method TEXT NOT NULL,
                orientation TEXT,
                z_l1 REAL NOT NULL,
                z_l2 REAL NOT NULL,
                z_fiber REAL NOT NULL,
                total_len_mm REAL NOT NULL,
                coupling REAL NOT NULL,
                f1_mm REAL NOT NULL,
                f2_mm REAL NOT NULL,
                computation_time_seconds REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
            )
        """)
        
        # Create indexes for faster queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_results_run_id 
            ON results(run_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_results_coupling 
            ON results(coupling DESC)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_results_lens_pair 
            ON results(lens1, lens2)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_runs_timestamp 
            ON runs(timestamp DESC)
        """)
        
        self.conn.commit()
    
    def insert_run(self, run_id: str, method: str, medium: str, 
                   n_rays: int, wavelength_nm: float, pressure_atm: float,
                   temperature_k: float, humidity_fraction: float,
                   config: Optional[Dict] = None, notes: Optional[str] = None) -> None:
        """Insert a new optimization run record."""
        cursor = self.conn.cursor()
        timestamp = datetime.now().isoformat()
        config_json = json.dumps(config) if config else None
        
        cursor.execute("""
            INSERT INTO runs (run_id, timestamp, method, medium, n_rays, 
                            wavelength_nm, pressure_atm, temperature_k, 
                            humidity_fraction, config_json, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (run_id, timestamp, method, medium, n_rays, wavelength_nm,
              pressure_atm, temperature_k, humidity_fraction, config_json, notes))
        
        self.conn.commit()
    
    def insert_result(self, run_id: str, result: Dict[str, Any], 
                     computation_time: Optional[float] = None) -> None:
        """Insert a single optimization result."""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            INSERT INTO results (run_id, lens1, lens2, method, orientation,
                               z_l1, z_l2, z_fiber, total_len_mm, coupling,
                               f1_mm, f2_mm, computation_time_seconds)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            run_id,
            result.get('lens1'),
            result.get('lens2'),
            result.get('method', 'unknown'),
            result.get('orientation'),
            result['z_l1'],
            result['z_l2'],
            result['z_fiber'],
            result['total_len_mm'],
            result['coupling'],
            result['f1_mm'],
            result['f2_mm'],
            computation_time
        ))
        
        self.conn.commit()
    
    def insert_results_batch(self, run_id: str, results: List[Dict[str, Any]],
                            computation_times: Optional[List[float]] = None) -> None:
        """Insert multiple optimization results efficiently."""
        cursor = self.conn.cursor()
        
        if computation_times is None:
            computation_times_to_use = [None] * len(results)
        else:
            computation_times_to_use = computation_times
        
        data = [
            (
                run_id,
                result.get('lens1'),
                result.get('lens2'),
                result.get('method', 'unknown'),
                result.get('orientation'),
                result['z_l1'],
                result['z_l2'],
                result['z_fiber'],
                result['total_len_mm'],
                result['coupling'],
                result['f1_mm'],
                result['f2_mm'],
                comp_time
            )
            for result, comp_time in zip(results, computation_times_to_use)
        ]
        
        cursor.executemany("""
            INSERT INTO results (run_id, lens1, lens2, method, orientation,
                               z_l1, z_l2, z_fiber, total_len_mm, coupling,
                               f1_mm, f2_mm, computation_time_seconds)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, data)
        
        self.conn.commit()
    
    def get_run(self, run_id: str) -> Optional[Dict]:
        """Retrieve run metadata by run_id."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,))
        row = cursor.fetchone()
        
        if row:
            result = dict(row)
            if result['config_json']:
                result['config'] = json.loads(result['config_json'])
            return result
        return None
    
    def get_results(self, run_id: str, min_coupling: Optional[float] = None,
                   max_length: Optional[float] = None, 
                   limit: Optional[int] = None) -> List[Dict]:
        """Retrieve results for a run with optional filtering."""
        cursor = self.conn.cursor()
        
        query = "SELECT * FROM results WHERE run_id = ?"
        params: List[Any] = [run_id]
        
        if min_coupling is not None:
            query += " AND coupling >= ?"
            params.append(min_coupling)
        
        if max_length is not None:
            query += " AND total_len_mm <= ?"
            params.append(max_length)
        
        query += " ORDER BY coupling DESC, total_len_mm ASC"
        
        if limit:
            query += " LIMIT ?"
            params.append(limit)
        
        cursor.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]
    
    def get_best_results(self, limit: int = 10, 
                        medium: Optional[str] = None,
                        min_coupling: Optional[float] = None) -> List[Dict]:
        """Get best results across all runs."""
        cursor = self.conn.cursor()
        
        query = """
            SELECT r.*, ru.medium, ru.method as run_method, ru.timestamp
            FROM results r
            JOIN runs ru ON r.run_id = ru.run_id
            WHERE 1=1
        """
        params: List[Any] = []
        
        if medium:
            query += " AND ru.medium = ?"
            params.append(medium)
        
        if min_coupling:
            query += " AND r.coupling >= ?"
            params.append(min_coupling)
        
        query += " ORDER BY r.coupling DESC, r.total_len_mm ASC LIMIT ?"
        params.append(limit)
        
        cursor.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]
    
    def get_lens_pair_history(self, lens1: str, lens2: str) -> List[Dict]:
        """Get all results for a specific lens pair across all runs."""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            SELECT r.*, ru.medium, ru.timestamp, ru.method as run_method
            FROM results r
            JOIN runs ru ON r.run_id = ru.run_id
            WHERE (r.lens1 = ? AND r.lens2 = ?) OR (r.lens1 = ? AND r.lens2 = ?)
            ORDER BY r.coupling DESC
        """, (lens1, lens2, lens2, lens1))
        
        return [dict(row) for row in cursor.fetchall()]
    
    def get_all_runs(self, method: Optional[str] = None, 
                    medium: Optional[str] = None) -> List[Dict]:
        """Get all runs with optional filtering."""
        cursor = self.conn.cursor()
        
        query = "SELECT * FROM runs WHERE 1=1"
        params: List[Any] = []
        
        if method:
            query += " AND method = ?"
            params.append(method)
        
        if medium:
            query += " AND medium = ?"
            params.append(medium)
        
        query += " ORDER BY timestamp DESC"
        
        cursor.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]
    
    def export_to_csv(self, run_id: str, output_path: str) -> None:
        """Export results for a run to CSV file."""
        results = self.get_results(run_id)
        df = pd.DataFrame(results)
        
        # Drop database-specific columns
        cols_to_drop = ['id', 'created_at']
        df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
        
        df.to_csv(output_path, index=False)
    
    def export_best_to_csv(self, output_path: str, limit: int = 100,
                          medium: Optional[str] = None) -> None:
        """Export best results across all runs to CSV."""
        results = self.get_best_results(limit=limit, medium=medium)
        df = pd.DataFrame(results)
        
        # Drop database-specific columns
        cols_to_drop = ['id', 'created_at']
        df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
        
        df.to_csv(output_path, index=False)
    
    def get_statistics(self, run_id: Optional[str] = None) -> Dict:
        """Get summary statistics for results."""
        cursor = self.conn.cursor()
        
        if run_id:
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_results,
                    AVG(coupling) as avg_coupling,
                    MAX(coupling) as max_coupling,
                    MIN(coupling) as min_coupling,
                    AVG(total_len_mm) as avg_length,
                    MIN(total_len_mm) as min_length,
                    MAX(total_len_mm) as max_length,
                    AVG(computation_time_seconds) as avg_time
                FROM results
                WHERE run_id = ?
            """, (run_id,))
        else:
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_results,
                    AVG(coupling) as avg_coupling,
                    MAX(coupling) as max_coupling,
                    MIN(coupling) as min_coupling,
                    AVG(total_len_mm) as avg_length,
                    MIN(total_len_mm) as min_length,
                    MAX(total_len_mm) as max_length,
                    AVG(computation_time_seconds) as avg_time
                FROM results
            """)
        
        row = cursor.fetchone()
        return dict(row) if row else {}
    
    def delete_run(self, run_id: str) -> None:
        """Delete a run and all associated results."""
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))
        self.conn.commit()
    
    def close(self):
        """Close database connection."""
        self.conn.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
