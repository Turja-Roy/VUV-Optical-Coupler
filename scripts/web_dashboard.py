"""
Web dashboard for viewing and comparing lens optimization results.

Provides a Flask-based web interface for:
- Viewing optimization results from database or CSV files
- Filtering by coupling threshold, medium, lens type
- Comparing lens pairs side-by-side
- Interactive plots of coupling vs wavelength
- Exporting filtered results

Usage:
    python raytrace.py dashboard [--port 5000] [--db path/to/db]
"""

import io
import base64
from pathlib import Path
from flask import Flask, render_template_string, request, jsonify, send_file
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Optional, Dict, List

from scripts import consts as C


class DashboardServer:
    """Flask server for lens optimization results dashboard."""
    
    def __init__(self, db_path: Optional[str] = None, results_dir: str = './results'):
        """
        Initialize dashboard server.
        
        Parameters
        ----------
        db_path : str, optional
            Path to SQLite database. If None, will look for database at default location.
        results_dir : str
            Directory containing CSV result files (used if database not available)
        """
        self.app = Flask(__name__)
        self.db_path = db_path or C.DATABASE_PATH
        self.results_dir = Path(results_dir)
        self.use_database = Path(self.db_path).exists()
        
        if self.use_database:
            from scripts.database import OptimizationDatabase
            self.db = OptimizationDatabase(self.db_path)
            print(f"Dashboard connected to database: {self.db_path}")
        else:
            self.db = None
            print(f"Database not found, will use CSV files from: {self.results_dir}")
        
        # Initialize lens database
        self.lens_db_path = C.LENS_DATABASE_PATH
        self.use_lens_database = Path(self.lens_db_path).exists()
        if self.use_lens_database:
            from scripts.lens_database import LensDatabase
            self.lens_db = LensDatabase(self.lens_db_path)
            print(f"Dashboard connected to lens database: {self.lens_db_path}")
        else:
            self.lens_db = None
            print(f"Lens database not found at: {self.lens_db_path}")
        
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup Flask routes."""
        
        @self.app.route('/')
        def index():
            """Main dashboard page."""
            return render_template_string(DASHBOARD_HTML)
        
        @self.app.route('/api/results')
        def get_results():
            """
            Get optimization results with optional filtering.
            
            Query parameters:
            - min_coupling: Minimum coupling threshold (0-1)
            - max_coupling: Maximum coupling threshold (0-1)
            - medium: Filter by medium (air, argon, helium)
            - lens1: Filter by first lens
            - lens2: Filter by second lens
            - method: Filter by optimization method
            - limit: Maximum number of results (default: 1000)
            - sort_by: Column to sort by (default: coupling)
            - sort_desc: Sort descending (default: true)
            """
            min_coupling = request.args.get('min_coupling', type=float, default=0.0)
            max_coupling = request.args.get('max_coupling', type=float, default=1.0)
            medium = request.args.get('medium', type=str, default=None)
            lens1 = request.args.get('lens1', type=str, default=None)
            lens2 = request.args.get('lens2', type=str, default=None)
            method = request.args.get('method', type=str, default=None)
            limit = request.args.get('limit', type=int, default=1000)
            sort_by = request.args.get('sort_by', type=str, default='coupling')
            sort_desc = request.args.get('sort_desc', type=str, default='true').lower() == 'true'
            
            if self.use_database:
                df = self._query_database(min_coupling, max_coupling, medium, 
                                         lens1, lens2, method, limit, sort_by, sort_desc)
            else:
                df = self._query_csv_files(min_coupling, max_coupling, medium,
                                          lens1, lens2, method, limit, sort_by, sort_desc)
            
            if df is None or df.empty:
                return jsonify({'results': [], 'count': 0})
            
            # Round numeric columns for display
            numeric_cols = ['coupling', 'z_l1', 'z_l2', 'z_fiber', 'total_len_mm', 'f1_mm', 'f2_mm']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = df[col].round(4)
            
            return jsonify({
                'results': df.to_dict(orient='records'),
                'count': len(df)
            })
        
        @self.app.route('/api/stats')
        def get_stats():
            """Get summary statistics."""
            if self.use_database:
                stats = self._get_database_stats()
            else:
                stats = self._get_csv_stats()
            
            return jsonify(stats)
        
        @self.app.route('/api/lens_pairs')
        def get_lens_pairs():
            """Get list of unique lens pairs."""
            if self.use_database:
                pairs = self._get_database_lens_pairs()
            else:
                pairs = self._get_csv_lens_pairs()
            
            return jsonify({'lens_pairs': pairs})
        
        @self.app.route('/api/plot/coupling_histogram')
        def plot_coupling_histogram():
            """Generate coupling distribution histogram."""
            if self.use_database:
                df = self._query_database(0, 1, None, None, None, None, 10000, 'coupling', True)
            else:
                df = self._query_csv_files(0, 1, None, None, None, None, 10000, 'coupling', True)
            
            if df is None or df.empty:
                return jsonify({'error': 'No data available'}), 404
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(df['coupling'], bins=50, edgecolor='black', alpha=0.7)
            ax.set_xlabel('Coupling Efficiency')
            ax.set_ylabel('Count')
            ax.set_title('Coupling Efficiency Distribution')
            ax.grid(True, alpha=0.3)
            
            img_bytes = self._fig_to_bytes(fig)
            plt.close(fig)
            
            return send_file(img_bytes, mimetype='image/png')
        
        @self.app.route('/api/plot/compare/<lens1>/<lens2>')
        def plot_lens_comparison(lens1, lens2):
            """Compare different optimization methods for a lens pair."""
            if self.use_database:
                df = self._query_database(0, 1, None, lens1, lens2, None, 1000, 'coupling', True)
            else:
                df = self._query_csv_files(0, 1, None, lens1, lens2, None, 1000, 'coupling', True)
            
            if df is None or df.empty:
                return jsonify({'error': 'No data for this lens pair'}), 404
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            # Coupling by method
            if 'method' in df.columns:
                method_groups = df.groupby('method')['coupling'].agg(['mean', 'max', 'count'])
                method_groups = method_groups.sort_values('mean', ascending=False)
                
                ax1.bar(range(len(method_groups)), method_groups['mean'], alpha=0.7)
                ax1.set_xticks(range(len(method_groups)))
                ax1.set_xticklabels(method_groups.index, rotation=45, ha='right')
                ax1.set_ylabel('Mean Coupling')
                ax1.set_title(f'{lens1} + {lens2}: Coupling by Method')
                ax1.grid(True, alpha=0.3)
            
            # Length vs coupling scatter
            ax2.scatter(df['total_len_mm'], df['coupling'], alpha=0.5)
            ax2.set_xlabel('Total Length (mm)')
            ax2.set_ylabel('Coupling Efficiency')
            ax2.set_title(f'{lens1} + {lens2}: Length vs Coupling')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            img_bytes = self._fig_to_bytes(fig)
            plt.close(fig)
            
            return send_file(img_bytes, mimetype='image/png')
        
        @self.app.route('/api/export')
        def export_csv():
            """Export filtered results as CSV."""
            min_coupling = request.args.get('min_coupling', type=float, default=0.0)
            medium = request.args.get('medium', type=str, default=None)
            
            if self.use_database:
                df = self._query_database(min_coupling, 1, medium, None, None, None, 10000, 'coupling', True)
            else:
                df = self._query_csv_files(min_coupling, 1, medium, None, None, None, 10000, 'coupling', True)
            
            if df is None or df.empty:
                return jsonify({'error': 'No data to export'}), 404
            
            output = io.StringIO()
            df.to_csv(output, index=False)
            output.seek(0)
            
            return send_file(
                io.BytesIO(output.getvalue().encode()),
                mimetype='text/csv',
                as_attachment=True,
                download_name='lens_optimization_results.csv'
            )
        
        @self.app.route('/api/lenses')
        def get_lenses():
            """
            Get lenses from catalog with optional filtering.
            
            Query parameters:
            - lens_type: Filter by type (Plano-Convex, Bi-Convex, etc.)
            - min_diameter: Minimum diameter in mm
            - max_diameter: Maximum diameter in mm
            - min_focal_length: Minimum focal length in mm
            - max_focal_length: Maximum focal length in mm
            - vendor: Filter by vendor (ThorLabs, Edmund)
            - search: Search by item number
            - limit: Maximum number of results (default: 100)
            - sort_by: Column to sort by (default: focal_length_mm)
            - sort_desc: Sort descending (default: false)
            """
            if not self.use_lens_database:
                return jsonify({'error': 'Lens database not available'}), 404
            
            lens_type = request.args.get('lens_type', type=str, default=None)
            min_diameter = request.args.get('min_diameter', type=float, default=None)
            max_diameter = request.args.get('max_diameter', type=float, default=None)
            min_focal = request.args.get('min_focal_length', type=float, default=None)
            max_focal = request.args.get('max_focal_length', type=float, default=None)
            vendor = request.args.get('vendor', type=str, default=None)
            search = request.args.get('search', type=str, default=None)
            limit = request.args.get('limit', type=int, default=100)
            sort_by = request.args.get('sort_by', type=str, default='focal_length_mm')
            sort_desc = request.args.get('sort_desc', type=str, default='false').lower() == 'true'
            
            lenses = self.lens_db.get_all_lenses(
                lens_type=lens_type,
                min_diameter=min_diameter,
                max_diameter=max_diameter,
                min_focal_length=min_focal,
                max_focal_length=max_focal,
                vendor=vendor
            )
            
            # Apply search filter if provided
            if search:
                search_lower = search.lower()
                lenses = [l for l in lenses if search_lower in l['item_number'].lower()]
            
            # Sort results
            if sort_by in ['item_number', 'lens_type', 'focal_length_mm', 'diameter_mm', 'vendor']:
                lenses.sort(key=lambda x: x.get(sort_by, ''), reverse=sort_desc)
            
            # Apply limit
            limited_lenses = lenses[:limit]
            
            return jsonify({
                'lenses': limited_lenses,
                'count': len(limited_lenses),
                'total': len(lenses)
            })
        
        @self.app.route('/api/lenses/<item_number>')
        def get_lens_detail(item_number):
            """Get detailed information about a specific lens."""
            if not self.use_lens_database:
                return jsonify({'error': 'Lens database not available'}), 404
            
            lens = self.lens_db.get_lens(item_number)
            if not lens:
                return jsonify({'error': 'Lens not found'}), 404
            
            return jsonify(lens)
        
        @self.app.route('/api/lenses/<item_number>/results')
        def get_lens_results(item_number):
            """Get optimization results for lenses containing this item number."""
            if not self.use_database:
                return jsonify({'error': 'Optimization database not available'}), 404
            
            # Query results where lens appears as lens1 or lens2
            query = """
                SELECT r.*, runs.medium, runs.method as run_method
                FROM results r
                LEFT JOIN runs ON r.run_id = runs.run_id
                WHERE r.lens1 = ? OR r.lens2 = ?
                ORDER BY r.coupling DESC
                LIMIT 100
            """
            
            try:
                df = pd.read_sql_query(query, self.db.conn, params=[item_number, item_number])
                
                if df.empty:
                    return jsonify({'results': [], 'count': 0})
                
                # Round numeric columns
                numeric_cols = ['coupling', 'z_l1', 'z_l2', 'z_fiber', 'total_len_mm', 'f1_mm', 'f2_mm']
                for col in numeric_cols:
                    if col in df.columns:
                        df[col] = df[col].round(4)
                
                return jsonify({
                    'results': df.to_dict(orient='records'),
                    'count': len(df),
                    'best_coupling': float(df['coupling'].max()) if len(df) > 0 else 0,
                    'avg_coupling': float(df['coupling'].mean()) if len(df) > 0 else 0
                })
            except Exception as e:
                print(f"Error querying lens results: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/lenses/stats')
        def get_lens_stats():
            """Get lens catalog statistics."""
            if not self.use_lens_database:
                return jsonify({'error': 'Lens database not available'}), 404
            
            stats = self.lens_db.get_statistics()
            return jsonify(stats)
    
    def _query_database(self, min_coupling, max_coupling, medium, lens1, lens2, 
                       method, limit, sort_by, sort_desc):
        """Query results from database."""
        if not self.db:
            return None
        
        query = """
            SELECT r.*, runs.medium, runs.method as run_method
            FROM results r
            LEFT JOIN runs ON r.run_id = runs.run_id
            WHERE r.coupling >= ? AND r.coupling <= ?
        """
        params = [min_coupling, max_coupling]
        
        if medium:
            query += " AND runs.medium = ?"
            params.append(medium)
        if lens1:
            query += " AND r.lens1 = ?"
            params.append(lens1)
        if lens2:
            query += " AND r.lens2 = ?"
            params.append(lens2)
        if method:
            query += " AND r.method = ?"
            params.append(method)
        
        order = 'DESC' if sort_desc else 'ASC'
        query += f" ORDER BY r.{sort_by} {order} LIMIT ?"
        params.append(limit)
        
        try:
            df = pd.read_sql_query(query, self.db.conn, params=params)
            return df
        except Exception as e:
            print(f"Database query error: {e}")
            return None
    
    def _query_csv_files(self, min_coupling, max_coupling, medium, lens1, lens2,
                        method, limit, sort_by, sort_desc):
        """Query results from CSV files."""
        all_dfs = []
        
        # Find all CSV files in results directory
        for csv_file in self.results_dir.rglob('*.csv'):
            # Skip tolerance and wavelength analysis CSVs
            if 'tolerance' in csv_file.name or 'wavelength' in csv_file.name:
                continue
            
            try:
                df = pd.read_csv(csv_file)
                
                # Try to infer medium from directory name
                if 'medium' not in df.columns:
                    parent_name = csv_file.parent.name.lower()
                    if 'argon' in parent_name:
                        df['medium'] = 'argon'
                    elif 'helium' in parent_name:
                        df['medium'] = 'helium'
                    else:
                        df['medium'] = 'air'
                
                all_dfs.append(df)
            except Exception as e:
                print(f"Error reading {csv_file}: {e}")
                continue
        
        if not all_dfs:
            return None
        
        df = pd.concat(all_dfs, ignore_index=True)
        
        # Apply filters
        df = df[df['coupling'] >= min_coupling]
        df = df[df['coupling'] <= max_coupling]
        
        if medium and 'medium' in df.columns:
            df = df[df['medium'] == medium]
        if lens1:
            df = df[df['lens1'] == lens1]
        if lens2:
            df = df[df['lens2'] == lens2]
        if method and 'method' in df.columns:
            df = df[df['method'] == method]
        
        # Sort and limit
        if sort_by in df.columns:
            df = df.sort_values(sort_by, ascending=not sort_desc)
        
        return df.head(limit)
    
    def _get_database_stats(self):
        """Get statistics from database."""
        cursor = self.db.conn.cursor()
        
        total_results = cursor.execute("SELECT COUNT(*) FROM results").fetchone()[0]
        total_runs = cursor.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
        
        best_coupling = cursor.execute(
            "SELECT lens1, lens2, coupling, total_len_mm FROM results ORDER BY coupling DESC LIMIT 1"
        ).fetchone()
        
        avg_coupling = cursor.execute("SELECT AVG(coupling) FROM results").fetchone()[0]
        
        return {
            'total_results': total_results,
            'total_runs': total_runs,
            'best_coupling': dict(best_coupling) if best_coupling else None,
            'avg_coupling': round(avg_coupling, 4) if avg_coupling else 0
        }
    
    def _get_csv_stats(self):
        """Get statistics from CSV files."""
        df = self._query_csv_files(0, 1, None, None, None, None, 100000, 'coupling', True)
        
        if df is None or df.empty:
            return {'total_results': 0, 'total_runs': 0, 'best_coupling': None, 'avg_coupling': 0}
        
        best = df.loc[df['coupling'].idxmax()]
        
        return {
            'total_results': len(df),
            'total_runs': len(set(df.get('run_id', ['unknown']))),
            'best_coupling': {
                'lens1': best['lens1'],
                'lens2': best['lens2'],
                'coupling': float(best['coupling']),
                'total_len_mm': float(best['total_len_mm'])
            },
            'avg_coupling': float(df['coupling'].mean())
        }
    
    def _get_database_lens_pairs(self):
        """Get unique lens pairs from database."""
        cursor = self.db.conn.cursor()
        pairs = cursor.execute(
            "SELECT DISTINCT lens1, lens2 FROM results ORDER BY lens1, lens2"
        ).fetchall()
        return [f"{p[0]}+{p[1]}" for p in pairs]
    
    def _get_csv_lens_pairs(self):
        """Get unique lens pairs from CSV files."""
        df = self._query_csv_files(0, 1, None, None, None, None, 100000, 'coupling', True)
        
        if df is None or df.empty:
            return []
        
        pairs = df[['lens1', 'lens2']].drop_duplicates()
        return [f"{row['lens1']}+{row['lens2']}" for _, row in pairs.iterrows()]
    
    def _fig_to_bytes(self, fig):
        """Convert matplotlib figure to bytes for Flask response."""
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        return buf
    
    def run(self, host='127.0.0.1', port=5000, debug=False):
        """Start the Flask server."""
        print(f"\nStarting dashboard server at http://{host}:{port}")
        print(f"Press Ctrl+C to stop\n")
        self.app.run(host=host, port=port, debug=debug)


# HTML template for dashboard
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Lens Optimization Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        :root {
            --bg-primary: #f5f7fa;
            --bg-secondary: #ffffff;
            --bg-tertiary: #f8f9fa;
            --text-primary: #333333;
            --text-secondary: #666666;
            --border-color: #e1e4e8;
            --accent-color: #2196F3;
            --accent-hover: #1976D2;
            --success-color: #4caf50;
            --warning-color: #ff9800;
            --danger-color: #f44336;
            --shadow-sm: 0 1px 3px rgba(0,0,0,0.08);
            --shadow-md: 0 4px 6px rgba(0,0,0,0.1);
            --shadow-lg: 0 10px 20px rgba(0,0,0,0.12);
        }
        
        body.dark-mode {
            --bg-primary: #1a1a1a;
            --bg-secondary: #2d2d2d;
            --bg-tertiary: #3a3a3a;
            --text-primary: #e0e0e0;
            --text-secondary: #a0a0a0;
            --border-color: #404040;
            --shadow-sm: 0 1px 3px rgba(0,0,0,0.3);
            --shadow-md: 0 4px 6px rgba(0,0,0,0.4);
            --shadow-lg: 0 10px 20px rgba(0,0,0,0.5);
        }
        
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body { 
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            padding: 20px;
            transition: background 0.3s, color 0.3s;
        }
        
        .container { max-width: 1600px; margin: 0 auto; }
        
        header { 
            background: var(--bg-secondary); 
            padding: 25px 30px; 
            border-radius: 12px; 
            margin-bottom: 25px;
            box-shadow: var(--shadow-md);
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 15px;
        }
        
        .header-content h1 { 
            color: var(--text-primary); 
            margin-bottom: 8px;
            font-size: 28px;
            font-weight: 700;
        }
        
        .header-content p {
            color: var(--text-secondary);
            font-size: 14px;
        }
        
        .header-actions {
            display: flex;
            gap: 10px;
            align-items: center;
        }
        
        .theme-toggle {
            background: var(--bg-tertiary);
            border: 1px solid var(--border-color);
            padding: 8px 16px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 14px;
            color: var(--text-primary);
            transition: all 0.2s;
        }
        
        .theme-toggle:hover {
            background: var(--border-color);
        }
        
        .stats { 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); 
            gap: 20px; 
            margin-bottom: 25px;
        }
        
        .stat-card { 
            background: var(--bg-secondary); 
            padding: 20px; 
            border-radius: 12px;
            box-shadow: var(--shadow-md);
            border-left: 4px solid var(--accent-color);
            transition: transform 0.2s, box-shadow 0.2s;
        }
        
        .stat-card:hover {
            transform: translateY(-2px);
            box-shadow: var(--shadow-lg);
        }
        
        .stat-value { 
            font-size: 32px; 
            font-weight: bold; 
            color: var(--accent-color);
            margin-bottom: 8px;
        }
        
        .stat-label { 
            color: var(--text-secondary); 
            font-size: 13px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            font-weight: 500;
        }
        
        .filters { 
            background: var(--bg-secondary); 
            padding: 25px; 
            border-radius: 12px; 
            margin-bottom: 25px;
            box-shadow: var(--shadow-md);
        }
        
        .filters h2 {
            margin-bottom: 20px;
            font-size: 18px;
            font-weight: 600;
        }
        
        .filter-row { 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); 
            gap: 15px; 
            margin-bottom: 20px;
        }
        
        .filter-group { display: flex; flex-direction: column; }
        
        label { 
            font-weight: 500; 
            margin-bottom: 6px; 
            color: var(--text-primary);
            font-size: 13px;
        }
        
        input, select { 
            padding: 10px 12px; 
            border: 1px solid var(--border-color); 
            border-radius: 6px;
            font-size: 14px;
            background: var(--bg-primary);
            color: var(--text-primary);
            transition: border-color 0.2s;
        }
        
        input:focus, select:focus {
            outline: none;
            border-color: var(--accent-color);
        }
        
        button { 
            padding: 10px 20px; 
            background: var(--accent-color); 
            color: white; 
            border: none; 
            border-radius: 6px; 
            cursor: pointer;
            font-size: 14px;
            font-weight: 500;
            transition: background 0.2s, transform 0.1s;
        }
        
        button:hover { 
            background: var(--accent-hover);
            transform: translateY(-1px);
        }
        
        button:active {
            transform: translateY(0);
        }
        
        button.secondary {
            background: var(--bg-tertiary);
            color: var(--text-primary);
            border: 1px solid var(--border-color);
        }
        
        button.secondary:hover {
            background: var(--border-color);
        }
        
        .action-buttons { 
            display: flex; 
            gap: 10px; 
            flex-wrap: wrap;
        }
        
        .tabs {
            display: flex;
            gap: 10px;
            margin-bottom: 25px;
            border-bottom: 2px solid var(--border-color);
            padding-bottom: 0;
        }
        
        .tab {
            padding: 12px 24px;
            background: none;
            border: none;
            border-bottom: 3px solid transparent;
            cursor: pointer;
            font-size: 14px;
            font-weight: 500;
            color: var(--text-secondary);
            transition: all 0.2s;
            margin-bottom: -2px;
        }
        
        .tab:hover {
            color: var(--text-primary);
            background: var(--bg-tertiary);
        }
        
        .tab.active {
            color: var(--accent-color);
            border-bottom-color: var(--accent-color);
        }
        
        .tab-content {
            display: none;
        }
        
        .tab-content.active {
            display: block;
        }
        
        .results-section { 
            background: var(--bg-secondary); 
            padding: 25px; 
            border-radius: 12px;
            box-shadow: var(--shadow-md);
            margin-bottom: 25px;
        }
        
        .results-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            flex-wrap: wrap;
            gap: 15px;
        }
        
        .results-header h2 {
            font-size: 18px;
            font-weight: 600;
        }
        
        .pagination {
            display: flex;
            gap: 5px;
            align-items: center;
        }
        
        .pagination button {
            padding: 6px 12px;
            font-size: 13px;
        }
        
        .pagination span {
            color: var(--text-secondary);
            font-size: 13px;
            margin: 0 10px;
        }
        
        .table-container {
            overflow-x: auto;
            border-radius: 8px;
            border: 1px solid var(--border-color);
        }
        
        table { 
            width: 100%; 
            border-collapse: collapse;
        }
        
        th, td { 
            padding: 14px 16px; 
            text-align: left; 
            border-bottom: 1px solid var(--border-color);
        }
        
        th { 
            background: var(--bg-tertiary); 
            font-weight: 600; 
            color: var(--text-primary);
            position: sticky;
            top: 0;
            cursor: pointer;
            user-select: none;
            white-space: nowrap;
        }
        
        th:hover {
            background: var(--border-color);
        }
        
        th .sort-icon {
            font-size: 10px;
            margin-left: 5px;
            opacity: 0.5;
        }
        
        th.sorted .sort-icon {
            opacity: 1;
        }
        
        tr:hover { 
            background: var(--bg-tertiary);
            cursor: pointer;
        }
        
        tr:last-child td {
            border-bottom: none;
        }
        
        .loading { 
            text-align: center; 
            padding: 60px 20px; 
            color: var(--text-secondary);
        }
        
        .spinner {
            border: 3px solid var(--border-color);
            border-top: 3px solid var(--accent-color);
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto 15px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .plot-section { 
            background: var(--bg-secondary); 
            padding: 25px; 
            border-radius: 12px; 
            margin-bottom: 25px;
            box-shadow: var(--shadow-md);
        }
        
        .plot-section h2 {
            font-size: 18px;
            font-weight: 600;
            margin-bottom: 20px;
        }
        
        .plot-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
        }
        
        .plot-container { 
            position: relative;
            background: var(--bg-primary);
            padding: 15px;
            border-radius: 8px;
            border: 1px solid var(--border-color);
        }
        
        canvas {
            max-width: 100%;
            height: auto !important;
        }
        
        .modal {
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.6);
            backdrop-filter: blur(4px);
        }
        
        .modal.active {
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        .modal-content {
            background: var(--bg-secondary);
            padding: 30px;
            border-radius: 12px;
            max-width: 800px;
            width: 90%;
            max-height: 90vh;
            overflow-y: auto;
            box-shadow: var(--shadow-lg);
            position: relative;
        }
        
        .modal-close {
            position: absolute;
            top: 15px;
            right: 15px;
            font-size: 28px;
            font-weight: bold;
            color: var(--text-secondary);
            cursor: pointer;
            line-height: 1;
            padding: 5px 10px;
        }
        
        .modal-close:hover {
            color: var(--text-primary);
        }
        
        .detail-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }
        
        .detail-item {
            padding: 12px;
            background: var(--bg-tertiary);
            border-radius: 6px;
        }
        
        .detail-label {
            font-size: 12px;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 5px;
        }
        
        .detail-value {
            font-size: 16px;
            font-weight: 600;
            color: var(--text-primary);
        }
        
        .comparison-section {
            background: var(--bg-secondary);
            padding: 25px;
            border-radius: 12px;
            box-shadow: var(--shadow-md);
            margin-bottom: 25px;
        }
        
        .lens-selector {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 20px;
        }
        
        @media (max-width: 768px) {
            .header-content h1 { font-size: 22px; }
            .stats { grid-template-columns: 1fr; }
            .filter-row { grid-template-columns: 1fr; }
            .plot-grid { grid-template-columns: 1fr; }
            .lens-selector { grid-template-columns: 1fr; }
            .results-header { flex-direction: column; align-items: flex-start; }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="header-content">
                <h1>Lens Optimization Dashboard</h1>
                <p>XE Flashlamp to Fiber Coupling Optimization Results</p>
            </div>
            <div class="header-actions">
                <button class="theme-toggle" onclick="toggleTheme()">
                    <span id="theme-icon">🌙</span> Dark Mode
                </button>
            </div>
        </header>

        <div class="stats" id="stats">
            <div class="stat-card">
                <div class="stat-value" id="total-results">-</div>
                <div class="stat-label">Total Results</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="total-runs">-</div>
                <div class="stat-label">Optimization Runs</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="best-coupling">-</div>
                <div class="stat-label">Best Coupling</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="avg-coupling">-</div>
                <div class="stat-label">Average Coupling</div>
            </div>
        </div>

        <div class="tabs">
            <button class="tab active" onclick="switchTab('results')">Results</button>
            <button class="tab" onclick="switchTab('charts')">Charts</button>
            <button class="tab" onclick="switchTab('compare')">Compare</button>
            <button class="tab" onclick="switchTab('lenses')">Lens Catalog</button>
        </div>

        <div id="tab-results" class="tab-content active">
            <div class="filters">
                <h2>Filters</h2>
                <div class="filter-row">
                    <div class="filter-group">
                        <label>Min Coupling</label>
                        <input type="number" id="min-coupling" min="0" max="1" step="0.01" value="0.0" 
                               onchange="loadResults()">
                    </div>
                    <div class="filter-group">
                        <label>Max Coupling</label>
                        <input type="number" id="max-coupling" min="0" max="1" step="0.01" value="1.0"
                               onchange="loadResults()">
                    </div>
                    <div class="filter-group">
                        <label>Medium</label>
                        <select id="medium" onchange="loadResults()">
                            <option value="">All</option>
                            <option value="air">Air</option>
                            <option value="argon">Argon</option>
                            <option value="helium">Helium</option>
                        </select>
                    </div>
                    <div class="filter-group">
                        <label>Lens 1</label>
                        <input type="text" id="lens1" placeholder="e.g., LA4001" onchange="loadResults()">
                    </div>
                    <div class="filter-group">
                        <label>Lens 2</label>
                        <input type="text" id="lens2" placeholder="e.g., LA4647" onchange="loadResults()">
                    </div>
                </div>
                <div class="action-buttons">
                    <button onclick="loadResults()">Apply Filters</button>
                    <button class="secondary" onclick="exportCSV()">Export CSV</button>
                    <button class="secondary" onclick="resetFilters()">Reset</button>
                </div>
            </div>

            <div class="results-section">
                <div class="results-header">
                    <h2>Results (<span id="result-count">0</span>)</h2>
                    <div class="pagination">
                        <button onclick="changePage(-1)" id="prev-btn">Previous</button>
                        <span id="page-info">Page 1 of 1</span>
                        <button onclick="changePage(1)" id="next-btn">Next</button>
                    </div>
                </div>
                <div class="table-container">
                    <div id="results-table">
                        <div class="loading">
                            <div class="spinner"></div>
                            Loading results...
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <div id="tab-charts" class="tab-content">
            <div class="plot-section">
                <h2>Coupling Distribution</h2>
                <div class="plot-grid">
                    <div class="plot-container">
                        <canvas id="coupling-histogram"></canvas>
                    </div>
                    <div class="plot-container">
                        <canvas id="length-vs-coupling"></canvas>
                    </div>
                </div>
            </div>
            
            <div class="plot-section">
                <h2>Method Comparison</h2>
                <div class="plot-grid">
                    <div class="plot-container">
                        <canvas id="method-comparison"></canvas>
                    </div>
                    <div class="plot-container">
                        <canvas id="medium-comparison"></canvas>
                    </div>
                </div>
            </div>
        </div>

        <div id="tab-compare" class="tab-content">
            <div class="comparison-section">
                <h2>Compare Lens Pairs</h2>
                <div class="lens-selector">
                    <div class="filter-group">
                        <label>Lens Pair 1</label>
                        <select id="compare-lens1">
                            <option value="">Select lens pair...</option>
                        </select>
                    </div>
                    <div class="filter-group">
                        <label>Lens Pair 2</label>
                        <select id="compare-lens2">
                            <option value="">Select lens pair...</option>
                        </select>
                    </div>
                </div>
                <button onclick="comparelensPairs()">Compare</button>
                <div id="comparison-results" style="margin-top: 20px;"></div>
            </div>
        </div>

        <div id="tab-lenses" class="tab-content">
            <div class="filters">
                <h2>Lens Catalog Filters</h2>
                <div class="filter-row">
                    <div class="filter-group">
                        <label>Search</label>
                        <input type="text" id="lens-search" placeholder="Item number..." onchange="loadLenses()">
                    </div>
                    <div class="filter-group">
                        <label>Lens Type</label>
                        <select id="lens-type-filter" onchange="loadLenses()">
                            <option value="">All Types</option>
                            <option value="Plano-Convex">Plano-Convex</option>
                            <option value="Bi-Convex">Bi-Convex</option>
                            <option value="Aspheric">Aspheric</option>
                        </select>
                    </div>
                    <div class="filter-group">
                        <label>Vendor</label>
                        <select id="lens-vendor-filter" onchange="loadLenses()">
                            <option value="">All Vendors</option>
                            <option value="ThorLabs">ThorLabs</option>
                            <option value="Edmund">Edmund Optics</option>
                        </select>
                    </div>
                </div>
                <div class="filter-row">
                    <div class="filter-group">
                        <label>Min Diameter (mm)</label>
                        <input type="number" id="lens-min-diameter" min="0" step="0.1" 
                               placeholder="0" onchange="loadLenses()">
                    </div>
                    <div class="filter-group">
                        <label>Max Diameter (mm)</label>
                        <input type="number" id="lens-max-diameter" min="0" step="0.1" 
                               placeholder="No limit" onchange="loadLenses()">
                    </div>
                    <div class="filter-group">
                        <label>Min Focal Length (mm)</label>
                        <input type="number" id="lens-min-focal" min="0" step="0.1" 
                               placeholder="0" onchange="loadLenses()">
                    </div>
                    <div class="filter-group">
                        <label>Max Focal Length (mm)</label>
                        <input type="number" id="lens-max-focal" min="0" step="0.1" 
                               placeholder="No limit" onchange="loadLenses()">
                    </div>
                </div>
                <div class="actions">
                    <button onclick="loadLenses()" class="btn-primary">Apply Filters</button>
                    <button onclick="clearLensFilters()" class="btn-secondary">Clear Filters</button>
                </div>
            </div>

            <div class="results-summary">
                <h2>Lens Catalog</h2>
                <p><span id="lens-count">0</span> lenses found</p>
            </div>

            <div id="lenses-table" class="results-table">
                <div class="loading">
                    <div class="spinner"></div>
                    Loading lenses...
                </div>
            </div>
        </div>

        <div id="detail-modal" class="modal" onclick="closeModal(event)">
            <div class="modal-content">
                <span class="modal-close" onclick="closeModal()">&times;</span>
                <h2 id="modal-title">Result Details</h2>
                <div id="modal-body"></div>
            </div>
        </div>
    </div>

    <script>
        // Global state
        let currentPage = 1;
        const resultsPerPage = 50;
        let allResults = [];
        let sortColumn = 'coupling';
        let sortDesc = true;
        let charts = {};

        // Theme toggle
        function toggleTheme() {
            document.body.classList.toggle('dark-mode');
            const icon = document.getElementById('theme-icon');
            icon.textContent = document.body.classList.contains('dark-mode') ? '☀️' : '🌙';
            localStorage.setItem('theme', document.body.classList.contains('dark-mode') ? 'dark' : 'light');
            
            // Redraw charts if they exist
            if (Object.keys(charts).length > 0) {
                loadCharts();
            }
        }

        // Load saved theme
        if (localStorage.getItem('theme') === 'dark') {
            document.body.classList.add('dark-mode');
            document.getElementById('theme-icon').textContent = '☀️';
        }

        // Tab switching
        function switchTab(tabName) {
            document.querySelectorAll('.tab').forEach(tab => tab.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));
            
            event.target.classList.add('active');
            document.getElementById(`tab-${tabName}`).classList.add('active');
            
            if (tabName === 'charts') {
                loadCharts();
            } else if (tabName === 'compare') {
                loadLensPairs();
            } else if (tabName === 'lenses') {
                loadLenses();
                loadLensStats();
            }
        }

        // Load statistics
        async function loadStats() {
            try {
                const response = await fetch('/api/stats');
                const data = await response.json();
                
                document.getElementById('total-results').textContent = data.total_results?.toLocaleString() || 0;
                document.getElementById('total-runs').textContent = data.total_runs?.toLocaleString() || 0;
                
                if (data.best_coupling) {
                    document.getElementById('best-coupling').textContent = 
                        data.best_coupling.coupling.toFixed(4);
                }
                
                document.getElementById('avg-coupling').textContent = 
                    (data.avg_coupling || 0).toFixed(4);
            } catch (error) {
                console.error('Error loading stats:', error);
            }
        }

        // Load results with filters
        async function loadResults() {
            const minCoupling = document.getElementById('min-coupling').value;
            const maxCoupling = document.getElementById('max-coupling').value;
            const medium = document.getElementById('medium').value;
            const lens1 = document.getElementById('lens1').value;
            const lens2 = document.getElementById('lens2').value;
            
            const params = new URLSearchParams({
                min_coupling: minCoupling,
                max_coupling: maxCoupling,
                limit: 1000,
                sort_by: sortColumn,
                sort_desc: sortDesc
            });
            
            if (medium) params.append('medium', medium);
            if (lens1) params.append('lens1', lens1);
            if (lens2) params.append('lens2', lens2);
            
            try {
                document.getElementById('results-table').innerHTML = 
                    '<div class="loading"><div class="spinner"></div>Loading results...</div>';
                
                const response = await fetch(`/api/results?${params}`);
                const data = await response.json();
                
                allResults = data.results;
                currentPage = 1;
                displayResults();
            } catch (error) {
                console.error('Error loading results:', error);
                document.getElementById('results-table').innerHTML = 
                    '<div class="loading">Error loading results</div>';
            }
        }

        // Display paginated results
        function displayResults() {
            const startIdx = (currentPage - 1) * resultsPerPage;
            const endIdx = startIdx + resultsPerPage;
            const pageResults = allResults.slice(startIdx, endIdx);
            
            document.getElementById('result-count').textContent = allResults.length.toLocaleString();
            
            if (allResults.length === 0) {
                document.getElementById('results-table').innerHTML = 
                    '<div class="loading">No results found</div>';
                return;
            }
            
            const table = createTable(pageResults);
            document.getElementById('results-table').innerHTML = '';
            document.getElementById('results-table').appendChild(table);
            
            // Update pagination
            const totalPages = Math.ceil(allResults.length / resultsPerPage);
            document.getElementById('page-info').textContent = `Page ${currentPage} of ${totalPages}`;
            document.getElementById('prev-btn').disabled = currentPage === 1;
            document.getElementById('next-btn').disabled = currentPage === totalPages;
        }

        // Create sortable table
        function createTable(results) {
            const table = document.createElement('table');
            const columns = ['lens1', 'lens2', 'coupling', 'total_len_mm', 'z_l1', 'z_l2', 'z_fiber', 'method'];
            
            // Header
            const thead = document.createElement('thead');
            const headerRow = document.createElement('tr');
            
            columns.forEach(col => {
                const th = document.createElement('th');
                th.innerHTML = `${col.replace('_', ' ').toUpperCase()} <span class="sort-icon">↕</span>`;
                th.onclick = () => sortTable(col);
                if (col === sortColumn) {
                    th.classList.add('sorted');
                    th.querySelector('.sort-icon').textContent = sortDesc ? '↓' : '↑';
                }
                headerRow.appendChild(th);
            });
            thead.appendChild(headerRow);
            table.appendChild(thead);
            
            // Body
            const tbody = document.createElement('tbody');
            results.forEach((result, idx) => {
                const row = document.createElement('tr');
                row.onclick = () => showDetail(result);
                
                columns.forEach(col => {
                    const td = document.createElement('td');
                    const value = result[col];
                    
                    if (typeof value === 'number') {
                        td.textContent = value.toFixed(4);
                        if (col === 'coupling') {
                            const pct = (value * 100).toFixed(2);
                            td.title = `${pct}%`;
                        }
                    } else {
                        td.textContent = value !== undefined ? value : '-';
                    }
                    
                    row.appendChild(td);
                });
                tbody.appendChild(row);
            });
            table.appendChild(tbody);
            
            return table;
        }

        // Sort table
        function sortTable(column) {
            if (column === sortColumn) {
                sortDesc = !sortDesc;
            } else {
                sortColumn = column;
                sortDesc = true;
            }
            
            allResults.sort((a, b) => {
                const aVal = a[column];
                const bVal = b[column];
                
                if (aVal === bVal) return 0;
                if (aVal === undefined) return 1;
                if (bVal === undefined) return -1;
                
                const comparison = aVal < bVal ? -1 : 1;
                return sortDesc ? -comparison : comparison;
            });
            
            displayResults();
        }

        // Pagination
        function changePage(delta) {
            currentPage += delta;
            displayResults();
        }

        // Show detail modal
        function showDetail(result) {
            const modal = document.getElementById('detail-modal');
            const body = document.getElementById('modal-body');
            
            document.getElementById('modal-title').textContent = 
                `${result.lens1} + ${result.lens2}`;
            
            const fields = {
                'Lens 1': result.lens1,
                'Lens 2': result.lens2,
                'Coupling': (result.coupling * 100).toFixed(2) + '%',
                'Z L1': result.z_l1?.toFixed(4) + ' mm',
                'Z L2': result.z_l2?.toFixed(4) + ' mm',
                'Z Fiber': result.z_fiber?.toFixed(4) + ' mm',
                'Total Length': result.total_len_mm?.toFixed(4) + ' mm',
                'F1': result.f1_mm?.toFixed(4) + ' mm',
                'F2': result.f2_mm?.toFixed(4) + ' mm',
                'Method': result.method || '-',
                'Medium': result.medium || '-'
            };
            
            let html = '<div class="detail-grid">';
            for (const [label, value] of Object.entries(fields)) {
                html += `
                    <div class="detail-item">
                        <div class="detail-label">${label}</div>
                        <div class="detail-value">${value}</div>
                    </div>
                `;
            }
            html += '</div>';
            
            body.innerHTML = html;
            modal.classList.add('active');
        }

        // Close modal
        function closeModal(event) {
            if (!event || event.target.id === 'detail-modal') {
                document.getElementById('detail-modal').classList.remove('active');
            }
        }

        // Load charts
        async function loadCharts() {
            try {
                const response = await fetch('/api/results?limit=10000');
                const data = await response.json();
                
                if (data.results.length === 0) return;
                
                const results = data.results;
                const isDark = document.body.classList.contains('dark-mode');
                const textColor = isDark ? '#e0e0e0' : '#333333';
                const gridColor = isDark ? '#404040' : '#e1e4e8';
                
                Chart.defaults.color = textColor;
                Chart.defaults.borderColor = gridColor;
                
                // Coupling histogram
                const couplingValues = results.map(r => r.coupling);
                createHistogram('coupling-histogram', couplingValues, 'Coupling Distribution', isDark);
                
                // Length vs coupling scatter
                createScatter('length-vs-coupling', results, isDark);
                
                // Method comparison
                if (results[0].method) {
                    createMethodComparison('method-comparison', results, isDark);
                }
                
                // Medium comparison
                if (results[0].medium) {
                    createMediumComparison('medium-comparison', results, isDark);
                }
                
            } catch (error) {
                console.error('Error loading charts:', error);
            }
        }

        function createHistogram(canvasId, values, title, isDark) {
            const ctx = document.getElementById(canvasId);
            if (charts[canvasId]) charts[canvasId].destroy();
            
            const bins = 30;
            const min = Math.min(...values);
            const max = Math.max(...values);
            const binSize = (max - min) / bins;
            const histogram = new Array(bins).fill(0);
            
            values.forEach(v => {
                const binIdx = Math.min(Math.floor((v - min) / binSize), bins - 1);
                histogram[binIdx]++;
            });
            
            const labels = histogram.map((_, i) => (min + i * binSize).toFixed(3));
            
            charts[canvasId] = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: labels,
                    datasets: [{
                        label: 'Count',
                        data: histogram,
                        backgroundColor: isDark ? 'rgba(33, 150, 243, 0.6)' : 'rgba(33, 150, 243, 0.7)',
                        borderColor: 'rgba(33, 150, 243, 1)',
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    plugins: {
                        title: { display: true, text: title },
                        legend: { display: false }
                    },
                    scales: {
                        x: { title: { display: true, text: 'Coupling' } },
                        y: { title: { display: true, text: 'Frequency' } }
                    }
                }
            });
        }

        function createScatter(canvasId, results, isDark) {
            const ctx = document.getElementById(canvasId);
            if (charts[canvasId]) charts[canvasId].destroy();
            
            const data = results.map(r => ({ x: r.total_len_mm, y: r.coupling }));
            
            charts[canvasId] = new Chart(ctx, {
                type: 'scatter',
                data: {
                    datasets: [{
                        label: 'Results',
                        data: data,
                        backgroundColor: isDark ? 'rgba(33, 150, 243, 0.5)' : 'rgba(33, 150, 243, 0.6)',
                        borderColor: 'rgba(33, 150, 243, 0.8)',
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    plugins: {
                        title: { display: true, text: 'Total Length vs Coupling' },
                        legend: { display: false }
                    },
                    scales: {
                        x: { title: { display: true, text: 'Total Length (mm)' } },
                        y: { title: { display: true, text: 'Coupling' } }
                    }
                }
            });
        }

        function createMethodComparison(canvasId, results, isDark) {
            const ctx = document.getElementById(canvasId);
            if (charts[canvasId]) charts[canvasId].destroy();
            
            const methodStats = {};
            results.forEach(r => {
                if (!r.method) return;
                if (!methodStats[r.method]) {
                    methodStats[r.method] = { sum: 0, count: 0, max: 0 };
                }
                methodStats[r.method].sum += r.coupling;
                methodStats[r.method].count++;
                methodStats[r.method].max = Math.max(methodStats[r.method].max, r.coupling);
            });
            
            const methods = Object.keys(methodStats);
            const avgCoupling = methods.map(m => methodStats[m].sum / methodStats[m].count);
            const maxCoupling = methods.map(m => methodStats[m].max);
            
            charts[canvasId] = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: methods,
                    datasets: [
                        {
                            label: 'Average Coupling',
                            data: avgCoupling,
                            backgroundColor: isDark ? 'rgba(76, 175, 80, 0.6)' : 'rgba(76, 175, 80, 0.7)',
                            borderColor: 'rgba(76, 175, 80, 1)',
                            borderWidth: 1
                        },
                        {
                            label: 'Max Coupling',
                            data: maxCoupling,
                            backgroundColor: isDark ? 'rgba(33, 150, 243, 0.6)' : 'rgba(33, 150, 243, 0.7)',
                            borderColor: 'rgba(33, 150, 243, 1)',
                            borderWidth: 1
                        }
                    ]
                },
                options: {
                    responsive: true,
                    plugins: {
                        title: { display: true, text: 'Coupling by Optimization Method' }
                    },
                    scales: {
                        y: { title: { display: true, text: 'Coupling' } }
                    }
                }
            });
        }

        function createMediumComparison(canvasId, results, isDark) {
            const ctx = document.getElementById(canvasId);
            if (charts[canvasId]) charts[canvasId].destroy();
            
            const mediumStats = {};
            results.forEach(r => {
                if (!r.medium) return;
                if (!mediumStats[r.medium]) {
                    mediumStats[r.medium] = { sum: 0, count: 0 };
                }
                mediumStats[r.medium].sum += r.coupling;
                mediumStats[r.medium].count++;
            });
            
            const media = Object.keys(mediumStats);
            const avgCoupling = media.map(m => mediumStats[m].sum / mediumStats[m].count);
            
            charts[canvasId] = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: media,
                    datasets: [{
                        label: 'Average Coupling',
                        data: avgCoupling,
                        backgroundColor: ['rgba(255, 152, 0, 0.7)', 'rgba(33, 150, 243, 0.7)', 'rgba(156, 39, 176, 0.7)'],
                        borderColor: ['rgba(255, 152, 0, 1)', 'rgba(33, 150, 243, 1)', 'rgba(156, 39, 176, 1)'],
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    plugins: {
                        title: { display: true, text: 'Coupling by Medium' }
                    },
                    scales: {
                        y: { title: { display: true, text: 'Average Coupling' } }
                    }
                }
            });
        }

        // Load lens pairs for comparison
        async function loadLensPairs() {
            try {
                const response = await fetch('/api/lens_pairs');
                const data = await response.json();
                
                const select1 = document.getElementById('compare-lens1');
                const select2 = document.getElementById('compare-lens2');
                
                data.lens_pairs.forEach(pair => {
                    const option1 = document.createElement('option');
                    option1.value = pair;
                    option1.textContent = pair;
                    select1.appendChild(option1);
                    
                    const option2 = document.createElement('option');
                    option2.value = pair;
                    option2.textContent = pair;
                    select2.appendChild(option2);
                });
            } catch (error) {
                console.error('Error loading lens pairs:', error);
            }
        }

        // Compare lens pairs
        async function comparelensPairs() {
            const pair1 = document.getElementById('compare-lens1').value;
            const pair2 = document.getElementById('compare-lens2').value;
            
            if (!pair1 || !pair2) {
                alert('Please select both lens pairs');
                return;
            }
            
            const [l1_1, l1_2] = pair1.split('+');
            const [l2_1, l2_2] = pair2.split('+');
            
            try {
                const response1 = await fetch(`/api/results?lens1=${l1_1}&lens2=${l1_2}&limit=100`);
                const response2 = await fetch(`/api/results?lens1=${l2_1}&lens2=${l2_2}&limit=100`);
                
                const data1 = await response1.json();
                const data2 = await response2.json();
                
                const container = document.getElementById('comparison-results');
                
                if (data1.results.length === 0 || data2.results.length === 0) {
                    container.innerHTML = '<div class="loading">No results for one or both lens pairs</div>';
                    return;
                }
                
                const stats1 = calculateStats(data1.results);
                const stats2 = calculateStats(data2.results);
                
                container.innerHTML = `
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 20px;">
                        <div style="background: var(--bg-tertiary); padding: 20px; border-radius: 8px;">
                            <h3>${pair1}</h3>
                            <div class="detail-grid" style="margin-top: 15px;">
                                <div class="detail-item">
                                    <div class="detail-label">Results</div>
                                    <div class="detail-value">${data1.count}</div>
                                </div>
                                <div class="detail-item">
                                    <div class="detail-label">Best Coupling</div>
                                    <div class="detail-value">${stats1.max.toFixed(4)}</div>
                                </div>
                                <div class="detail-item">
                                    <div class="detail-label">Avg Coupling</div>
                                    <div class="detail-value">${stats1.avg.toFixed(4)}</div>
                                </div>
                                <div class="detail-item">
                                    <div class="detail-label">Avg Length</div>
                                    <div class="detail-value">${stats1.avgLen.toFixed(2)} mm</div>
                                </div>
                            </div>
                        </div>
                        <div style="background: var(--bg-tertiary); padding: 20px; border-radius: 8px;">
                            <h3>${pair2}</h3>
                            <div class="detail-grid" style="margin-top: 15px;">
                                <div class="detail-item">
                                    <div class="detail-label">Results</div>
                                    <div class="detail-value">${data2.count}</div>
                                </div>
                                <div class="detail-item">
                                    <div class="detail-label">Best Coupling</div>
                                    <div class="detail-value">${stats2.max.toFixed(4)}</div>
                                </div>
                                <div class="detail-item">
                                    <div class="detail-label">Avg Coupling</div>
                                    <div class="detail-value">${stats2.avg.toFixed(4)}</div>
                                </div>
                                <div class="detail-item">
                                    <div class="detail-label">Avg Length</div>
                                    <div class="detail-value">${stats2.avgLen.toFixed(2)} mm</div>
                                </div>
                            </div>
                        </div>
                    </div>
                `;
            } catch (error) {
                console.error('Error comparing lens pairs:', error);
            }
        }

        function calculateStats(results) {
            const couplings = results.map(r => r.coupling);
            const lengths = results.map(r => r.total_len_mm);
            
            return {
                max: Math.max(...couplings),
                avg: couplings.reduce((a, b) => a + b, 0) / couplings.length,
                avgLen: lengths.reduce((a, b) => a + b, 0) / lengths.length
            };
        }

        // Reset filters
        function resetFilters() {
            document.getElementById('min-coupling').value = 0.0;
            document.getElementById('max-coupling').value = 1.0;
            document.getElementById('medium').value = '';
            document.getElementById('lens1').value = '';
            document.getElementById('lens2').value = '';
            loadResults();
        }

        // Export results as CSV
        function exportCSV() {
            const minCoupling = document.getElementById('min-coupling').value;
            const medium = document.getElementById('medium').value;
            
            const params = new URLSearchParams({ min_coupling: minCoupling });
            if (medium) params.append('medium', medium);
            
            window.location.href = `/api/export?${params}`;
        }

        // Load lens catalog
        async function loadLenses() {
            const search = document.getElementById('lens-search')?.value || '';
            const lensType = document.getElementById('lens-type-filter')?.value || '';
            const vendor = document.getElementById('lens-vendor-filter')?.value || '';
            const minDiameter = document.getElementById('lens-min-diameter')?.value || '';
            const maxDiameter = document.getElementById('lens-max-diameter')?.value || '';
            const minFocal = document.getElementById('lens-min-focal')?.value || '';
            const maxFocal = document.getElementById('lens-max-focal')?.value || '';
            
            const params = new URLSearchParams({ limit: 200 });
            if (search) params.append('search', search);
            if (lensType) params.append('lens_type', lensType);
            if (vendor) params.append('vendor', vendor);
            if (minDiameter) params.append('min_diameter', minDiameter);
            if (maxDiameter) params.append('max_diameter', maxDiameter);
            if (minFocal) params.append('min_focal_length', minFocal);
            if (maxFocal) params.append('max_focal_length', maxFocal);
            
            try {
                document.getElementById('lenses-table').innerHTML = 
                    '<div class="loading"><div class="spinner"></div>Loading lenses...</div>';
                
                const response = await fetch(`/api/lenses?${params}`);
                const data = await response.json();
                
                if (data.error) {
                    document.getElementById('lenses-table').innerHTML = 
                        `<div class="loading">Error: ${data.error}</div>`;
                    return;
                }
                
                document.getElementById('lens-count').textContent = data.count.toLocaleString();
                displayLenses(data.lenses);
            } catch (error) {
                console.error('Error loading lenses:', error);
                document.getElementById('lenses-table').innerHTML = 
                    '<div class="loading">Error loading lenses</div>';
            }
        }

        // Display lenses in table
        function displayLenses(lenses) {
            if (lenses.length === 0) {
                document.getElementById('lenses-table').innerHTML = 
                    '<div class="loading">No lenses found</div>';
                return;
            }

            let html = `
                <table>
                    <thead>
                        <tr>
                            <th>Item Number</th>
                            <th>Type</th>
                            <th>Diameter (mm)</th>
                            <th>Focal Length (mm)</th>
                            <th>NA</th>
                            <th>Vendor</th>
                            <th>Actions</th>
                        </tr>
                    </thead>
                    <tbody>
            `;
            
            lenses.forEach(lens => {
                html += `
                    <tr>
                        <td><strong>${lens.item_number}</strong></td>
                        <td>${lens.lens_type || 'N/A'}</td>
                        <td>${lens.diameter_mm?.toFixed(2) || 'N/A'}</td>
                        <td>${lens.focal_length_mm?.toFixed(2) || 'N/A'}</td>
                        <td>${lens.numerical_aperture?.toFixed(3) || 'N/A'}</td>
                        <td>${lens.vendor || 'N/A'}</td>
                        <td>
                            <button onclick="showLensDetail('${lens.item_number}')" class="btn-small">
                                View Details
                            </button>
                            <button onclick="showLensResults('${lens.item_number}')" class="btn-small">
                                View Results
                            </button>
                        </td>
                    </tr>
                `;
            });
            
            html += '</tbody></table>';
            document.getElementById('lenses-table').innerHTML = html;
        }

        // Show lens detail modal
        async function showLensDetail(itemNumber) {
            try {
                const response = await fetch(`/api/lenses/${itemNumber}`);
                const lens = await response.json();
                
                if (lens.error) {
                    alert('Lens not found');
                    return;
                }
                
                let html = '<div class="detail-grid">';
                
                const fields = {
                    'Item Number': lens.item_number,
                    'Type': lens.lens_type,
                    'Diameter': lens.diameter_mm ? lens.diameter_mm.toFixed(3) + ' mm' : 'N/A',
                    'Focal Length': lens.focal_length_mm ? lens.focal_length_mm.toFixed(3) + ' mm' : 'N/A',
                    'Radius R1': lens.radius_r1_mm ? lens.radius_r1_mm.toFixed(3) + ' mm' : 'N/A',
                    'Radius R2': lens.radius_r2_mm ? lens.radius_r2_mm.toFixed(3) + ' mm' : 'N/A',
                    'Center Thickness': lens.center_thickness_mm ? lens.center_thickness_mm.toFixed(3) + ' mm' : 'N/A',
                    'Edge Thickness': lens.edge_thickness_mm ? lens.edge_thickness_mm.toFixed(3) + ' mm' : 'N/A',
                    'Back Focal Length': lens.back_focal_length_mm ? lens.back_focal_length_mm.toFixed(3) + ' mm' : 'N/A',
                    'Numerical Aperture': lens.numerical_aperture ? lens.numerical_aperture.toFixed(4) : 'N/A',
                    'Substrate': lens.substrate || 'N/A',
                    'Coating': lens.coating || 'N/A',
                    'Wavelength Range': lens.wavelength_range_nm || 'N/A',
                    'Vendor': lens.vendor || 'N/A'
                };
                
                if (lens.asphere_diameter_mm) {
                    fields['Asphere Diameter'] = lens.asphere_diameter_mm.toFixed(3) + ' mm';
                    fields['Conic Constant'] = lens.conic_constant || 'N/A';
                }
                
                if (lens.notes) {
                    fields['Notes'] = lens.notes;
                }
                
                for (const [key, value] of Object.entries(fields)) {
                    html += `
                        <div class="detail-item">
                            <div class="detail-label">${key}:</div>
                            <div class="detail-value">${value}</div>
                        </div>
                    `;
                }
                
                html += '</div>';
                
                document.getElementById('modal-title').textContent = `Lens: ${lens.item_number}`;
                document.getElementById('modal-body').innerHTML = html;
                document.getElementById('detail-modal').classList.add('show');
            } catch (error) {
                console.error('Error loading lens details:', error);
                alert('Error loading lens details');
            }
        }

        // Show optimization results for a lens
        async function showLensResults(itemNumber) {
            try {
                const response = await fetch(`/api/lenses/${itemNumber}/results`);
                const data = await response.json();
                
                if (data.error) {
                    alert('No results found for this lens');
                    return;
                }
                
                let html = `
                    <div class="detail-summary">
                        <p><strong>${data.count}</strong> optimization results found</p>
                        <p>Best Coupling: <strong>${(data.best_coupling * 100).toFixed(2)}%</strong></p>
                        <p>Average Coupling: <strong>${(data.avg_coupling * 100).toFixed(2)}%</strong></p>
                    </div>
                    <div style="max-height: 400px; overflow-y: auto;">
                        <table>
                            <thead>
                                <tr>
                                    <th>Lens Pair</th>
                                    <th>Coupling</th>
                                    <th>Medium</th>
                                    <th>Total Length</th>
                                    <th>Method</th>
                                </tr>
                            </thead>
                            <tbody>
                `;
                
                data.results.slice(0, 50).forEach(result => {
                    html += `
                        <tr>
                            <td>${result.lens1} + ${result.lens2}</td>
                            <td>${(result.coupling * 100).toFixed(2)}%</td>
                            <td>${result.medium || 'N/A'}</td>
                            <td>${result.total_len_mm?.toFixed(2) || 'N/A'} mm</td>
                            <td>${result.method || 'N/A'}</td>
                        </tr>
                    `;
                });
                
                if (data.count > 50) {
                    html += `
                        <tr>
                            <td colspan="5" style="text-align: center; font-style: italic;">
                                Showing top 50 of ${data.count} results
                            </td>
                        </tr>
                    `;
                }
                
                html += '</tbody></table></div>';
                
                document.getElementById('modal-title').textContent = `Results for ${itemNumber}`;
                document.getElementById('modal-body').innerHTML = html;
                document.getElementById('detail-modal').classList.add('show');
            } catch (error) {
                console.error('Error loading lens results:', error);
                alert('Error loading results');
            }
        }

        // Clear lens filters
        function clearLensFilters() {
            document.getElementById('lens-search').value = '';
            document.getElementById('lens-type-filter').value = '';
            document.getElementById('lens-vendor-filter').value = '';
            document.getElementById('lens-min-diameter').value = '';
            document.getElementById('lens-max-diameter').value = '';
            document.getElementById('lens-min-focal').value = '';
            document.getElementById('lens-max-focal').value = '';
            loadLenses();
        }

        // Load lens statistics
        async function loadLensStats() {
            try {
                const response = await fetch('/api/lenses/stats');
                const data = await response.json();
                
                if (!data.error) {
                    console.log('Lens catalog statistics:', data);
                    // Could add stats display to UI if desired
                }
            } catch (error) {
                console.error('Error loading lens stats:', error);
            }
        }

        // Initialize on page load
        loadStats();
        loadResults();
    </script>
</body>
</html>
"""


def start_dashboard(port=5000, db_path=None, results_dir='./results'):
    """
    Start the web dashboard server.
    
    Parameters
    ----------
    port : int
        Port to run server on (default: 5000)
    db_path : str, optional
        Path to database file
    results_dir : str
        Directory containing CSV results
    """
    server = DashboardServer(db_path=db_path, results_dir=results_dir)
    server.run(host='127.0.0.1', port=port, debug=False)
