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
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: #f5f5f5;
            padding: 20px;
        }
        .container { max-width: 1400px; margin: 0 auto; }
        header { 
            background: white; 
            padding: 20px; 
            border-radius: 8px; 
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        h1 { color: #333; margin-bottom: 10px; }
        .stats { 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); 
            gap: 15px; 
            margin: 20px 0;
        }
        .stat-card { 
            background: white; 
            padding: 15px; 
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .stat-value { font-size: 24px; font-weight: bold; color: #2196F3; }
        .stat-label { color: #666; font-size: 14px; margin-top: 5px; }
        .filters { 
            background: white; 
            padding: 20px; 
            border-radius: 8px; 
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .filter-row { 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); 
            gap: 15px; 
            margin-bottom: 15px;
        }
        .filter-group { display: flex; flex-direction: column; }
        label { font-weight: 500; margin-bottom: 5px; color: #333; }
        input, select { 
            padding: 8px; 
            border: 1px solid #ddd; 
            border-radius: 4px;
            font-size: 14px;
        }
        button { 
            padding: 10px 20px; 
            background: #2196F3; 
            color: white; 
            border: none; 
            border-radius: 4px; 
            cursor: pointer;
            font-size: 14px;
            font-weight: 500;
        }
        button:hover { background: #1976D2; }
        .results-section { 
            background: white; 
            padding: 20px; 
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        table { 
            width: 100%; 
            border-collapse: collapse; 
            margin-top: 15px;
        }
        th, td { 
            padding: 12px; 
            text-align: left; 
            border-bottom: 1px solid #eee;
        }
        th { 
            background: #f8f9fa; 
            font-weight: 600; 
            color: #333;
            position: sticky;
            top: 0;
        }
        tr:hover { background: #f8f9fa; }
        .loading { 
            text-align: center; 
            padding: 40px; 
            color: #666;
        }
        .plot-section { 
            background: white; 
            padding: 20px; 
            border-radius: 8px; 
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .plot-container { 
            margin-top: 15px; 
            text-align: center;
        }
        img { max-width: 100%; height: auto; }
        .action-buttons { 
            display: flex; 
            gap: 10px; 
            margin-bottom: 15px;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🔬 Lens Optimization Dashboard</h1>
            <p style="color: #666; margin-top: 5px;">XE Flashlamp to Fiber Coupling Optimization Results</p>
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

        <div class="filters">
            <h2 style="margin-bottom: 15px;">Filters</h2>
            <div class="filter-row">
                <div class="filter-group">
                    <label>Min Coupling</label>
                    <input type="number" id="min-coupling" min="0" max="1" step="0.01" value="0.0">
                </div>
                <div class="filter-group">
                    <label>Max Coupling</label>
                    <input type="number" id="max-coupling" min="0" max="1" step="0.01" value="1.0">
                </div>
                <div class="filter-group">
                    <label>Medium</label>
                    <select id="medium">
                        <option value="">All</option>
                        <option value="air">Air</option>
                        <option value="argon">Argon</option>
                        <option value="helium">Helium</option>
                    </select>
                </div>
                <div class="filter-group">
                    <label>Lens 1</label>
                    <input type="text" id="lens1" placeholder="e.g., LA4001">
                </div>
                <div class="filter-group">
                    <label>Lens 2</label>
                    <input type="text" id="lens2" placeholder="e.g., LA4647">
                </div>
            </div>
            <div class="action-buttons">
                <button onclick="loadResults()">Apply Filters</button>
                <button onclick="exportCSV()">Export CSV</button>
                <button onclick="loadPlots()">Load Plots</button>
            </div>
        </div>

        <div class="plot-section">
            <h2>Coupling Distribution</h2>
            <div class="plot-container">
                <img id="histogram-plot" src="/api/plot/coupling_histogram" alt="Coupling histogram">
            </div>
        </div>

        <div class="results-section">
            <h2>Results (<span id="result-count">0</span>)</h2>
            <div id="results-table">
                <div class="loading">Loading...</div>
            </div>
        </div>
    </div>

    <script>
        // Load statistics on page load
        async function loadStats() {
            try {
                const response = await fetch('/api/stats');
                const data = await response.json();
                
                document.getElementById('total-results').textContent = data.total_results || 0;
                document.getElementById('total-runs').textContent = data.total_runs || 0;
                
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
                limit: 100
            });
            
            if (medium) params.append('medium', medium);
            if (lens1) params.append('lens1', lens1);
            if (lens2) params.append('lens2', lens2);
            
            try {
                const response = await fetch(`/api/results?${params}`);
                const data = await response.json();
                
                document.getElementById('result-count').textContent = data.count;
                
                if (data.results.length === 0) {
                    document.getElementById('results-table').innerHTML = 
                        '<div class="loading">No results found</div>';
                    return;
                }
                
                const table = createTable(data.results);
                document.getElementById('results-table').innerHTML = '';
                document.getElementById('results-table').appendChild(table);
            } catch (error) {
                console.error('Error loading results:', error);
                document.getElementById('results-table').innerHTML = 
                    '<div class="loading">Error loading results</div>';
            }
        }

        // Create HTML table from results
        function createTable(results) {
            const table = document.createElement('table');
            
            // Header
            const thead = document.createElement('thead');
            const headerRow = document.createElement('tr');
            const columns = ['lens1', 'lens2', 'coupling', 'total_len_mm', 'z_l1', 'z_l2', 'z_fiber', 'method'];
            
            columns.forEach(col => {
                const th = document.createElement('th');
                th.textContent = col.replace('_', ' ').toUpperCase();
                headerRow.appendChild(th);
            });
            thead.appendChild(headerRow);
            table.appendChild(thead);
            
            // Body
            const tbody = document.createElement('tbody');
            results.forEach(result => {
                const row = document.createElement('tr');
                columns.forEach(col => {
                    const td = document.createElement('td');
                    td.textContent = result[col] !== undefined ? result[col] : '-';
                    row.appendChild(td);
                });
                tbody.appendChild(row);
            });
            table.appendChild(tbody);
            
            return table;
        }

        // Export results as CSV
        function exportCSV() {
            const minCoupling = document.getElementById('min-coupling').value;
            const medium = document.getElementById('medium').value;
            
            const params = new URLSearchParams({ min_coupling: minCoupling });
            if (medium) params.append('medium', medium);
            
            window.location.href = `/api/export?${params}`;
        }

        // Load plots
        function loadPlots() {
            document.getElementById('histogram-plot').src = 
                '/api/plot/coupling_histogram?' + new Date().getTime();
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
