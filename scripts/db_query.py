"""
Database query utility for optimization results.

Provides CLI commands to query, export, and analyze optimization results
stored in the SQLite database.
"""

import argparse
from pathlib import Path
from typing import Optional
import pandas as pd
from scripts.database import OptimizationDatabase
from scripts import consts as C


def list_runs(db: OptimizationDatabase, method: Optional[str] = None, 
              medium: Optional[str] = None):
    """List all optimization runs."""
    runs = db.get_all_runs(method=method, medium=medium)
    
    if not runs:
        print("No runs found")
        return
    
    df = pd.DataFrame(runs)
    # Select key columns for display
    display_cols = ['run_id', 'timestamp', 'method', 'medium', 'n_rays']
    df_display = df[display_cols]
    
    print(f"\nFound {len(runs)} run(s):")
    print("=" * 80)
    print(df_display.to_string(index=False))
    print("=" * 80)


def show_run(db: OptimizationDatabase, run_id: str):
    """Show details of a specific run."""
    run = db.get_run(run_id)
    
    if not run:
        print(f"Run not found: {run_id}")
        return
    
    print(f"\nRun Details: {run_id}")
    print("=" * 80)
    print(f"Timestamp: {run['timestamp']}")
    print(f"Method: {run['method']}")
    print(f"Medium: {run['medium']}")
    print(f"Number of rays: {run['n_rays']}")
    print(f"Wavelength: {run['wavelength_nm']} nm")
    print(f"Pressure: {run['pressure_atm']} atm")
    print(f"Temperature: {run['temperature_k']} K")
    print(f"Humidity: {run['humidity_fraction']}")
    
    if run.get('notes'):
        print(f"Notes: {run['notes']}")
    
    # Get statistics
    stats = db.get_statistics(run_id)
    print("\nStatistics:")
    print(f"  Total results: {stats['total_results']}")
    print(f"  Coupling: {stats['min_coupling']:.4f} - {stats['max_coupling']:.4f} (avg: {stats['avg_coupling']:.4f})")
    print(f"  Length: {stats['min_length']:.2f} - {stats['max_length']:.2f} mm (avg: {stats['avg_length']:.2f} mm)")
    if stats['avg_time']:
        print(f"  Avg computation time: {stats['avg_time']:.2f} seconds")
    print("=" * 80)


def show_results(db: OptimizationDatabase, run_id: str, limit: int = 10,
                min_coupling: Optional[float] = None, max_length: Optional[float] = None):
    """Show top results for a run."""
    results = db.get_results(run_id, min_coupling=min_coupling, 
                            max_length=max_length, limit=limit)
    
    if not results:
        print(f"No results found for run: {run_id}")
        return
    
    df = pd.DataFrame(results)
    # Select key columns for display
    display_cols = ['lens1', 'lens2', 'coupling', 'total_len_mm', 'z_l1', 'z_l2', 'orientation']
    df_display = df[display_cols]
    
    print(f"\nTop {limit} results for run {run_id}:")
    print("=" * 80)
    print(df_display.to_string(index=False))
    print("=" * 80)


def show_best(db: OptimizationDatabase, limit: int = 10, 
             medium: Optional[str] = None, min_coupling: Optional[float] = None):
    """Show best results across all runs."""
    results = db.get_best_results(limit=limit, medium=medium, min_coupling=min_coupling)
    
    if not results:
        print("No results found")
        return
    
    df = pd.DataFrame(results)
    # Select key columns for display
    display_cols = ['lens1', 'lens2', 'coupling', 'total_len_mm', 'run_id', 'medium', 'timestamp']
    df_display = df[display_cols]
    
    print(f"\nTop {limit} results across all runs:")
    print("=" * 80)
    print(df_display.to_string(index=False))
    print("=" * 80)


def lens_pair_history(db: OptimizationDatabase, lens1: str, lens2: str):
    """Show history of a specific lens pair."""
    results = db.get_lens_pair_history(lens1, lens2)
    
    if not results:
        print(f"No results found for lens pair: {lens1} + {lens2}")
        return
    
    df = pd.DataFrame(results)
    # Select key columns for display
    display_cols = ['coupling', 'total_len_mm', 'run_id', 'medium', 'method', 'timestamp']
    df_display = df[display_cols]
    
    print(f"\nHistory for lens pair {lens1} + {lens2}:")
    print("=" * 80)
    print(df_display.to_string(index=False))
    print("=" * 80)


def export_run(db: OptimizationDatabase, run_id: str, output: str):
    """Export run results to CSV."""
    db.export_to_csv(run_id, output)
    print(f"Exported run {run_id} to {output}")


def export_best(db: OptimizationDatabase, output: str, limit: int = 100, 
               medium: Optional[str] = None):
    """Export best results to CSV."""
    db.export_best_to_csv(output, limit=limit, medium=medium)
    print(f"Exported top {limit} results to {output}")


def delete_run(db: OptimizationDatabase, run_id: str, confirm: bool = False):
    """Delete a run and all its results."""
    if not confirm:
        response = input(f"Delete run {run_id} and all its results? (yes/no): ")
        if response.lower() != 'yes':
            print("Aborted")
            return
    
    db.delete_run(run_id)
    print(f"Deleted run: {run_id}")


def overall_stats(db: OptimizationDatabase):
    """Show overall database statistics."""
    stats = db.get_statistics()
    runs = db.get_all_runs()
    
    print("\nDatabase Statistics:")
    print("=" * 80)
    print(f"Total runs: {len(runs)}")
    print(f"Total results: {stats['total_results']}")
    print(f"Coupling range: {stats['min_coupling']:.4f} - {stats['max_coupling']:.4f}")
    print(f"Average coupling: {stats['avg_coupling']:.4f}")
    print(f"Length range: {stats['min_length']:.2f} - {stats['max_length']:.2f} mm")
    print(f"Average length: {stats['avg_length']:.2f} mm")
    if stats['avg_time']:
        print(f"Average computation time: {stats['avg_time']:.2f} seconds")
    print("=" * 80)


def main():
    """Main entry point for database query CLI."""
    parser = argparse.ArgumentParser(
        description='Query and manage optimization results database',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all runs
  python -m scripts.db_query list-runs
  
  # Show details of a specific run
  python -m scripts.db_query show-run <run_id>
  
  # Show top 10 results for a run
  python -m scripts.db_query show-results <run_id> --limit 10
  
  # Show best results across all runs
  python -m scripts.db_query best --limit 20 --medium air
  
  # Show history of a lens pair
  python -m scripts.db_query lens-pair LA4001 LA4647
  
  # Export run to CSV
  python -m scripts.db_query export-run <run_id> output.csv
  
  # Show overall statistics
  python -m scripts.db_query stats
        """)
    
    parser.add_argument('--db', type=str, default=None,
                       help='Path to database file (default: from config)')
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # list-runs command
    list_parser = subparsers.add_parser('list-runs', help='List all runs')
    list_parser.add_argument('--method', type=str, help='Filter by optimization method')
    list_parser.add_argument('--medium', type=str, help='Filter by medium (air/argon/helium)')
    
    # show-run command
    show_run_parser = subparsers.add_parser('show-run', help='Show details of a run')
    show_run_parser.add_argument('run_id', type=str, help='Run ID to show')
    
    # show-results command
    results_parser = subparsers.add_parser('show-results', help='Show results for a run')
    results_parser.add_argument('run_id', type=str, help='Run ID to query')
    results_parser.add_argument('--limit', type=int, default=10, help='Number of results to show')
    results_parser.add_argument('--min-coupling', type=float, help='Minimum coupling efficiency')
    results_parser.add_argument('--max-length', type=float, help='Maximum total length (mm)')
    
    # best command
    best_parser = subparsers.add_parser('best', help='Show best results across all runs')
    best_parser.add_argument('--limit', type=int, default=10, help='Number of results to show')
    best_parser.add_argument('--medium', type=str, help='Filter by medium')
    best_parser.add_argument('--min-coupling', type=float, help='Minimum coupling efficiency')
    
    # lens-pair command
    pair_parser = subparsers.add_parser('lens-pair', help='Show history of a lens pair')
    pair_parser.add_argument('lens1', type=str, help='First lens item number')
    pair_parser.add_argument('lens2', type=str, help='Second lens item number')
    
    # export-run command
    export_run_parser = subparsers.add_parser('export-run', help='Export run to CSV')
    export_run_parser.add_argument('run_id', type=str, help='Run ID to export')
    export_run_parser.add_argument('output', type=str, help='Output CSV file path')
    
    # export-best command
    export_best_parser = subparsers.add_parser('export-best', help='Export best results to CSV')
    export_best_parser.add_argument('output', type=str, help='Output CSV file path')
    export_best_parser.add_argument('--limit', type=int, default=100, help='Number of results')
    export_best_parser.add_argument('--medium', type=str, help='Filter by medium')
    
    # delete-run command
    delete_parser = subparsers.add_parser('delete-run', help='Delete a run')
    delete_parser.add_argument('run_id', type=str, help='Run ID to delete')
    delete_parser.add_argument('--yes', action='store_true', help='Skip confirmation')
    
    # stats command
    subparsers.add_parser('stats', help='Show overall database statistics')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Get database path
    db_path = args.db if args.db else C.DATABASE_PATH
    
    if not Path(db_path).exists():
        print(f"Database not found: {db_path}")
        print("Database will be created when you run optimizations with database enabled.")
        return
    
    # Execute command
    with OptimizationDatabase(db_path) as db:
        if args.command == 'list-runs':
            list_runs(db, method=args.method, medium=args.medium)
        elif args.command == 'show-run':
            show_run(db, args.run_id)
        elif args.command == 'show-results':
            show_results(db, args.run_id, limit=args.limit, 
                        min_coupling=args.min_coupling, max_length=args.max_length)
        elif args.command == 'best':
            show_best(db, limit=args.limit, medium=args.medium, 
                     min_coupling=args.min_coupling)
        elif args.command == 'lens-pair':
            lens_pair_history(db, args.lens1, args.lens2)
        elif args.command == 'export-run':
            export_run(db, args.run_id, args.output)
        elif args.command == 'export-best':
            export_best(db, args.output, limit=args.limit, medium=args.medium)
        elif args.command == 'delete-run':
            delete_run(db, args.run_id, confirm=args.yes)
        elif args.command == 'stats':
            overall_stats(db)


if __name__ == '__main__':
    main()
