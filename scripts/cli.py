import sys
import regex as re

from scripts import consts as C


def print_usage():
    """Print usage information."""
    print("""
Usage:
    python raytrace.py <command> [options]

Commands:
    particular <lens1> <lens2>    Run optimization for specific lens pair
    compare <lens1> <lens2>       Compare all optimization methods
    select                        Run all L1 x L2 combinations
    combine                       Run all combinations from Combined_Lenses.csv
    analyze                       Analyze high-coupling results with all methods

Options:
    --opt <method>                Optimization method (default: differential_evolution)
                                  Options: differential_evolution, dual_annealing,
                                           nelder_mead, powell, bayesian, grid_search
    --alpha <value>               Weight for coupling vs. length (0-1, default: 0.7)
                                  Higher = prioritize coupling more
    --medium <type>               Medium for light propagation (default: air)
                                  Options: air, argon, helium
    --coupling-threshold <value>  Minimum coupling for analyze mode (required for analyze)
    --results-file <path>         Path to results CSV file (required for analyze)
    continue                      Continue incomplete batch run
    <YYYY-MM-DD>                  Specify run date

Examples:
    # Fast global optimization (recommended)
    python raytrace.py combine --opt differential_evolution
    
    # Prioritize coupling more (90% coupling, 10% length)
    python raytrace.py combine --opt differential_evolution --alpha 0.9
    
    # Run in argon (no UV absorption)
    python raytrace.py particular LA4001 LA4647 --medium argon
    
    # Bayesian optimization (install: pip install scikit-optimize)
    python raytrace.py combine --opt bayesian
    
    # Compare methods on a test case
    python raytrace.py compare LA4001 LA4647
    
    # Use legacy grid search
    python raytrace.py combine --opt grid_search
    
    # Continue from specific date
    python raytrace.py combine continue 2025-10-14
    
    # Analyze high-coupling results with all methods
    python raytrace.py analyze --results-file results/2025-10-16/results_*.csv --coupling-threshold 0.2
""")


def parse_arguments():
    """Parse command line arguments."""
    args = {
        'mode': None,
        'lens1': None,
        'lens2': None,
        'method': None,
        'optimizer': 'differential_evolution',
        'alpha': 0.7,
        'medium': 'air',
        'continue': False,
        'date': C.DATE_STR,
        'coupling_threshold': None,
        'results_file': None
    }

    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)

    if sys.argv[1] == 'particular':
        if len(sys.argv) < 4:
            print("Error: particular mode requires two lens names")
            print_usage()
            sys.exit(1)
        args['mode'] = 'particular'
        args['lens1'] = sys.argv[2]
        args['lens2'] = sys.argv[3]

    elif sys.argv[1] == 'compare':
        if len(sys.argv) < 4:
            print("Error: compare mode requires two lens names")
            print_usage()
            sys.exit(1)
        args['mode'] = 'compare'
        args['lens1'] = sys.argv[2]
        args['lens2'] = sys.argv[3]

    elif sys.argv[1] in ['select', 'combine']:
        args['mode'] = 'method'
        args['method'] = sys.argv[1]

    elif sys.argv[1] == 'analyze':
        args['mode'] = 'analyze'

    else:
        print(f"Error: Unknown command '{sys.argv[1]}'")
        print_usage()
        sys.exit(1)

    i = 2 if args['mode'] in ['method', 'analyze'] else 4
    while i < len(sys.argv):
        arg = sys.argv[i]

        if arg == '--opt':
            if i + 1 < len(sys.argv):
                args['optimizer'] = sys.argv[i + 1]
                i += 2
            else:
                print("Error: --opt requires an optimizer name")
                sys.exit(1)

        elif arg == '--alpha':
            if i + 1 < len(sys.argv):
                try:
                    args['alpha'] = float(sys.argv[i + 1])
                    if not 0 <= args['alpha'] <= 1:
                        raise ValueError
                except ValueError:
                    print("Error: --alpha must be between 0 and 1")
                    sys.exit(1)
                i += 2
            else:
                print("Error: --alpha requires a value")
                sys.exit(1)

        elif arg == '--medium':
            if i + 1 < len(sys.argv):
                medium = sys.argv[i + 1]
                if medium not in ['air', 'argon', 'helium']:
                    print("Error: --medium must be one of: air, argon, helium")
                    sys.exit(1)
                args['medium'] = medium
                i += 2
            else:
                print("Error: --medium requires a value")
                sys.exit(1)

        elif arg == '--coupling-threshold':
            if i + 1 < len(sys.argv):
                try:
                    args['coupling_threshold'] = float(sys.argv[i + 1])
                except ValueError:
                    print("Error: --coupling-threshold must be a number")
                    sys.exit(1)
                i += 2
            else:
                print("Error: --coupling-threshold requires a value")
                sys.exit(1)

        elif arg == '--results-file':
            if i + 1 < len(sys.argv):
                args['results_file'] = sys.argv[i + 1]
                i += 2
            else:
                print("Error: --results-file requires a file path")
                sys.exit(1)

        elif arg == 'continue':
            args['continue'] = True
            i += 1

        elif re.match(r'\d{4}-\d{2}-\d{2}', arg):
            args['date'] = arg
            i += 1

        else:
            print(f"Error: Unknown argument '{arg}'")
            print_usage()
            sys.exit(1)

    if args['mode'] == 'analyze':
        if args['coupling_threshold'] is None:
            print("Error: analyze mode requires --coupling-threshold")
            print_usage()
            sys.exit(1)
        if args['results_file'] is None:
            print("Error: analyze mode requires --results-file")
            print_usage()
            sys.exit(1)

    return args
