import sys
import regex as re

from scripts import consts as C


def print_usage():
    print("""
Usage:
    python raytrace.py <command> [options]

Commands:
    particular <lens1> <lens2>    Run optimization for specific lens pair
    compare <lens1> <lens2>       Compare all optimization methods
    select                        Run all L1 x L2 combinations (smart filtered)
    select-ext                    Run all L1 x L2 combinations (extended - all L2 diameters)
    combine                       Run all combinations from Combined_Lenses.csv
    analyze                       Analyze high-coupling results with all methods
    wavelength-analyze            Analyze wavelength dependence of lens combinations
    wavelength-analyze-plot       Create plots from wavelength analysis results
    tolerance <lens1> <lens2>     Analyze manufacturing tolerance sensitivity
    dashboard                     Start web dashboard for viewing results

Options:
    --config <file>               Load configuration from YAML file
    --profile <name>              Load preset configuration profile
                                  Available: quick_test, argon_batch, wavelength_study
    --opt <method>                Optimization method (default: differential_evolution)
                                  Options: differential_evolution, dual_annealing,
                                           nelder_mead, powell, bayesian, grid_search
    --alpha <value>               Weight for coupling vs. length (0-1, default: 0.7)
                                  Higher = prioritize coupling more
    --medium <type>               Medium for light propagation (default: air)
                                  Options: air, argon, helium
    --coupling-threshold <value>  Minimum coupling for analyze mode (required for analyze)
    --results-file <path>         Path to results CSV file (required for analyze and wavelength-analyze)
    --results-dir <path>          Path to results directory (required for wavelength-analyze-plot)
    --fit <type>                  Curve fitting type for wavelength plots (optional)
                                  Options: polynomial, spline, all (comma-separated: polynomial,spline)
    --aggregate                   Generate aggregated plots with error bars
    --wl-start <nm>               Starting wavelength for wavelength-analyze (default: 180)
    --wl-end <nm>                 Ending wavelength for wavelength-analyze (default: 300)
    --wl-step <nm>                Wavelength step for wavelength-analyze (default: 10)
    --n-rays <count>              Number of rays for ray tracing (default: 1000)
    --z-range <mm>                Range for tolerance analysis z-displacement (default: 0.5)
    --n-samples <count>           Number of tolerance test samples (default: 21)
    --port <number>               Port for dashboard server (default: 5000)
    --db <path>                   Path to database for dashboard (default: auto-detect)
    continue                      Continue incomplete batch run
    <YYYY-MM-DD>                  Specify run date

Examples:
    # Use quick test profile (100 rays, fast)
    python raytrace.py particular LA4001 LA4647 --profile quick_test
    
    # Use argon batch profile
    python raytrace.py combine --profile argon_batch
    
    # Load custom config file
    python raytrace.py combine --config my_config.yaml
    
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
    
    # Analyze manufacturing tolerance sensitivity
    python raytrace.py tolerance LA4001 LA4647 --opt powell
    
    # Tolerance with custom parameters
    python raytrace.py tolerance LA4001 LA4647 --opt powell --z-range 1.0 --n-samples 41 --n-rays 5000
    
    # Run wavelength analysis on specific lens combinations
    python raytrace.py wavelength-analyze --results-file results/2025-10-17/36-681+LA4647.csv
    
    # Custom wavelength range and finer steps
    python raytrace.py wavelength-analyze --results-file results/2025-10-17/36-681+LA4647.csv --wl-start 200 --wl-end 250 --wl-step 5
    
    # Plot wavelength analysis results
    python raytrace.py wavelength-analyze-plot --results-dir results/wavelength_analyze_2025-10-18
    
    # Plot with polynomial curve fitting
    python raytrace.py wavelength-analyze-plot --results-dir results/wavelength_analyze_2025-10-18 --fit polynomial
    
    # Plot with spline curve fitting
    python raytrace.py wavelength-analyze-plot --results-dir results/wavelength_analyze_2025-10-18 --fit spline
    
    # Plot with all fit types (no fit, polynomial, and spline)
    python raytrace.py wavelength-analyze-plot --results-dir results/wavelength_analyze_2025-10-18 --fit all
    
    # Plot with multiple specific fit types
    python raytrace.py wavelength-analyze-plot --results-dir results/wavelength_analyze_2025-10-18 --fit polynomial,spline
    
    # Plot aggregated results with error bars
    python raytrace.py wavelength-analyze-plot --results-dir results/wavelength_analyze_2025-10-18 --aggregate
    
    # Plot aggregated results with all fit types
    python raytrace.py wavelength-analyze-plot --results-dir results/wavelength_analyze_2025-10-18 --aggregate --fit all
    
    # Start web dashboard
    python raytrace.py dashboard
    
    # Dashboard with custom port and database
    python raytrace.py dashboard --port 8080 --db results/custom.db
""")


def parse_arguments():
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
        'results_file': None,
        'results_dir': None,
        'fit_types': [],
        'aggregate': False,
        'wl_start': 180,
        'wl_end': 300,
        'wl_step': 10,
        'n_rays': 1000,
        'z_range': 0.5,
        'n_samples': 21,
        'config_file': None,
        'profile': None,
        'port': 5000,
        'db_path': None
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
    
    elif sys.argv[1] == 'tolerance':
        if len(sys.argv) < 4:
            print("Error: tolerance mode requires two lens names")
            print_usage()
            sys.exit(1)
        args['mode'] = 'tolerance'
        args['lens1'] = sys.argv[2]
        args['lens2'] = sys.argv[3]

    elif sys.argv[1] in ['select', 'select-ext', 'combine']:
        args['mode'] = 'method'
        args['method'] = sys.argv[1].replace('-', '_')

    elif sys.argv[1] == 'analyze':
        args['mode'] = 'analyze'

    elif sys.argv[1] == 'wavelength-analyze':
        args['mode'] = 'wavelength-analyze'

    elif sys.argv[1] == 'wavelength-analyze-plot':
        args['mode'] = 'wavelength-analyze-plot'
    
    elif sys.argv[1] == 'dashboard':
        args['mode'] = 'dashboard'

    else:
        print(f"Error: Unknown command '{sys.argv[1]}'")
        print_usage()
        sys.exit(1)

    i = 2 if args['mode'] in ['method', 'analyze', 'wavelength-analyze', 'wavelength-analyze-plot', 'dashboard'] else 4
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

        elif arg == '--wl-start':
            if i + 1 < len(sys.argv):
                try:
                    args['wl_start'] = int(sys.argv[i + 1])
                except ValueError:
                    print("Error: --wl-start must be an integer")
                    sys.exit(1)
                i += 2
            else:
                print("Error: --wl-start requires a value")
                sys.exit(1)

        elif arg == '--wl-end':
            if i + 1 < len(sys.argv):
                try:
                    args['wl_end'] = int(sys.argv[i + 1])
                except ValueError:
                    print("Error: --wl-end must be an integer")
                    sys.exit(1)
                i += 2
            else:
                print("Error: --wl-end requires a value")
                sys.exit(1)

        elif arg == '--wl-step':
            if i + 1 < len(sys.argv):
                try:
                    args['wl_step'] = int(sys.argv[i + 1])
                except ValueError:
                    print("Error: --wl-step must be an integer")
                    sys.exit(1)
                i += 2
            else:
                print("Error: --wl-step requires a value")
                sys.exit(1)

        elif arg == '--n-rays':
            if i + 1 < len(sys.argv):
                try:
                    args['n_rays'] = int(sys.argv[i + 1])
                except ValueError:
                    print("Error: --n-rays must be an integer")
                    sys.exit(1)
                i += 2
            else:
                print("Error: --n-rays requires a value")
                sys.exit(1)
        
        elif arg == '--z-range':
            if i + 1 < len(sys.argv):
                try:
                    args['z_range'] = float(sys.argv[i + 1])
                except ValueError:
                    print("Error: --z-range must be a number")
                    sys.exit(1)
                i += 2
            else:
                print("Error: --z-range requires a value")
                sys.exit(1)
        
        elif arg == '--n-samples':
            if i + 1 < len(sys.argv):
                try:
                    args['n_samples'] = int(sys.argv[i + 1])
                except ValueError:
                    print("Error: --n-samples must be an integer")
                    sys.exit(1)
                i += 2
            else:
                print("Error: --n-samples requires a value")
                sys.exit(1)

        elif arg == '--config':
            if i + 1 < len(sys.argv):
                args['config_file'] = sys.argv[i + 1]
                i += 2
            else:
                print("Error: --config requires a file path")
                sys.exit(1)

        elif arg == '--profile':
            if i + 1 < len(sys.argv):
                args['profile'] = sys.argv[i + 1]
                i += 2
            else:
                print("Error: --profile requires a profile name")
                sys.exit(1)

        elif arg == '--results-dir':
            if i + 1 < len(sys.argv):
                args['results_dir'] = sys.argv[i + 1]
                i += 2
            else:
                print("Error: --results-dir requires a directory path")
                sys.exit(1)

        elif arg == '--fit':
            if i + 1 < len(sys.argv):
                fits = sys.argv[i + 1]
                # Support comma-separated values like "polynomial,spline"
                fit_list = [f.strip() for f in fits.split(',')]
                
                # Check if 'all' is specified
                if 'all' in fit_list:
                    if len(fit_list) > 1:
                        print("Error: --fit 'all' cannot be combined with other fit types")
                        sys.exit(1)
                    # 'all' means: no fit (None), polynomial, and spline
                    args['fit_types'] = [None, 'polynomial', 'spline']
                else:
                    # Validate each fit type
                    for fit in fit_list:
                        if fit not in ['polynomial', 'spline']:
                            print(f"Error: --fit value '{fit}' is invalid. Must be one of: polynomial, spline, all")
                            sys.exit(1)
                    # Initialize fit_types as list if not already
                    if 'fit_types' not in args:
                        args['fit_types'] = []
                    args['fit_types'].extend(fit_list)
                i += 2
            else:
                print("Error: --fit requires a type (polynomial, spline, all, or comma-separated list)")
                sys.exit(1)

        elif arg == '--aggregate':
            args['aggregate'] = True
            i += 1
        
        elif arg == '--port':
            if i + 1 < len(sys.argv):
                try:
                    args['port'] = int(sys.argv[i + 1])
                except ValueError:
                    print("Error: --port must be an integer")
                    sys.exit(1)
                i += 2
            else:
                print("Error: --port requires a value")
                sys.exit(1)
        
        elif arg == '--db':
            if i + 1 < len(sys.argv):
                args['db_path'] = sys.argv[i + 1]
                i += 2
            else:
                print("Error: --db requires a file path")
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

    # Load configuration file (auto-load default.yaml if not specified)
    if not args['config_file'] and not args['profile']:
        args['config_file'] = 'default.yaml'
    
    if args['config_file'] or args['profile']:
        from scripts.config_loader import load_config, apply_config
        
        try:
            config = load_config(config_file=args['config_file'], profile=args['profile'])
            apply_config(config)
            
            # Store config in args for optimizer parameter extraction
            args['_config'] = config
            
            # Override CLI args with config values if not explicitly set
            # (CLI arguments take precedence over config file)
            if 'optimization' in config and 'method' in config['optimization']:
                # Only use config optimizer if --opt was not provided
                if '--opt' not in sys.argv:
                    args['optimizer'] = config['optimization']['method']
            
            # Apply other config defaults if CLI args not provided
            if 'medium' in config and '--medium' not in sys.argv:
                if 'type' in config['medium']:
                    args['medium'] = config['medium']['type']
            
            if 'optimization' in config and '--alpha' not in sys.argv:
                # Alpha might be in optimization section or as top-level param
                # Check common locations
                if 'alpha' in config['optimization']:
                    args['alpha'] = config['optimization']['alpha']
            
            # Apply analyze mode config defaults
            if args['mode'] == 'analyze' and 'analyze' in config:
                if args['coupling_threshold'] is None and 'coupling_threshold' in config['analyze']:
                    args['coupling_threshold'] = config['analyze']['coupling_threshold']
                if args['results_file'] is None and 'results_file' in config['analyze']:
                    args['results_file'] = config['analyze']['results_file']
                if '--n-rays' not in sys.argv and 'n_rays' in config['analyze']:
                    args['n_rays'] = config['analyze']['n_rays']
            
            # Apply wavelength-analyze mode config defaults
            if args['mode'] == 'wavelength-analyze' and 'wavelength' in config:
                if args['results_file'] is None and 'results_file' in config['wavelength']:
                    args['results_file'] = config['wavelength']['results_file']
                if '--wl-start' not in sys.argv and 'wl_start' in config['wavelength']:
                    args['wl_start'] = config['wavelength']['wl_start']
                if '--wl-end' not in sys.argv and 'wl_end' in config['wavelength']:
                    args['wl_end'] = config['wavelength']['wl_end']
                if '--wl-step' not in sys.argv and 'wl_step' in config['wavelength']:
                    args['wl_step'] = config['wavelength']['wl_step']
                if '--n-rays' not in sys.argv and 'n_rays' in config['wavelength']:
                    args['n_rays'] = config['wavelength']['n_rays']
            
            # Apply wavelength-analyze-plot mode config defaults
            if args['mode'] == 'wavelength-analyze-plot' and 'wavelength_plot' in config:
                if args['results_dir'] is None and 'results_dir' in config['wavelength_plot']:
                    args['results_dir'] = config['wavelength_plot']['results_dir']
                # Only use config fit_types if no --fit arguments were provided via CLI
                if '--fit' not in sys.argv:
                    # Handle both 'fit_types' (list) and legacy 'fit_type' (single)
                    if 'fit_types' in config['wavelength_plot']:
                        fit_types_config = config['wavelength_plot']['fit_types']
                        # Handle 'all' keyword
                        if fit_types_config == 'all':
                            args['fit_types'] = [None, 'polynomial', 'spline']
                        elif isinstance(fit_types_config, list):
                            # Check if 'all' is in the list
                            if 'all' in fit_types_config:
                                args['fit_types'] = [None, 'polynomial', 'spline']
                            else:
                                args['fit_types'] = fit_types_config
                        elif fit_types_config:
                            args['fit_types'] = [fit_types_config]
                    elif 'fit_type' in config['wavelength_plot'] and config['wavelength_plot']['fit_type']:
                        # Backward compatibility
                        fit_type_val = config['wavelength_plot']['fit_type']
                        if fit_type_val == 'all':
                            args['fit_types'] = [None, 'polynomial', 'spline']
                        else:
                            args['fit_types'] = [fit_type_val]
                if '--aggregate' not in sys.argv and 'aggregate' in config['wavelength_plot']:
                    args['aggregate'] = config['wavelength_plot']['aggregate']
            
        except (FileNotFoundError, ValueError) as e:
            print(f"Error loading configuration: {e}")
            sys.exit(1)
    
    # Validation (after config loading so defaults are applied)
    if args['mode'] == 'analyze':
        if args['coupling_threshold'] is None:
            print("Error: analyze mode requires --coupling-threshold (or set analyze.coupling_threshold in config)")
            print_usage()
            sys.exit(1)
        if args['results_file'] is None:
            print("Error: analyze mode requires --results-file (or set analyze.results_file in config)")
            print_usage()
            sys.exit(1)

    if args['mode'] == 'wavelength-analyze':
        if args['results_file'] is None:
            print("Error: wavelength-analyze mode requires --results-file (or set wavelength.results_file in config)")
            print_usage()
            sys.exit(1)

    if args['mode'] == 'wavelength-analyze-plot':
        if args['results_dir'] is None:
            print("Error: wavelength-analyze-plot mode requires --results-dir (or set wavelength_plot.results_dir in config)")
            print_usage()
            sys.exit(1)

    return args
