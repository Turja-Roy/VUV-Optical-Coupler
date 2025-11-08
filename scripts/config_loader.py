"""
Configuration file loader and validator for raytrace-xe-flashlamp-fiber.

Supports YAML configuration files with validation and merging of profiles.
Configuration values override default constants in scripts.consts module.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional


class ConfigLoader:
    """Load and validate YAML configuration files."""
    
    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize the configuration loader.
        
        Parameters
        ----------
        config_dir : str, optional
            Directory containing configuration files. Defaults to 'configs/'
            relative to project root.
        """
        if config_dir is None:
            # Assume configs/ is in project root
            project_root = Path(__file__).parent.parent
            self.config_dir = project_root / "configs"
        else:
            self.config_dir = Path(config_dir)
        
    def load(self, config_file: Optional[str] = None, profile: Optional[str] = None) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Parameters
        ----------
        config_file : str
            Path to configuration file (relative to config_dir or absolute)
        profile : str, optional
            Name of a preset profile to load (e.g., 'quick_test', 'argon_batch')
            If provided, config_file is ignored.
            
        Returns
        -------
        config : dict
            Merged configuration dictionary
            
        Notes
        -----
        Profiles are loaded from configs/<profile>.yaml and merged with
        default.yaml. Custom config files fully override defaults unless
        they explicitly inherit values.
        """
        if profile:
            # Load preset profile
            config_path = self.config_dir / f"{profile}.yaml"
            if not config_path.exists():
                raise FileNotFoundError(f"Profile not found: {profile} ({config_path})")
            # Merge with default
            default_config = self._load_yaml(self.config_dir / "default.yaml")
            profile_config = self._load_yaml(config_path)
            return self._deep_merge(default_config, profile_config)
        elif config_file:
            # Load custom config file
            config_path_obj = Path(config_file)
            if not config_path_obj.is_absolute():
                config_path_obj = self.config_dir / config_file
            
            if not config_path_obj.exists():
                raise FileNotFoundError(f"Config file not found: {config_path_obj}")
                
            return self._load_yaml(config_path_obj)
        else:
            raise ValueError("Either config_file or profile must be provided")
    
    def _load_yaml(self, path: Path) -> Dict[str, Any]:
        """Load a single YAML file."""
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
        return config if config else {}
    
    def _deep_merge(self, base: Dict, override: Dict) -> Dict:
        """
        Deep merge two dictionaries.
        
        Values in 'override' take precedence over 'base'.
        Nested dicts are merged recursively.
        """
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result
    
    def apply_to_consts(self, config: Dict[str, Any]):
        """
        Apply configuration values to scripts.consts module.
        
        Parameters
        ----------
        config : dict
            Configuration dictionary loaded from YAML
            
        Notes
        -----
        This modifies the global constants in scripts.consts at runtime,
        allowing config files to override default values without code changes.
        """
        from scripts import consts as C
        
        # Map config keys to constant names
        if 'rays' in config:
            if 'n_rays' in config['rays']:
                C.N_RAYS = config['rays']['n_rays']
            if 'use_vectorized' in config['rays']:
                C.USE_VECTORIZED_TRACING = config['rays']['use_vectorized']
        
        if 'optics' in config:
            if 'wavelength_nm' in config['optics']:
                C.WAVELENGTH_NM = config['optics']['wavelength_nm']
                # Recompute glass index for new wavelength
                from scripts.calcs import fused_silica_n
                C.N_GLASS = fused_silica_n(C.WAVELENGTH_NM)
        
        if 'source' in config:
            src = config['source']
            if 'arc_diameter_mm' in src:
                C.SOURCE_ARC_DIAM_MM = src['arc_diameter_mm']
                C.SOURCE_RADIUS_MM = C.SOURCE_ARC_DIAM_MM / 2.0
            if 'window_diameter_mm' in src:
                C.WINDOW_DIAM_MM = src['window_diameter_mm']
            if 'window_distance_mm' in src:
                C.WINDOW_DISTANCE_MM = src['window_distance_mm']
                C.SOURCE_TO_LENS_OFFSET = C.WINDOW_DISTANCE_MM + 1
            if 'max_angle_deg' in src:
                C.MAX_ANGLE_DEG = src['max_angle_deg']
        
        if 'fiber' in config:
            fib = config['fiber']
            if 'core_diameter_mm' in fib:
                C.FIBER_CORE_DIAM_MM = fib['core_diameter_mm']
            if 'numerical_aperture' in fib:
                C.NA = fib['numerical_aperture']
                import numpy as np
                C.ACCEPTANCE_HALF_RAD = np.arcsin(C.NA)
        
        if 'medium' in config:
            med = config['medium']
            if 'type' in med:
                C.MEDIUM = med['type']
            if 'pressure_atm' in med:
                C.PRESSURE_ATM = med['pressure_atm']
            if 'temperature_k' in med:
                C.TEMPERATURE_K = med['temperature_k']
            if 'humidity_fraction' in med:
                C.HUMIDITY_FRACTION = med['humidity_fraction']
        
        if 'database' in config:
            db = config['database']
            if 'enabled' in db:
                C.USE_DATABASE = db['enabled']
            if 'path' in db:
                C.DATABASE_PATH = db['path']
    
    def get_tolerance_params(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract tolerance analysis parameters from config.
        
        Parameters
        ----------
        config : dict
            Configuration dictionary
            
        Returns
        -------
        tolerance_params : dict
            Dictionary with keys: z_range_mm, n_samples, n_rays
        """
        defaults = {
            'z_range_mm': 0.5,
            'n_samples': 21,
            'n_rays': 2000,
            'test_lateral': False,
            'test_tilt': False
        }
        
        if 'tolerance' not in config:
            return defaults
        
        tol = config['tolerance']
        return {
            'z_range_mm': tol.get('z_range_mm', defaults['z_range_mm']),
            'n_samples': tol.get('n_samples', defaults['n_samples']),
            'n_rays': tol.get('n_rays', defaults['n_rays']),
            'test_lateral': tol.get('test_lateral', defaults['test_lateral']),
            'test_tilt': tol.get('test_tilt', defaults['test_tilt'])
        }
    
    def get_batch_tolerance_params(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract batch tolerance analysis parameters from config.
        
        Parameters
        ----------
        config : dict
            Configuration dictionary
            
        Returns
        -------
        batch_tolerance_params : dict
            Dictionary with keys: results_file, coupling_threshold
        """
        defaults = {
            'results_file': None,
            'coupling_threshold': None
        }
        
        if 'batch_tolerance' not in config:
            return defaults
        
        batch_tol = config['batch_tolerance']
        return {
            'results_file': batch_tol.get('results_file', defaults['results_file']),
            'coupling_threshold': batch_tol.get('coupling_threshold', defaults['coupling_threshold'])
        }
    
    def get_analyze_params(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract analyze mode parameters from config.
        
        Parameters
        ----------
        config : dict
            Configuration dictionary
            
        Returns
        -------
        analyze_params : dict
            Dictionary with keys: n_rays, coupling_threshold, methods
        """
        defaults = {
            'n_rays': 1000,
            'coupling_threshold': 0.2,
            'methods': ['differential_evolution', 'dual_annealing', 
                       'nelder_mead', 'powell', 'grid_search', 'bayesian']
        }
        
        if 'analyze' not in config:
            return defaults
        
        analyze = config['analyze']
        return {
            'n_rays': analyze.get('n_rays', defaults['n_rays']),
            'coupling_threshold': analyze.get('coupling_threshold', defaults['coupling_threshold']),
            'methods': analyze.get('methods', defaults['methods'])
        }
    
    def get_wavelength_params(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract wavelength analysis parameters from config.
        
        Parameters
        ----------
        config : dict
            Configuration dictionary
            
        Returns
        -------
        wavelength_params : dict
            Dictionary with keys: wl_start, wl_end, wl_step, n_rays, methods
        """
        defaults = {
            'wl_start': 180,
            'wl_end': 300,
            'wl_step': 10,
            'n_rays': 2000,
            'methods': ['differential_evolution', 'dual_annealing', 
                       'nelder_mead', 'powell', 'grid_search', 'bayesian']
        }
        
        if 'wavelength' not in config:
            return defaults
        
        wl = config['wavelength']
        return {
            'wl_start': wl.get('wl_start', defaults['wl_start']),
            'wl_end': wl.get('wl_end', defaults['wl_end']),
            'wl_step': wl.get('wl_step', defaults['wl_step']),
            'n_rays': wl.get('n_rays', defaults['n_rays']),
            'methods': wl.get('methods', defaults['methods'])
        }
    
    def get_wavelength_plot_params(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract wavelength plot parameters from config.
        
        Parameters
        ----------
        config : dict
            Configuration dictionary
            
        Returns
        -------
        wavelength_plot_params : dict
            Dictionary with keys: results_dir, fit_types, aggregate
        """
        defaults = {
            'results_dir': None,
            'fit_types': [],
            'aggregate': False
        }
        
        if 'wavelength_plot' not in config:
            return defaults
        
        wl_plot = config['wavelength_plot']
        
        # Handle both old 'fit_type' (single) and new 'fit_types' (list)
        fit_types = wl_plot.get('fit_types', defaults['fit_types'])
        if 'fit_type' in wl_plot and wl_plot['fit_type'] is not None:
            # Support backward compatibility with single fit_type
            if isinstance(wl_plot['fit_type'], list):
                fit_types = wl_plot['fit_type']
            else:
                fit_types = [wl_plot['fit_type']]
        
        # Handle 'all' keyword - expand to [None, 'polynomial', 'spline']
        if fit_types == 'all' or (isinstance(fit_types, list) and 'all' in fit_types):
            fit_types = [None, 'polynomial', 'spline']
        
        return {
            'results_dir': wl_plot.get('results_dir', defaults['results_dir']),
            'fit_types': fit_types if isinstance(fit_types, list) else [fit_types] if fit_types else [],
            'aggregate': wl_plot.get('aggregate', defaults['aggregate'])
        }
    
    def get_dashboard_params(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract dashboard parameters from config.
        
        Parameters
        ----------
        config : dict
            Configuration dictionary
            
        Returns
        -------
        dashboard_params : dict
            Dictionary with keys: port, db_path, auto_open
        """
        defaults = {
            'port': 5000,
            'db_path': None,
            'auto_open': False
        }
        
        if 'dashboard' not in config:
            return defaults
        
        dash = config['dashboard']
        return {
            'port': dash.get('port', defaults['port']),
            'db_path': dash.get('db_path', defaults['db_path']),
            'auto_open': dash.get('auto_open', defaults['auto_open'])
        }
    
    def get_plot_style(self, config: Dict[str, Any]) -> str:
        """
        Extract plot style from config.
        
        Parameters
        ----------
        config : dict
            Configuration dictionary
            
        Returns
        -------
        plot_style : str
            Plot style: '2d', '3d', or 'both'. Default is '2d'.
            
        Raises
        ------
        ValueError
            If plot_style is not one of the valid options.
        """
        default_style = '2d'
        
        if 'output' not in config:
            return default_style
        
        output = config['output']
        plot_style = output.get('plot_style', default_style)
        
        # Validate plot_style
        valid_styles = {'2d', '3d', 'both'}
        if plot_style not in valid_styles:
            raise ValueError(
                f"Invalid plot_style '{plot_style}'. Must be one of: {valid_styles}"
            )
        
        return plot_style
    
    def get_orientation_mode(self, config: Dict[str, Any]) -> str:
        """
        Extract orientation mode from config.
        
        Parameters
        ----------
        config : dict
            Configuration dictionary
            
        Returns
        -------
        orientation_mode : str
            Orientation mode: 'ScffcF', 'SfccfF', or 'both'. Default is 'both'.
            
        Raises
        ------
        ValueError
            If orientation is not one of the valid options.
        """
        default_mode = 'both'
        
        if 'output' not in config:
            return default_mode
        
        output = config['output']
        orientation_mode = output.get('orientation', default_mode)
        
        # Validate orientation_mode
        valid_modes = {'ScffcF', 'SfccfF', 'both'}
        if orientation_mode not in valid_modes:
            raise ValueError(
                f"Invalid orientation '{orientation_mode}'. Must be one of: {valid_modes}"
            )
        
        return orientation_mode
    
    def list_profiles(self) -> list:
        """List all available preset profiles."""
        if not self.config_dir.exists():
            return []
        profiles = []
        for yaml_file in self.config_dir.glob("*.yaml"):
            if yaml_file.stem != "default":
                profiles.append(yaml_file.stem)
        return sorted(profiles)


def load_config(config_file: Optional[str] = None, 
                profile: Optional[str] = None) -> Dict[str, Any]:
    """
    Convenience function to load configuration.
    
    Parameters
    ----------
    config_file : str, optional
        Path to custom configuration file
    profile : str, optional
        Name of preset profile to load
        
    Returns
    -------
    config : dict
        Loaded configuration dictionary
        
    Examples
    --------
    Load default configuration:
    >>> config = load_config()
    
    Load preset profile:
    >>> config = load_config(profile='quick_test')
    
    Load custom config file:
    >>> config = load_config(config_file='my_config.yaml')
    """
    loader = ConfigLoader()
    
    if profile:
        return loader.load(config_file=None, profile=profile)
    elif config_file:
        return loader.load(config_file=config_file, profile=None)
    else:
        # Load default
        return loader.load(config_file="default.yaml", profile=None)


def apply_config(config: Dict[str, Any]):
    """
    Apply configuration to global constants.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary from load_config()
    """
    loader = ConfigLoader()
    loader.apply_to_consts(config)
