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
