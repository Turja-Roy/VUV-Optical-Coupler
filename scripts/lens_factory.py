"""
Lens Factory: Dynamically create appropriate lens objects based on type.

This module provides a factory function that creates the correct lens class
(PlanoConvex, BiConvex, or Aspheric) based on lens_data dictionary.

Supports both legacy dict format (with 'R_mm') and new format (with 'R1_mm', 'R2_mm').
"""

from scripts.PlanoConvex import PlanoConvex
from scripts.BiConvex import BiConvex
from scripts.Aspheric import Aspheric
from typing import Dict, Union


def create_lens(lens_data: Dict, vertex_z: float, flipped: bool = False):
    """
    Factory function to create appropriate lens object based on lens type.
    
    This function handles backward compatibility with legacy CSV data that only
    has 'R_mm' field, and new database format with 'R1_mm' and 'R2_mm' fields.
    
    Parameters:
    -----------
    lens_data : dict
        Dictionary containing lens specifications. Must include:
        - 'dia': diameter in mm
        - 'tc_mm': center thickness in mm
        - 'te_mm': edge thickness in mm
        - 'R_mm' (legacy) or 'R1_mm' (new): front surface radius
        - 'lens_type' (optional): 'Plano-Convex', 'Bi-Convex', or 'Aspheric'
        - 'R2_mm' (optional, for bi-convex/aspheric): back surface radius
        
    vertex_z : float
        Z position of the front vertex of the lens (mm)
        
    flipped : bool, optional
        Whether to flip the lens orientation (default: False)
        
    Returns:
    --------
    Lens object (PlanoConvex, BiConvex, or Aspheric)
    
    Raises:
    -------
    ValueError : If lens type is unknown or required parameters are missing
    
    Examples:
    ---------
    # Legacy format (plano-convex)
    lens_data = {'dia': 12.7, 'f_mm': 15.0, 'R_mm': 6.9, 'tc_mm': 6.0, 'te_mm': 1.8}
    lens = create_lens(lens_data, vertex_z=10.0, flipped=False)
    
    # New format (bi-convex from database)
    lens_data = {'dia': 12.7, 'f_mm': 15.0, 'R1_mm': 6.9, 'R2_mm': -6.9, 
                 'tc_mm': 6.0, 'te_mm': 1.8, 'lens_type': 'Bi-Convex'}
    lens = create_lens(lens_data, vertex_z=10.0, flipped=False)
    
    # Aspheric lens (k=0 approximation)
    lens_data = {'dia': 25.4, 'f_mm': 16.0, 'R1_mm': 7.18, 'R2_mm': 70.0,
                 'tc_mm': 16.0, 'te_mm': 2.0, 'lens_type': 'Aspheric',
                 'conic_constant': 0.0}
    lens = create_lens(lens_data, vertex_z=10.0, flipped=False)
    """
    # Determine lens type
    lens_type = lens_data.get('lens_type', 'Plano-Convex')
    
    # Get diameter (convert to radius for lens constructors)
    aperture_radius_mm = lens_data['dia'] / 2.0
    
    # Get common parameters
    center_thickness_mm = lens_data['tc_mm']
    edge_thickness_mm = lens_data['te_mm']
    
    # Handle both old format (R_mm) and new format (R1_mm)
    r1_mm = lens_data.get('R1_mm', lens_data.get('R_mm'))
    if r1_mm is None:
        raise ValueError("Lens data must contain either 'R_mm' or 'R1_mm' field")
    
    # Create appropriate lens type
    if lens_type == 'Plano-Convex':
        return PlanoConvex(
            vertex_z_front=vertex_z,
            R_front_mm=r1_mm,
            center_thickness_mm=center_thickness_mm,
            edge_thickness_mm=edge_thickness_mm,
            ap_rad_mm=aperture_radius_mm,
            flipped=flipped
        )
    
    elif lens_type == 'Bi-Convex':
        r2_mm = lens_data.get('R2_mm')
        if r2_mm is None:
            raise ValueError("Bi-Convex lens requires 'R2_mm' field in lens_data")
        
        return BiConvex(
            vertex_z_front=vertex_z,
            R_front_mm=r1_mm,
            R_back_mm=r2_mm,
            center_thickness_mm=center_thickness_mm,
            edge_thickness_mm=edge_thickness_mm,
            ap_rad_mm=aperture_radius_mm,
            flipped=flipped
        )
    
    elif lens_type == 'Aspheric':
        r2_mm = lens_data.get('R2_mm')
        if r2_mm is None:
            raise ValueError("Aspheric lens requires 'R2_mm' field in lens_data")
        
        # Get aspheric parameters if available (currently all NULL in database)
        # NOTE: Currently k=0 and no coefficients means aspheric surfaces are
        # traced as spheres with base radius. This is documented in Aspheric class.
        conic_constant = lens_data.get('conic_constant', 0.0)
        aspheric_coeffs = lens_data.get('aspheric_coeffs', None)
        
        return Aspheric(
            vertex_z_front=vertex_z,
            R_front_mm=r1_mm,  # Base radius of aspheric front surface
            R_back_mm=r2_mm,   # Radius of spherical back surface
            center_thickness_mm=center_thickness_mm,
            edge_thickness_mm=edge_thickness_mm,
            ap_rad_mm=aperture_radius_mm,
            conic_constant=conic_constant,
            aspheric_coeffs=aspheric_coeffs,
            flipped=flipped
        )
    
    else:
        raise ValueError(
            f"Unknown lens type: '{lens_type}'. "
            f"Supported types: 'Plano-Convex', 'Bi-Convex', 'Aspheric'"
        )


def convert_db_lens_to_dict(db_lens: Dict) -> Dict:
    """
    Convert a lens dictionary from database format to optimization format.
    
    Database format uses full column names (radius_r1_mm, diameter_mm, etc.)
    Optimization code expects short names (R1_mm, dia, tc_mm, etc.)
    
    Parameters:
    -----------
    db_lens : dict
        Lens dictionary from database with full column names
        
    Returns:
    --------
    dict : Lens dictionary in optimization format
    
    Examples:
    ---------
    db_lens = db.get_lens('LA4001')  # Returns database format
    opt_lens = convert_db_lens_to_dict(db_lens)  # Convert to optimization format
    lens_obj = create_lens(opt_lens, vertex_z=10.0)
    """
    return {
        'dia': db_lens['diameter_mm'],
        'f_mm': db_lens['focal_length_mm'],
        'R1_mm': db_lens['radius_r1_mm'],
        'R2_mm': db_lens.get('radius_r2_mm'),
        'tc_mm': db_lens['center_thickness_mm'],
        'te_mm': db_lens['edge_thickness_mm'],
        'BFL_mm': db_lens['back_focal_length_mm'],
        'lens_type': db_lens.get('lens_type', 'Plano-Convex'),
        'conic_constant': db_lens.get('conic_constant'),
        'aspheric_coeffs': db_lens.get('aspheric_coeffs'),
        'numerical_aperture': db_lens.get('numerical_aperture'),
        'substrate': db_lens.get('substrate'),
        'coating': db_lens.get('coating'),
        'wavelength_range_nm': db_lens.get('wavelength_range_nm'),
        'vendor': db_lens.get('vendor')
    }


def get_lens_info_string(lens_data: Dict) -> str:
    """
    Generate a human-readable string describing a lens.
    
    Parameters:
    -----------
    lens_data : dict
        Lens data dictionary
        
    Returns:
    --------
    str : Formatted lens description
    
    Examples:
    ---------
    >>> lens_data = {'dia': 12.7, 'f_mm': 15.0, 'lens_type': 'Plano-Convex'}
    >>> print(get_lens_info_string(lens_data))
    Plano-Convex, f=15.0mm, dia=12.7mm
    """
    lens_type = lens_data.get('lens_type', 'Plano-Convex')
    focal_length = lens_data.get('f_mm', 'Unknown')
    diameter = lens_data.get('dia', 'Unknown')
    vendor = lens_data.get('vendor', '')
    
    info = f"{lens_type}, f={focal_length}mm, dia={diameter}mm"
    if vendor:
        info += f" ({vendor})"
    
    return info
