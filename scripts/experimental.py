"""
Module for experimental lens setup coupling calculations.

This module allows users to specify exact lens positions and orientations
to calculate coupling efficiency for their physical optical setup.
"""

import numpy as np
from scripts import consts as C
from scripts.raytrace_helpers import sample_rays, trace_system
from scripts.lens_factory import create_lens
from scripts.data_io import fetch_lens_data


def run_experimental_setup(lens_specs, fiber_z=None, medium='air', n_rays=1000, 
                           use_database=False, lens_db=None):
    """
    Calculate coupling for an experimental lens setup with specified positions.
    
    Parameters
    ----------
    lens_specs : list of dict
        List of lens specifications, each containing:
        - name: lens model/part number (e.g., 'LA4001')
        - orientation: 'flat' (flat face first) or 'curve' (curved face first)
        - z: starting z-position in mm
            - for 'flat': z position of the flat face
            - for 'curve': z position where the curve apex is
    fiber_z : float, optional
        Z-position of the fiber face in mm. If None, will be optimized.
    medium : str, default='air'
        Propagation medium ('air', 'argon', 'helium')
    n_rays : int, default=1000
        Number of rays to trace
    use_database : bool, default=False
        Whether to use database for lens specifications
    lens_db : LensDatabase, optional
        Database instance for lens data
    
    Returns
    -------
    dict or None
        Result dictionary containing coupling and configuration info,
        or None if calculation failed.
    """
    
    # Load lens data
    lenses = fetch_lens_data('combine', use_database=use_database, db=lens_db)
    
    # Validate that all lenses exist
    for spec in lens_specs:
        if spec['name'] not in lenses:
            print(f"Error: Lens '{spec['name']}' not found in database/catalog.")
            print(f"Available lenses: {', '.join(sorted(lenses.keys()))}")
            return None
    
    # Create lens objects with specified positions and orientations
    lens_objects = []
    for i, spec in enumerate(lens_specs):
        lens_data = lenses[spec['name']]
        orientation = spec['orientation']
        z_pos = spec['z']
        
        # Determine if lens should be flipped
        # 'flat' means flat face first (flipped=True)
        # 'curve' means curved face first (flipped=False, normal orientation)
        flipped = (orientation == 'flat')
        
        # Calculate vertex_z_front based on orientation and z_pos
        if not flipped:
            # Curved face first: z_pos is the position of the curve apex
            # The vertex_z_front is where the curved surface's vertex is
            vertex_z_front = z_pos
        else:
            # Flat face first: z_pos is the position of the flat face
            # The vertex_z_front is at the flat face
            vertex_z_front = z_pos
        
        lens_obj = create_lens(lens_data, vertex_z_front, flipped=flipped)
        lens_objects.append(lens_obj)
        
        print(f"Lens {i+1} ({spec['name']}):")
        print(f"  Orientation: {orientation} face first (flipped={flipped})")
        print(f"  Input z-position: {z_pos:.2f} mm")
        print(f"  Calculated vertex_z_front: {vertex_z_front:.2f} mm")
        print(f"  Lens z_center: {lens_obj.z_center:.2f} mm")
        print(f"  Lens vertex_z_back: {lens_obj.vertex_z_back:.2f} mm")
    
    # If fiber_z not specified, optimize it
    if fiber_z is None:
        print("\nOptimizing fiber position...")
        fiber_z = _optimize_fiber_position(lens_objects, medium, n_rays)
        print(f"Optimized fiber z-position: {fiber_z:.2f} mm")
    
    # Calculate coupling with the specified/optimized fiber position (include ray data)
    coupling, n_coupled, origins, dirs, accepted = _calculate_coupling(
        lens_objects, fiber_z, medium, n_rays, return_rays=True
    )
    
    # Calculate total system length
    total_len = fiber_z  # Distance from source (at z=0) to fiber
    
    # Determine orientation strings for plotting
    orientation1_str = 'Scf' if lens_specs[0]['orientation'] == 'curve' else 'Sfc'
    orientation2_str = 'cf' if lens_specs[1]['orientation'] == 'curve' else 'fc'
    orientation_code = f"{orientation1_str}{orientation2_str}F"  # e.g., "ScfcfF" or "SfcfcF"
    
    result = {
        'lens1': lens_specs[0]['name'],
        'lens2': lens_specs[1]['name'],
        'orientation1': lens_specs[0]['orientation'],
        'orientation2': lens_specs[1]['orientation'],
        'orientation': orientation_code,  # For plotting
        'z_l1': lens_objects[0].z_center,
        'z_l2': lens_objects[1].z_center,
        'z_fiber': fiber_z,
        'coupling': coupling,
        'n_coupled': n_coupled,
        'n_rays': n_rays,
        'total_len_mm': total_len,
        'medium': medium,
        # Add ray data for plotting
        'origins': origins,
        'dirs': dirs,
        'accepted': accepted,
        # Add lens object references for plotting
        'f1_mm': lenses[lens_specs[0]['name']].get('f_mm', 0),
        'f2_mm': lenses[lens_specs[1]['name']].get('f_mm', 0)
    }
    
    return result


def _calculate_coupling(lens_objects, fiber_z, medium, n_rays, return_rays=False):
    """
    Calculate coupling efficiency for given lens configuration.
    
    Parameters
    ----------
    lens_objects : list
        List of lens objects (PlanoConvex, BiConvex, or Aspheric)
    fiber_z : float
        Z-position of fiber face in mm
    medium : str
        Propagation medium
    n_rays : int
        Number of rays to trace
    return_rays : bool, default=False
        If True, return ray data for plotting
    
    Returns
    -------
    tuple
        If return_rays=False: (coupling_efficiency, number_of_coupled_rays)
        If return_rays=True: (coupling_efficiency, number_of_coupled_rays, origins, dirs, accepted)
    """
    
    # Sample rays from source
    origins, dirs = sample_rays(n_rays, seed=42)
    
    # Trace through the system
    lens1 = lens_objects[0]
    lens2 = lens_objects[1]
    
    accepted, transmission_factors = trace_system(
        origins, dirs, lens1, lens2, fiber_z,
        C.FIBER_CORE_DIAM_MM / 2.0,  # Convert diameter to radius
        C.ACCEPTANCE_HALF_RAD,
        medium=medium,
        pressure_atm=C.PRESSURE_ATM,
        temp_k=C.TEMPERATURE_K,
        humidity_fraction=C.HUMIDITY_FRACTION
    )
    
    # Calculate coupling
    n_coupled = np.sum(accepted)
    coupling = n_coupled / n_rays
    
    # Apply transmission factors (atmospheric absorption)
    coupling_with_transmission = np.sum(transmission_factors[accepted]) / n_rays * C.GEOMETRIC_LOSS_FACTOR;
    
    if return_rays:
        return coupling_with_transmission, n_coupled, origins, dirs, accepted
    
    return coupling_with_transmission, n_coupled


def _optimize_fiber_position(lens_objects, medium, n_rays):
    """
    Optimize fiber position to maximize coupling.
    
    Uses a simple grid search around the expected focal region.
    
    Parameters
    ----------
    lens_objects : list
        List of lens objects
    medium : str
        Propagation medium
    n_rays : int
        Number of rays for optimization
    
    Returns
    -------
    float
        Optimized fiber z-position in mm
    """
    from scipy.optimize import minimize_scalar
    
    # Estimate a reasonable range based on lens positions
    lens2_back = lens_objects[1].vertex_z_back
    
    # Define objective function (negative coupling for minimization)
    def objective(z_fiber):
        coupling, _ = _calculate_coupling(lens_objects, z_fiber, medium, n_rays)
        return -coupling  # Negative because we're minimizing
    
    # Search in a range from just after lens2 to ~50mm beyond
    result = minimize_scalar(
        objective,
        bounds=(lens2_back + 1.0, lens2_back + 100.0),
        method='bounded',
        options={'xatol': 0.1}  # 0.1 mm tolerance
    )
    
    return result.x
