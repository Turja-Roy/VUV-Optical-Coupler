import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Patch
from matplotlib.lines import Line2D

from scripts.PlanoConvex import PlanoConvex
from scripts.BiConvex import BiConvex
from scripts.Aspheric import Aspheric
from scripts.raytrace_helpers import sample_rays
from scripts import consts as C


def _get_r_mm(lens_data):
    """
    Helper function to get radius of curvature from lens data.
    Handles both legacy format (R_mm) and new database format (R1_mm).
    
    For plano-convex lenses in new format, R1_mm is the curved surface.
    For bi-convex lenses, this returns R1_mm (front surface).
    """
    return lens_data.get('R1_mm', lens_data.get('R_mm'))


def _draw_planoconvex_2d(ax, z_pos, lens_data, flipped, alpha=0.2):
    """
    Draw a plano-convex lens profile in 2D with actual curved surface.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to draw on
    z_pos : float
        Z position of the front vertex of the lens
    lens_data : dict
        Dictionary containing 'R_mm', 'tc_mm', 'te_mm', 'dia'
    flipped : bool
        If False: curved front, flat back (normal orientation)
        If True: flat front, curved back (flipped orientation)
    alpha : float
        Transparency for the fill
    """
    R = _get_r_mm(lens_data)
    tc = lens_data['tc_mm']
    te = lens_data['te_mm']
    ap_rad = lens_data['dia'] / 2.0
    
    # Calculate sag (how much the curved surface protrudes)
    # sag = R - sqrt(R^2 - ap_rad^2)
    sag = R - np.sqrt(R**2 - ap_rad**2)
    
    # Generate points for curved surface profile
    n_points = 100
    y_curve = np.linspace(-ap_rad, ap_rad, n_points)
    
    if not flipped:
        # Normal orientation: curved front, FLAT back
        # Curved surface equation: z = z_pos + R - sqrt(R^2 - y^2)
        z_curve = z_pos + R - np.sqrt(np.maximum(R**2 - y_curve**2, 0))
        
        # Back surface is perfectly flat at z = z_pos + tc
        z_flat = z_pos + tc
        
        # Build lens outline: curved front (bottom to top) + flat back (top to bottom) + close
        z_outline = np.concatenate([z_curve, [z_flat, z_flat], [z_curve[0]]])
        y_outline = np.concatenate([y_curve, [ap_rad, -ap_rad], [y_curve[0]]])
        
        # Fill the lens body
        ax.fill(z_outline, y_outline, color='b', alpha=alpha, edgecolor='b', linewidth=1.5)
        
    else:
        # Flipped orientation: flat front, curved back
        z_flat = z_pos
        # Curved back surface: center at z_pos + tc - R
        # For back surface: z = (z_pos + tc) - R + sqrt(R^2 - y^2)
        z_curve = (z_pos + tc) - R + np.sqrt(np.maximum(R**2 - y_curve**2, 0))
        
        # Build lens outline: flat front (bottom to top) + curved back (top to bottom, reversed) + close
        z_outline = np.concatenate([[z_flat, z_flat], z_curve[::-1], [z_flat]])
        y_outline = np.concatenate([[-ap_rad, ap_rad], y_curve[::-1], [-ap_rad]])
        
        # Fill the lens body
        ax.fill(z_outline, y_outline, color='b', alpha=alpha, edgecolor='b', linewidth=1.5)


def _draw_biconvex_2d(ax, z_pos, lens_data, flipped, alpha=0.2):
    """
    Draw a bi-convex lens profile in 2D with actual curved surfaces.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to draw on
    z_pos : float
        Z position of the front vertex of the lens
    lens_data : dict
        Dictionary containing 'R1_mm', 'R2_mm', 'tc_mm', 'te_mm', 'dia'
    flipped : bool
        If False: R1 front, R2 back (normal orientation)
        If True: R2 front, R1 back (flipped orientation)
    alpha : float
        Transparency for the fill
    """
    tc = lens_data['tc_mm']
    te = lens_data['te_mm']
    ap_rad = lens_data['dia'] / 2.0
    
    # Get radii - swap if flipped
    # IMPORTANT: Take absolute values to match BiConvex class behavior
    # Some lenses have negative R2 values in the database
    if not flipped:
        R_front = abs(lens_data['R1_mm'])
        R_back = abs(lens_data.get('R2_mm', lens_data['R1_mm']))  # Fallback to R1 if R2 missing
    else:
        R_front = abs(lens_data.get('R2_mm', lens_data['R1_mm']))
        R_back = abs(lens_data['R1_mm'])
    
    # Generate points for curved surface profiles
    n_points = 100
    y_curve = np.linspace(-ap_rad, ap_rad, n_points)
    
    # Front curved surface: z = z_pos + R_front - sqrt(R_front^2 - y^2)
    z_front = z_pos + R_front - np.sqrt(np.maximum(R_front**2 - y_curve**2, 0))
    
    # Back curved surface: z = (z_pos + tc) - R_back + sqrt(R_back^2 - y^2)
    z_back = (z_pos + tc) - R_back + np.sqrt(np.maximum(R_back**2 - y_curve**2, 0))
    
    # Build lens outline: front curve (bottom to top) + back curve (top to bottom, reversed) + close
    z_outline = np.concatenate([z_front, z_back[::-1], [z_front[0]]])
    y_outline = np.concatenate([y_curve, y_curve[::-1], [y_curve[0]]])
    
    # Fill the lens body
    ax.fill(z_outline, y_outline, color='b', alpha=alpha, edgecolor='b', linewidth=1.5)


def _draw_biconvex_3d(ax, z_pos, lens_data, flipped, alpha=0.3):
    """
    Draw a bi-convex lens in 3D with two curved surfaces.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes3D
        3D axis to draw on
    z_pos : float
        Z position of the front vertex of the lens
    lens_data : dict
        Dictionary containing 'R1_mm', 'R2_mm', 'tc_mm', 'dia'
    flipped : bool
        If False: R1 front, R2 back (normal orientation)
        If True: R2 front, R1 back (flipped orientation)
    alpha : float
        Transparency for the surfaces
    """
    tc = lens_data['tc_mm']
    ap_rad = lens_data['dia'] / 2.0
    
    # Get radii - swap if flipped
    # IMPORTANT: Take absolute values to match BiConvex class behavior
    # Some lenses have negative R2 values in the database
    if not flipped:
        R_front = abs(lens_data['R1_mm'])
        R_back = abs(lens_data.get('R2_mm', lens_data['R1_mm']))
    else:
        R_front = abs(lens_data.get('R2_mm', lens_data['R1_mm']))
        R_back = abs(lens_data['R1_mm'])
    
    # Azimuthal angle (full circle)
    theta = np.linspace(0, 2*np.pi, 50)
    
    # 1. Curved front surface (spherical cap bulging outward)
    # Sphere center at (0, 0, z_pos + R_front)
    # Polar angle range determined by aperture radius
    phi_max_front = np.arcsin(min(ap_rad / R_front, 1.0))
    phi_front = np.linspace(np.pi - phi_max_front, np.pi, 30)
    
    theta_grid, phi_grid = np.meshgrid(theta, phi_front)
    x_front = R_front * np.sin(phi_grid) * np.cos(theta_grid)
    y_front = R_front * np.sin(phi_grid) * np.sin(theta_grid)
    z_front = z_pos + R_front + R_front * np.cos(phi_grid)
    
    ax.plot_surface(x_front, y_front, z_front, alpha=alpha, color='b')
    
    # 2. Curved back surface (spherical cap bulging outward)
    # Sphere center at (0, 0, z_pos + tc - R_back)
    phi_max_back = np.arcsin(min(ap_rad / R_back, 1.0))
    phi_back = np.linspace(0, phi_max_back, 30)
    
    theta_grid, phi_grid = np.meshgrid(theta, phi_back)
    x_back = R_back * np.sin(phi_grid) * np.cos(theta_grid)
    y_back = R_back * np.sin(phi_grid) * np.sin(theta_grid)
    z_back = (z_pos + tc - R_back) + R_back * np.cos(phi_grid)
    
    ax.plot_surface(x_back, y_back, z_back, alpha=alpha, color='b')


def _draw_aspheric_2d(ax, z_pos, lens_data, flipped, alpha=0.2):
    """
    Draw an aspheric lens profile in 2D with actual curved surfaces.
    
    NOTE: Currently uses spherical approximation (k=0) for visualization.
    This matches the ray tracing approximation in Aspheric class.
    
    Handles three back surface types:
    - Convex (R2 > 0): Curved outward
    - Concave (R2 < 0): Curved inward
    - Plano (|R2| > 1000): Flat surface
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to draw on
    z_pos : float
        Z position of the front vertex of the lens
    lens_data : dict
        Dictionary containing 'R1_mm', 'R2_mm', 'tc_mm', 'te_mm', 'dia'
    flipped : bool
        If False: R1 front (aspheric), R2 back (spherical) - normal orientation
        If True: R2 front (spherical), R1 back (aspheric) - flipped orientation
    alpha : float
        Transparency for the fill
    """
    tc = lens_data['tc_mm']
    te = lens_data['te_mm']
    ap_rad = lens_data['dia'] / 2.0
    
    # Get radii - swap if flipped (preserve sign!)
    if not flipped:
        R_front = lens_data['R1_mm']  # Aspheric front (base radius, k=0 approx)
        R_back = lens_data.get('R2_mm', lens_data['R1_mm'])  # Spherical back
    else:
        R_front = lens_data.get('R2_mm', lens_data['R1_mm'])  # Spherical front
        R_back = lens_data['R1_mm']  # Aspheric back (base radius, k=0 approx)
    
    # Detect back surface type
    if abs(R_back) > 1000.0:
        back_type = 'plano'
    elif R_back > 0:
        back_type = 'convex'
    else:
        back_type = 'concave'
    
    # Generate points for surface profiles
    n_points = 100
    y_curve = np.linspace(-ap_rad, ap_rad, n_points)
    
    # Front curved surface: z = z_pos + abs(R_front) - sqrt(R_front^2 - y^2)
    # (Aspheric front is always convex in our lenses)
    R_front_abs = abs(R_front)
    z_front = z_pos + R_front_abs - np.sqrt(np.maximum(R_front_abs**2 - y_curve**2, 0))
    
    # Back surface depends on type
    if back_type == 'plano':
        # Flat back surface at z = z_pos + tc
        z_back = np.full_like(y_curve, z_pos + tc)
        
    elif back_type == 'convex':
        # Convex back: z = (z_pos + tc) - R_back + sqrt(R_back^2 - y^2)
        z_back = (z_pos + tc) - R_back + np.sqrt(np.maximum(R_back**2 - y_curve**2, 0))
        
    else:  # concave
        # Concave back: center at z_c = (z_pos + tc) - R_back, surface at z = z_c - sqrt(R_back^2 - y^2)
        # This simplifies to: z = (z_pos + tc) + |R_back| - sqrt(R_back^2 - y^2)
        R_back_abs = abs(R_back)
        center_z_back = z_pos + tc - R_back  # Center on +z side of vertex (R_back is negative)
        
        # ISSUE: If |R_back| < aperture radius, the concave sphere doesn't reach the edge
        # We need to blend with edge thickness constraint
        te = lens_data.get('te_mm', tc)  # edge thickness, default to center if not available
        
        # Calculate z for concave sphere (only valid where |y| <= |R_back|)
        discriminant = R_back**2 - y_curve**2
        z_back = np.zeros_like(y_curve)
        
        # For points within concave radius: use sphere equation
        valid_mask = discriminant >= 0
        z_back[valid_mask] = center_z_back - np.sqrt(discriminant[valid_mask])
        
        # For points outside concave radius: interpolate to edge thickness
        if not np.all(valid_mask):
            # Find the z-coordinate at the front surface edge
            z_front_edge = z_pos + R_front_abs - np.sqrt(np.maximum(R_front_abs**2 - y_curve**2, 0))
            z_back_edge = z_front_edge + te
            
            # At edge: back surface must be at z_front_edge + te
            # Blend from last valid concave point to edge
            z_back[~valid_mask] = z_back_edge[~valid_mask]
    
    # Build lens outline: front curve (bottom to top) + back curve (top to bottom, reversed) + close
    z_outline = np.concatenate([z_front, z_back[::-1], [z_front[0]]])
    y_outline = np.concatenate([y_curve, y_curve[::-1], [y_curve[0]]])
    
    # Fill the lens body with blue color (same as other lens types)
    ax.fill(z_outline, y_outline, color='b', alpha=alpha, edgecolor='b', linewidth=1.5)


def _draw_aspheric_3d(ax, z_pos, lens_data, flipped, alpha=0.3):
    """
    Draw an aspheric lens in 3D with two surfaces.
    
    NOTE: Currently uses spherical approximation (k=0) for visualization.
    This matches the ray tracing approximation in Aspheric class.
    
    Handles three back surface types:
    - Convex (R2 > 0): Curved outward spherical cap
    - Concave (R2 < 0): Curved inward spherical cap
    - Plano (|R2| > 1000): Flat circular disk
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes3D
        3D axis to draw on
    z_pos : float
        Z position of the front vertex of the lens
    lens_data : dict
        Dictionary containing 'R1_mm', 'R2_mm', 'tc_mm', 'dia'
    flipped : bool
        If False: R1 front (aspheric), R2 back (spherical) - normal orientation
        If True: R2 front (spherical), R1 back (aspheric) - flipped orientation
    alpha : float
        Transparency for the surfaces
    """
    tc = lens_data['tc_mm']
    ap_rad = lens_data['dia'] / 2.0
    
    # Get radii - swap if flipped (preserve sign!)
    if not flipped:
        R_front = lens_data['R1_mm']  # Aspheric front (base radius, k=0 approx)
        R_back = lens_data.get('R2_mm', lens_data['R1_mm'])  # Spherical back
    else:
        R_front = lens_data.get('R2_mm', lens_data['R1_mm'])  # Spherical front
        R_back = lens_data['R1_mm']  # Aspheric back (base radius, k=0 approx)
    
    # Detect back surface type
    if abs(R_back) > 1000.0:
        back_type = 'plano'
    elif R_back > 0:
        back_type = 'convex'
    else:
        back_type = 'concave'
    
    # Azimuthal angle (full circle)
    theta = np.linspace(0, 2*np.pi, 50)
    
    # 1. Curved front surface (spherical cap bulging outward) - using k=0 approximation
    # Sphere center at (0, 0, z_pos + R_front)
    # (Aspheric front is always convex in our lenses)
    R_front_abs = abs(R_front)
    phi_max_front = np.arcsin(min(ap_rad / R_front_abs, 1.0))
    phi_front = np.linspace(np.pi - phi_max_front, np.pi, 30)
    
    theta_grid, phi_grid = np.meshgrid(theta, phi_front)
    x_front = R_front_abs * np.sin(phi_grid) * np.cos(theta_grid)
    y_front = R_front_abs * np.sin(phi_grid) * np.sin(theta_grid)
    z_front = z_pos + R_front_abs + R_front_abs * np.cos(phi_grid)
    
    # Use same blue color as other lens types
    ax.plot_surface(x_front, y_front, z_front, alpha=alpha, color='b')
    
    # 2. Back surface - depends on type
    if back_type == 'plano':
        # Flat circular disk at z = z_pos + tc
        r_disk = np.linspace(0, ap_rad, 20)
        theta_disk = np.linspace(0, 2*np.pi, 50)
        r_grid, theta_grid = np.meshgrid(r_disk, theta_disk)
        
        x_back = r_grid * np.cos(theta_grid)
        y_back = r_grid * np.sin(theta_grid)
        z_back = np.full_like(x_back, z_pos + tc)
        
        ax.plot_surface(x_back, y_back, z_back, alpha=alpha, color='b')
        
    elif back_type == 'convex':
        # Convex back surface (spherical cap bulging outward)
        # Sphere center at (0, 0, z_pos + tc - R_back)
        phi_max_back = np.arcsin(min(ap_rad / R_back, 1.0))
        phi_back = np.linspace(0, phi_max_back, 30)
        
        theta_grid, phi_grid = np.meshgrid(theta, phi_back)
        x_back = R_back * np.sin(phi_grid) * np.cos(theta_grid)
        y_back = R_back * np.sin(phi_grid) * np.sin(theta_grid)
        z_back = (z_pos + tc - R_back) + R_back * np.cos(phi_grid)
        
        ax.plot_surface(x_back, y_back, z_back, alpha=alpha, color='b')
        
    else:  # concave
        # Concave back surface (spherical cap curving inward)
        # Center at (0, 0, z_pos + tc - R_back), where R_back < 0
        R_back_abs = abs(R_back)
        center_z_back = z_pos + tc - R_back  # Positive since R_back is negative
        
        # If concave radius is smaller than aperture, only draw the valid spherical region
        # and blend to edge thickness constraint
        if R_back_abs < ap_rad:
            # Only draw concave sphere up to its maximum radius
            effective_rad = R_back_abs * 0.99  # Slightly less to avoid numerical issues
        else:
            effective_rad = ap_rad
        
        phi_max_back = np.arcsin(min(effective_rad / R_back_abs, 1.0))
        phi_back = np.linspace(np.pi - phi_max_back, np.pi, 30)
        
        theta_grid, phi_grid = np.meshgrid(theta, phi_back)
        x_back = R_back_abs * np.sin(phi_grid) * np.cos(theta_grid)
        y_back = R_back_abs * np.sin(phi_grid) * np.sin(theta_grid)
        z_back = center_z_back + R_back_abs * np.cos(phi_grid)
        
        ax.plot_surface(x_back, y_back, z_back, alpha=alpha, color='b')
        
        # If concave doesn't reach full aperture, add a transitional surface
        if R_back_abs < ap_rad:
            # Add an outer ring/cone surface to connect concave to edge
            te = lens_data.get('te_mm', tc)
            
            # Create a conical/linear blend from effective_rad to ap_rad
            r_ring = np.linspace(effective_rad, ap_rad, 10)
            theta_ring = np.linspace(0, 2*np.pi, 50)
            r_grid_ring, theta_grid_ring = np.meshgrid(r_ring, theta_ring)
            
            x_ring = r_grid_ring * np.cos(theta_grid_ring)
            y_ring = r_grid_ring * np.sin(theta_grid_ring)
            
            # Z coordinate: interpolate from last concave point to edge thickness
            # At r=effective_rad: z = center_z_back - sqrt(R_back_abs^2 - effective_rad^2)
            z_at_effective = center_z_back - np.sqrt(R_back_abs**2 - effective_rad**2)
            # At r=ap_rad: need to calculate based on edge thickness
            # Front surface at edge: z_front_edge
            z_front_edge = z_pos + R_front_abs - np.sqrt(R_front_abs**2 - ap_rad**2)
            z_at_edge = z_front_edge + te
            
            # Linear interpolation in r
            z_ring = z_at_effective + (z_at_edge - z_at_effective) * (r_grid_ring - effective_rad) / (ap_rad - effective_rad)
            
            ax.plot_surface(x_ring, y_ring, z_ring, alpha=alpha, color='b')


def _draw_planoconvex_3d(ax, z_pos, lens_data, flipped, alpha=0.3):
    """
    Draw a plano-convex lens in 3D with actual curved surface.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes3D
        3D axis to draw on
    z_pos : float
        Z position of the front vertex of the lens
    lens_data : dict
        Dictionary containing 'R_mm', 'tc_mm', 'dia'
    flipped : bool
        If False: curved front, flat back (normal orientation)
        If True: flat front, curved back (flipped orientation)
    alpha : float
        Transparency for the surfaces
    """
    R = _get_r_mm(lens_data)
    tc = lens_data['tc_mm']
    ap_rad = lens_data['dia'] / 2.0
    
    # Azimuthal angle (full circle)
    theta = np.linspace(0, 2*np.pi, 50)
    
    if not flipped:
        # Normal orientation: curved front, flat back
        
        # 1. Curved front surface (spherical cap)
        # Sphere center at (0, 0, z_pos + R)
        # Polar angle range determined by aperture radius
        phi_max = np.arcsin(min(ap_rad / R, 1.0))  # Angle subtended by aperture
        phi = np.linspace(np.pi - phi_max, np.pi, 30)
        
        theta_grid, phi_grid = np.meshgrid(theta, phi)
        x_front = R * np.sin(phi_grid) * np.cos(theta_grid)
        y_front = R * np.sin(phi_grid) * np.sin(theta_grid)
        z_front = z_pos + R + R * np.cos(phi_grid)
        
        ax.plot_surface(x_front, y_front, z_front, alpha=alpha, color='b')
        
        # 2. Flat back surface (circular disc)
        r = np.linspace(0, ap_rad, 2)
        theta_flat = np.linspace(0, 2*np.pi, 50)
        r_grid, theta_grid = np.meshgrid(r, theta_flat)
        x_back = r_grid * np.cos(theta_grid)
        y_back = r_grid * np.sin(theta_grid)
        z_back = z_pos + tc + np.zeros_like(x_back)
        
        ax.plot_surface(x_back, y_back, z_back, alpha=alpha, color='b')
        
    else:
        # Flipped orientation: flat front, curved back
        
        # 1. Flat front surface (circular disc)
        r = np.linspace(0, ap_rad, 2)
        theta_flat = np.linspace(0, 2*np.pi, 50)
        r_grid, theta_grid = np.meshgrid(r, theta_flat)
        x_front = r_grid * np.cos(theta_grid)
        y_front = r_grid * np.sin(theta_grid)
        z_front = z_pos + np.zeros_like(x_front)
        
        ax.plot_surface(x_front, y_front, z_front, alpha=alpha, color='b')
        
        # 2. Curved back surface (spherical cap)
        # Sphere center at (0, 0, z_pos + tc - R)
        phi_max = np.arcsin(min(ap_rad / R, 1.0))
        phi = np.linspace(0, phi_max, 30)
        
        theta_grid, phi_grid = np.meshgrid(theta, phi)
        x_back = R * np.sin(phi_grid) * np.cos(theta_grid)
        y_back = R * np.sin(phi_grid) * np.sin(theta_grid)
        z_back = (z_pos + tc - R) + R * np.cos(phi_grid)
        
        ax.plot_surface(x_back, y_back, z_back, alpha=alpha, color='b')


def _plot_rays_on_axis(ax, lenses, result, n_plot_rays=1000):
    z_l1 = result['z_l1']
    z_l2 = result['z_l2']
    z_fiber = result['z_fiber']
    lens1_data = lenses[result['lens1']]
    lens2_data = lenses[result['lens2']]
    
    # Parse orientation to determine flipped flags
    orientation = result.get('orientation', 'ScffcF')
    if orientation == 'SfccfF':
        flipped1, flipped2 = True, False
    else:  # Default to 'ScffcF'
        flipped1, flipped2 = False, True

    origins, dirs = sample_rays(n_plot_rays)

    # Create lens1 based on its type
    if lens1_data.get('lens_type') == 'Bi-Convex':
        lens1 = BiConvex(z_l1, lens1_data['R1_mm'], 
                        lens1_data.get('R2_mm', lens1_data['R1_mm']),
                        lens1_data['tc_mm'], lens1_data['te_mm'], 
                        lens1_data['dia']/2.0, flipped=flipped1)
    elif lens1_data.get('lens_type') == 'Aspheric':
        lens1 = Aspheric(z_l1, lens1_data['R1_mm'],
                        lens1_data.get('R2_mm', lens1_data['R1_mm']),
                        lens1_data['tc_mm'], lens1_data['te_mm'],
                        lens1_data['dia']/2.0, 
                        conic_constant=lens1_data.get('conic_constant', 0.0),
                        flipped=flipped1)
    else:
        lens1 = PlanoConvex(z_l1, _get_r_mm(lens1_data), lens1_data['tc_mm'],
                           lens1_data['te_mm'], lens1_data['dia']/2.0, flipped=flipped1)
    
    # Create lens2 based on its type
    if lens2_data.get('lens_type') == 'Bi-Convex':
        lens2 = BiConvex(z_l2, lens2_data['R1_mm'],
                        lens2_data.get('R2_mm', lens2_data['R1_mm']),
                        lens2_data['tc_mm'], lens2_data['te_mm'],
                        lens2_data['dia']/2.0, flipped=flipped2)
    elif lens2_data.get('lens_type') == 'Aspheric':
        lens2 = Aspheric(z_l2, lens2_data['R1_mm'],
                        lens2_data.get('R2_mm', lens2_data['R1_mm']),
                        lens2_data['tc_mm'], lens2_data['te_mm'],
                        lens2_data['dia']/2.0,
                        conic_constant=lens2_data.get('conic_constant', 0.0),
                        flipped=flipped2)
    else:
        lens2 = PlanoConvex(z_l2, _get_r_mm(lens2_data), lens2_data['tc_mm'],
                           lens2_data['te_mm'], lens2_data['dia']/2.0, flipped=flipped2)

    for i in range(n_plot_rays):
        points = []
        o = origins[i].copy()
        d = dirs[i].copy()
        points.append(o)

        out1 = lens1.trace_ray(o, d, 1.0)
        if out1[2] is False:
            points = np.array(points)
            ax.plot(points[:, 0], points[:, 1], points[:, 2], 'r-', alpha=0.2)
            continue
        o1, d1 = out1[0], out1[1]
        points.append(o1)

        out2 = lens2.trace_ray(o1, d1, 1.0)
        if out2[2] is False:
            points = np.array(points)
            ax.plot(points[:, 0], points[:, 1], points[:, 2], 'r-', alpha=0.2)
            continue
        o2, d2 = out2[0], out2[1]
        points.append(o2)

        if abs(d2[2]) < 1e-9:
            continue
        t = (z_fiber - o2[2]) / d2[2]
        if t < 0:
            continue
        p_f = o2 + t * d2
        points.append(p_f)

        r = __import__("math").hypot(p_f[0], p_f[1])
        theta = __import__("math").acos(abs(d2[2]) / np.linalg.norm(d2))
        color = 'g' if (r <= C.FIBER_CORE_DIAM_MM/2.0 and
                        theta <= C.ACCEPTANCE_HALF_RAD) else 'r'

        points = np.array(points)
        ax.plot(points[:, 0], points[:, 1], points[:, 2],
                color+'-', alpha=0.5)

    # Draw lenses with actual curved surfaces
    # Detect lens type for lens 1
    if lens1_data.get('lens_type') == 'Bi-Convex':
        _draw_biconvex_3d(ax, z_l1, lens1_data, flipped=flipped1, alpha=0.2)
    elif lens1_data.get('lens_type') == 'Aspheric':
        _draw_aspheric_3d(ax, z_l1, lens1_data, flipped=flipped1, alpha=0.2)
    else:
        _draw_planoconvex_3d(ax, z_l1, lens1_data, flipped=flipped1, alpha=0.2)
    
    # Detect lens type for lens 2
    if lens2_data.get('lens_type') == 'Bi-Convex':
        _draw_biconvex_3d(ax, z_l2, lens2_data, flipped=flipped2, alpha=0.2)
    elif lens2_data.get('lens_type') == 'Aspheric':
        _draw_aspheric_3d(ax, z_l2, lens2_data, flipped=flipped2, alpha=0.2)
    else:
        _draw_planoconvex_3d(ax, z_l2, lens2_data, flipped=flipped2, alpha=0.2)

    # Draw fiber face
    theta = np.linspace(0, 2*np.pi, 100)
    r = np.linspace(0, C.FIBER_CORE_DIAM_MM/2.0, 2)
    t, r = np.meshgrid(theta, r)
    x = r * np.cos(t)
    y = r * np.sin(t)
    ax.plot_surface(x, y, z_fiber + np.zeros_like(x), alpha=0.3, color='g')

    # Draw source arc at z=0 (3D circle)
    theta_source = np.linspace(0, 2*np.pi, 100)
    r_source = C.SOURCE_ARC_DIAM_MM / 2.0
    x_source = r_source * np.cos(theta_source)
    y_source = r_source * np.sin(theta_source)
    z_source = np.zeros_like(x_source)
    ax.plot(x_source, y_source, z_source, 'orange', linewidth=2, alpha=0.8, label='Arc source')
    
    # Draw lamp window at z=8.7mm (3D circle)
    r_lamp = C.LAMP_WINDOW_DIAM_MM / 2.0
    z_lamp = C.LAMP_WINDOW_DISTANCE_MM
    x_lamp = r_lamp * np.cos(theta_source)
    y_lamp = r_lamp * np.sin(theta_source)
    z_lamp_array = np.full_like(x_lamp, z_lamp)
    ax.plot(x_lamp, y_lamp, z_lamp_array, 'yellow', linewidth=2, alpha=0.6, linestyle='--', label='Lamp window')
    
    # Draw cooling jacket aperture at z=26mm (3D circle - M23 thread)
    r_cooling = C.COOLING_JACKET_THREAD_DIAM_MM / 2.0
    z_cooling = C.WINDOW_DISTANCE_MM - 1.0
    x_cooling = r_cooling * np.cos(theta_source)
    y_cooling = r_cooling * np.sin(theta_source)
    z_cooling_array = np.full_like(x_cooling, z_cooling)
    ax.plot(x_cooling, y_cooling, z_cooling_array, 'red', linewidth=3, alpha=0.7, label='Cooling jacket (M23)')

    ax.set_box_aspect([1, 1, 1])
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.view_init(elev=20, azim=45)


def _plot_rays_2d_dual_view(fig, lenses, result, n_plot_rays=1000):
    """
    Plot ray traces in dual 2D side views (X-Z and Y-Z projections).
    
    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure object to plot on
    lenses : dict
        Dictionary of lens specifications
    result : dict
        Result dictionary containing z_l1, z_l2, z_fiber, lens1, lens2, orientation
    n_plot_rays : int
        Number of rays to plot
    """
    z_l1 = result['z_l1']
    z_l2 = result['z_l2']
    z_fiber = result['z_fiber']
    lens1_data = lenses[result['lens1']]
    lens2_data = lenses[result['lens2']]
    
    # Parse orientation to determine flipped flags
    orientation = result.get('orientation', 'ScffcF')
    if orientation == 'SfccfF':
        flipped1, flipped2 = True, False
    else:  # Default to 'ScffcF'
        flipped1, flipped2 = False, True

    origins, dirs = sample_rays(n_plot_rays)

    # Create lens1 based on its type
    if lens1_data.get('lens_type') == 'Bi-Convex':
        lens1 = BiConvex(z_l1, lens1_data['R1_mm'], 
                        lens1_data.get('R2_mm', lens1_data['R1_mm']),
                        lens1_data['tc_mm'], lens1_data['te_mm'], 
                        lens1_data['dia']/2.0, flipped=flipped1)
    elif lens1_data.get('lens_type') == 'Aspheric':
        lens1 = Aspheric(z_l1, lens1_data['R1_mm'],
                        lens1_data.get('R2_mm', lens1_data['R1_mm']),
                        lens1_data['tc_mm'], lens1_data['te_mm'],
                        lens1_data['dia']/2.0, 
                        conic_constant=lens1_data.get('conic_constant', 0.0),
                        flipped=flipped1)
    else:
        lens1 = PlanoConvex(z_l1, _get_r_mm(lens1_data), lens1_data['tc_mm'],
                           lens1_data['te_mm'], lens1_data['dia']/2.0, flipped=flipped1)
    
    # Create lens2 based on its type
    if lens2_data.get('lens_type') == 'Bi-Convex':
        lens2 = BiConvex(z_l2, lens2_data['R1_mm'],
                        lens2_data.get('R2_mm', lens2_data['R1_mm']),
                        lens2_data['tc_mm'], lens2_data['te_mm'],
                        lens2_data['dia']/2.0, flipped=flipped2)
    elif lens2_data.get('lens_type') == 'Aspheric':
        lens2 = Aspheric(z_l2, lens2_data['R1_mm'],
                        lens2_data.get('R2_mm', lens2_data['R1_mm']),
                        lens2_data['tc_mm'], lens2_data['te_mm'],
                        lens2_data['dia']/2.0,
                        conic_constant=lens2_data.get('conic_constant', 0.0),
                        flipped=flipped2)
    else:
        lens2 = PlanoConvex(z_l2, _get_r_mm(lens2_data), lens2_data['tc_mm'],
                           lens2_data['te_mm'], lens2_data['dia']/2.0, flipped=flipped2)

    # Create two subplots for X-Z and Y-Z views
    ax1 = fig.add_subplot(2, 1, 1)  # X-Z view (top)
    ax2 = fig.add_subplot(2, 1, 2)  # Y-Z view (bottom)
    
    # Trace rays and plot each segment with correct direction
    for i in range(n_plot_rays):
        o = origins[i].copy()
        d = dirs[i].copy()

        # Trace through lens1 with detailed output
        result1 = lens1.trace_ray_detailed(o, d, 1.0)
        if result1[4] is False:  # success flag is at index 4
            # Ray failed - skip plotting for rejected rays at L1
            continue
        p1_entry, d1_in, p1_exit, d1_out = result1[0], result1[1], result1[2], result1[3]

        # Trace through lens2 with detailed output
        result2 = lens2.trace_ray_detailed(p1_exit, d1_out, 1.0)
        if result2[4] is False:  # success flag is at index 4
            # Ray made it through lens1 but failed lens2 - skip
            continue
        p2_entry, d2_in, p2_exit, d2_out = result2[0], result2[1], result2[2], result2[3]

        # Calculate fiber intersection
        if abs(d2_out[2]) < 1e-9:
            continue
        t = (z_fiber - p2_exit[2]) / d2_out[2]
        if t < 0:
            continue
        p_f = p2_exit + t * d2_out

        # Check acceptance criteria for color
        r = __import__("math").hypot(p_f[0], p_f[1])
        theta = __import__("math").acos(abs(d2_out[2]) / np.linalg.norm(d2_out))
        color = 'g' if (r <= C.FIBER_CORE_DIAM_MM/2.0 and
                        theta <= C.ACCEPTANCE_HALF_RAD) else 'r'

        # Plot each segment with proper direction vectors for both views
        # Segment 1: Origin → Lens1 entry (in air, direction d)
        ax1.plot([o[2], p1_entry[2]], [o[0], p1_entry[0]], 
               color+'-', alpha=0.5, linewidth=0.5)
        ax2.plot([o[2], p1_entry[2]], [o[1], p1_entry[1]], 
               color+'-', alpha=0.5, linewidth=0.5)
        
        # Segment 2: Lens1 entry → Lens1 exit (in glass, direction d1_in)
        ax1.plot([p1_entry[2], p1_exit[2]], [p1_entry[0], p1_exit[0]], 
               color+'-', alpha=0.5, linewidth=0.5)
        ax2.plot([p1_entry[2], p1_exit[2]], [p1_entry[1], p1_exit[1]], 
               color+'-', alpha=0.5, linewidth=0.5)
        
        # Segment 3: Lens1 exit → Lens2 entry (in air, direction d1_out)
        ax1.plot([p1_exit[2], p2_entry[2]], [p1_exit[0], p2_entry[0]], 
               color+'-', alpha=0.5, linewidth=0.5)
        ax2.plot([p1_exit[2], p2_entry[2]], [p1_exit[1], p2_entry[1]], 
               color+'-', alpha=0.5, linewidth=0.5)
        
        # Segment 4: Lens2 entry → Lens2 exit (in glass, direction d2_in)
        ax1.plot([p2_entry[2], p2_exit[2]], [p2_entry[0], p2_exit[0]], 
               color+'-', alpha=0.5, linewidth=0.5)
        ax2.plot([p2_entry[2], p2_exit[2]], [p2_entry[1], p2_exit[1]], 
               color+'-', alpha=0.5, linewidth=0.5)
        
        # Segment 5: Lens2 exit → Fiber (in air, direction d2_out)
        ax1.plot([p2_exit[2], p_f[2]], [p2_exit[0], p_f[0]], 
               color+'-', alpha=0.5, linewidth=0.5)
        ax2.plot([p2_exit[2], p_f[2]], [p2_exit[1], p_f[1]], 
               color+'-', alpha=0.5, linewidth=0.5)

    # Draw lenses with actual curved profiles
    # Detect lens type for lens 1
    if lens1_data.get('lens_type') == 'Bi-Convex':
        _draw_biconvex_2d(ax1, z_l1, lens1_data, flipped=flipped1, alpha=0.2)
        _draw_biconvex_2d(ax2, z_l1, lens1_data, flipped=flipped1, alpha=0.2)
    elif lens1_data.get('lens_type') == 'Aspheric':
        _draw_aspheric_2d(ax1, z_l1, lens1_data, flipped=flipped1, alpha=0.2)
        _draw_aspheric_2d(ax2, z_l1, lens1_data, flipped=flipped1, alpha=0.2)
    else:
        _draw_planoconvex_2d(ax1, z_l1, lens1_data, flipped=flipped1, alpha=0.2)
        _draw_planoconvex_2d(ax2, z_l1, lens1_data, flipped=flipped1, alpha=0.2)
    
    # Detect lens type for lens 2
    if lens2_data.get('lens_type') == 'Bi-Convex':
        _draw_biconvex_2d(ax1, z_l2, lens2_data, flipped=flipped2, alpha=0.2)
        _draw_biconvex_2d(ax2, z_l2, lens2_data, flipped=flipped2, alpha=0.2)
    elif lens2_data.get('lens_type') == 'Aspheric':
        _draw_aspheric_2d(ax1, z_l2, lens2_data, flipped=flipped2, alpha=0.2)
        _draw_aspheric_2d(ax2, z_l2, lens2_data, flipped=flipped2, alpha=0.2)
    else:
        _draw_planoconvex_2d(ax1, z_l2, lens2_data, flipped=flipped2, alpha=0.2)
        _draw_planoconvex_2d(ax2, z_l2, lens2_data, flipped=flipped2, alpha=0.2)
    
    # Draw fiber as vertical line
    fiber_half_dia = C.FIBER_CORE_DIAM_MM / 2.0
    ax1.plot([z_fiber, z_fiber], [-fiber_half_dia, fiber_half_dia], 
             'g-', linewidth=3, alpha=0.6, label='Fiber')
    ax2.plot([z_fiber, z_fiber], [-fiber_half_dia, fiber_half_dia], 
             'g-', linewidth=3, alpha=0.6, label='Fiber')
    
    # Mark fiber endpoints
    ax1.plot(z_fiber, -fiber_half_dia, 'go', markersize=5)
    ax1.plot(z_fiber, fiber_half_dia, 'go', markersize=5)
    ax2.plot(z_fiber, -fiber_half_dia, 'go', markersize=5)
    ax2.plot(z_fiber, fiber_half_dia, 'go', markersize=5)
    
    # Draw source arc at z=0
    source_radius = C.SOURCE_ARC_DIAM_MM / 2.0
    ax1.plot([0, 0], [-source_radius, source_radius], 
             'orange', linewidth=2, alpha=0.8, label='Arc source')
    ax2.plot([0, 0], [-source_radius, source_radius], 
             'orange', linewidth=2, alpha=0.8, label='Arc source')
    ax1.plot(0, -source_radius, 'o', color='orange', markersize=5)
    ax1.plot(0, source_radius, 'o', color='orange', markersize=5)
    ax2.plot(0, -source_radius, 'o', color='orange', markersize=5)
    ax2.plot(0, source_radius, 'o', color='orange', markersize=5)
    
    # Draw lamp window at z=8.7mm
    lamp_window_radius = C.LAMP_WINDOW_DIAM_MM / 2.0
    z_lamp_window = C.LAMP_WINDOW_DISTANCE_MM
    ax1.plot([z_lamp_window, z_lamp_window], [-lamp_window_radius, lamp_window_radius], 
             'yellow', linewidth=2, alpha=0.6, linestyle='--', label='Lamp window')
    ax2.plot([z_lamp_window, z_lamp_window], [-lamp_window_radius, lamp_window_radius], 
             'yellow', linewidth=2, alpha=0.6, linestyle='--', label='Lamp window')
    
    # Draw cooling jacket aperture at z=26mm (M23 thread)
    cooling_jacket_radius = C.COOLING_JACKET_THREAD_DIAM_MM / 2.0
    z_cooling_jacket = C.WINDOW_DISTANCE_MM - 1.0
    ax1.plot([z_cooling_jacket, z_cooling_jacket], [-cooling_jacket_radius, cooling_jacket_radius], 
             'red', linewidth=3, alpha=0.7, label='Cooling jacket (M23)')
    ax2.plot([z_cooling_jacket, z_cooling_jacket], [-cooling_jacket_radius, cooling_jacket_radius], 
             'red', linewidth=3, alpha=0.7, label='Cooling jacket (M23)')
    ax1.plot(z_cooling_jacket, -cooling_jacket_radius, 'rs', markersize=6)
    ax1.plot(z_cooling_jacket, cooling_jacket_radius, 'rs', markersize=6)
    ax2.plot(z_cooling_jacket, -cooling_jacket_radius, 'rs', markersize=6)
    ax2.plot(z_cooling_jacket, cooling_jacket_radius, 'rs', markersize=6)
    
    # Set labels and formatting
    ax1.set_ylabel('X (mm)', fontsize=10)
    ax1.set_title('X-Z Projection (Horizontal)', fontsize=11, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal', adjustable='box')
    
    ax2.set_xlabel('Z (mm)', fontsize=10)
    ax2.set_ylabel('Y (mm)', fontsize=10)
    ax2.set_title('Y-Z Projection (Vertical)', fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal', adjustable='box')
    
    # Add legend to first subplot only
    legend_elements = [
        Patch(facecolor='g', alpha=0.5, label='Accepted rays'),
        Patch(facecolor='r', alpha=0.5, label='Rejected rays'),
        Patch(facecolor='b', alpha=0.2, edgecolor='b', label='Lenses'),
        Line2D([0], [0], color='orange', linewidth=2, alpha=0.8, label='Arc source (Ø3mm)'),
        Line2D([0], [0], color='yellow', linewidth=2, alpha=0.6, linestyle='--', label='Lamp window (Ø14.3mm)'),
        Line2D([0], [0], color='red', linewidth=3, alpha=0.7, label='Cooling jacket M23 (Ø23mm)'),
        Line2D([0], [0], color='g', linewidth=3, alpha=0.6, label='Fiber core (Ø1mm)')
    ]
    ax1.legend(handles=legend_elements, loc='upper right', fontsize=8)


def _plot_single_2d_view(ax, lenses, result, n_plot_rays=500, projection='xz'):
    """
    Plot ray traces in a single 2D view (for combined method comparisons).
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis object to plot on
    lenses : dict
        Dictionary of lens specifications
    result : dict
        Result dictionary containing z_l1, z_l2, z_fiber, lens1, lens2, orientation
    n_plot_rays : int
        Number of rays to plot
    projection : str
        Either 'xz' for X-Z projection or 'yz' for Y-Z projection
    """
    z_l1 = result['z_l1']
    z_l2 = result['z_l2']
    z_fiber = result['z_fiber']
    lens1_data = lenses[result['lens1']]
    lens2_data = lenses[result['lens2']]
    
    # Parse orientation to determine flipped flags
    orientation = result.get('orientation', 'ScffcF')
    if orientation == 'SfccfF':
        flipped1, flipped2 = True, False
    else:  # Default to 'ScffcF'
        flipped1, flipped2 = False, True

    origins, dirs = sample_rays(n_plot_rays)

    # Create lens1 based on its type
    if lens1_data.get('lens_type') == 'Bi-Convex':
        lens1 = BiConvex(z_l1, lens1_data['R1_mm'], 
                        lens1_data.get('R2_mm', lens1_data['R1_mm']),
                        lens1_data['tc_mm'], lens1_data['te_mm'], 
                        lens1_data['dia']/2.0, flipped=flipped1)
    elif lens1_data.get('lens_type') == 'Aspheric':
        lens1 = Aspheric(z_l1, lens1_data['R1_mm'],
                        lens1_data.get('R2_mm', lens1_data['R1_mm']),
                        lens1_data['tc_mm'], lens1_data['te_mm'],
                        lens1_data['dia']/2.0, 
                        conic_constant=lens1_data.get('conic_constant', 0.0),
                        flipped=flipped1)
    else:
        lens1 = PlanoConvex(z_l1, _get_r_mm(lens1_data), lens1_data['tc_mm'],
                           lens1_data['te_mm'], lens1_data['dia']/2.0, flipped=flipped1)
    
    # Create lens2 based on its type
    if lens2_data.get('lens_type') == 'Bi-Convex':
        lens2 = BiConvex(z_l2, lens2_data['R1_mm'],
                        lens2_data.get('R2_mm', lens2_data['R1_mm']),
                        lens2_data['tc_mm'], lens2_data['te_mm'],
                        lens2_data['dia']/2.0, flipped=flipped2)
    elif lens2_data.get('lens_type') == 'Aspheric':
        lens2 = Aspheric(z_l2, lens2_data['R1_mm'],
                        lens2_data.get('R2_mm', lens2_data['R1_mm']),
                        lens2_data['tc_mm'], lens2_data['te_mm'],
                        lens2_data['dia']/2.0,
                        conic_constant=lens2_data.get('conic_constant', 0.0),
                        flipped=flipped2)
    else:
        lens2 = PlanoConvex(z_l2, _get_r_mm(lens2_data), lens2_data['tc_mm'],
                           lens2_data['te_mm'], lens2_data['dia']/2.0, flipped=flipped2)

    coord_idx = 0 if projection == 'xz' else 1  # X=0, Y=1
    coord_label = 'X' if projection == 'xz' else 'Y'
    
    # Trace rays and plot each segment with correct direction
    for i in range(n_plot_rays):
        o = origins[i].copy()
        d = dirs[i].copy()

        # Trace through lens1 with detailed output
        result1 = lens1.trace_ray_detailed(o, d, 1.0)
        if result1[4] is False:  # success flag is at index 4
            # Ray failed - just draw from origin toward lens1
            t_failed = 5.0  # Draw 5mm segment
            p_failed = o + t_failed * d
            ax.plot([o[2], p_failed[2]], [o[coord_idx], p_failed[coord_idx]], 
                   'r-', alpha=0.2, linewidth=0.5)
            continue
        p1_entry, d1_in, p1_exit, d1_out = result1[0], result1[1], result1[2], result1[3]

        # Trace through lens2 with detailed output
        result2 = lens2.trace_ray_detailed(p1_exit, d1_out, 1.0)
        if result2[4] is False:  # success flag is at index 4
            # Ray made it through lens1 but failed lens2
            # Draw: origin→lens1_entry, lens1_entry→lens1_exit, lens1_exit→partial
            ax.plot([o[2], p1_entry[2]], [o[coord_idx], p1_entry[coord_idx]], 
                   'r-', alpha=0.2, linewidth=0.5)
            ax.plot([p1_entry[2], p1_exit[2]], [p1_entry[coord_idx], p1_exit[coord_idx]], 
                   'r-', alpha=0.2, linewidth=0.5)
            # Draw partial segment after lens1
            t_partial = 5.0
            p_partial = p1_exit + t_partial * d1_out
            ax.plot([p1_exit[2], p_partial[2]], [p1_exit[coord_idx], p_partial[coord_idx]], 
                   'r-', alpha=0.2, linewidth=0.5)
            continue
        p2_entry, d2_in, p2_exit, d2_out = result2[0], result2[1], result2[2], result2[3]

        # Calculate fiber intersection
        if abs(d2_out[2]) < 1e-9:
            continue
        t = (z_fiber - p2_exit[2]) / d2_out[2]
        if t < 0:
            continue
        p_f = p2_exit + t * d2_out

        # Check acceptance criteria for color
        r = __import__("math").hypot(p_f[0], p_f[1])
        theta = __import__("math").acos(abs(d2_out[2]) / np.linalg.norm(d2_out))
        color = 'g' if (r <= C.FIBER_CORE_DIAM_MM/2.0 and
                        theta <= C.ACCEPTANCE_HALF_RAD) else 'r'

        # Plot each segment with proper direction vectors
        # Segment 1: Origin → Lens1 entry (in air, direction d)
        ax.plot([o[2], p1_entry[2]], [o[coord_idx], p1_entry[coord_idx]], 
               color+'-', alpha=0.5, linewidth=0.5)
        
        # Segment 2: Lens1 entry → Lens1 exit (in glass, direction d1_in)
        ax.plot([p1_entry[2], p1_exit[2]], [p1_entry[coord_idx], p1_exit[coord_idx]], 
               color+'-', alpha=0.5, linewidth=0.5)
        
        # Segment 3: Lens1 exit → Lens2 entry (in air, direction d1_out)
        ax.plot([p1_exit[2], p2_entry[2]], [p1_exit[coord_idx], p2_entry[coord_idx]], 
               color+'-', alpha=0.5, linewidth=0.5)
        
        # Segment 4: Lens2 entry → Lens2 exit (in glass, direction d2_in)
        ax.plot([p2_entry[2], p2_exit[2]], [p2_entry[coord_idx], p2_exit[coord_idx]], 
               color+'-', alpha=0.5, linewidth=0.5)
        
        # Segment 5: Lens2 exit → Fiber (in air, direction d2_out)
        ax.plot([p2_exit[2], p_f[2]], [p2_exit[coord_idx], p_f[coord_idx]], 
               color+'-', alpha=0.5, linewidth=0.5)

    # Draw lenses with actual curved profiles
    # Detect lens type for lens 1
    if lens1_data.get('lens_type') == 'Bi-Convex':
        _draw_biconvex_2d(ax, z_l1, lens1_data, flipped=flipped1, alpha=0.2)
    elif lens1_data.get('lens_type') == 'Aspheric':
        _draw_aspheric_2d(ax, z_l1, lens1_data, flipped=flipped1, alpha=0.2)
    else:
        _draw_planoconvex_2d(ax, z_l1, lens1_data, flipped=flipped1, alpha=0.2)
    
    # Detect lens type for lens 2
    if lens2_data.get('lens_type') == 'Bi-Convex':
        _draw_biconvex_2d(ax, z_l2, lens2_data, flipped=flipped2, alpha=0.2)
    elif lens2_data.get('lens_type') == 'Aspheric':
        _draw_aspheric_2d(ax, z_l2, lens2_data, flipped=flipped2, alpha=0.2)
    else:
        _draw_planoconvex_2d(ax, z_l2, lens2_data, flipped=flipped2, alpha=0.2)
    
    # Draw fiber as vertical line
    fiber_half_dia = C.FIBER_CORE_DIAM_MM / 2.0
    ax.plot([z_fiber, z_fiber], [-fiber_half_dia, fiber_half_dia], 
            'g-', linewidth=2, alpha=0.6)
    ax.plot(z_fiber, -fiber_half_dia, 'go', markersize=4)
    ax.plot(z_fiber, fiber_half_dia, 'go', markersize=4)
    
    # Draw source arc at z=0
    source_radius = C.SOURCE_ARC_DIAM_MM / 2.0
    ax.plot([0, 0], [-source_radius, source_radius], 
            'orange', linewidth=2, alpha=0.8)
    ax.plot(0, -source_radius, 'o', color='orange', markersize=4)
    ax.plot(0, source_radius, 'o', color='orange', markersize=4)
    
    # Draw lamp window at z=8.7mm
    lamp_window_radius = C.LAMP_WINDOW_DIAM_MM / 2.0
    z_lamp_window = C.LAMP_WINDOW_DISTANCE_MM
    ax.plot([z_lamp_window, z_lamp_window], [-lamp_window_radius, lamp_window_radius], 
            'yellow', linewidth=2, alpha=0.6, linestyle='--')
    
    # Draw cooling jacket aperture at z=26mm (M23 thread)
    cooling_jacket_radius = C.COOLING_JACKET_THREAD_DIAM_MM / 2.0
    z_cooling_jacket = C.WINDOW_DISTANCE_MM - 1.0
    ax.plot([z_cooling_jacket, z_cooling_jacket], [-cooling_jacket_radius, cooling_jacket_radius], 
            'red', linewidth=3, alpha=0.7)
    ax.plot(z_cooling_jacket, -cooling_jacket_radius, 'rs', markersize=5)
    ax.plot(z_cooling_jacket, cooling_jacket_radius, 'rs', markersize=5)
    
    # Set labels and formatting
    ax.set_xlabel('Z (mm)', fontsize=9)
    ax.set_ylabel(f'{coord_label} (mm)', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')


def plot_system_rays(lenses, best_result, run_id, n_plot_rays=1000, method=None, plot_style='2d'):
    """
    Plot ray trace visualization for a single configuration.
    
    Parameters
    ----------
    lenses : dict
        Dictionary of lens specifications
    best_result : dict
        Result dictionary containing optimization results
    run_id : str
        Run identifier for output directory
    n_plot_rays : int
        Number of rays to plot
    method : str, optional
        Optimization method name for filename
    plot_style : str
        Plot style: '2d', '3d', or 'both'
    """
    orientation = best_result.get('orientation', 'ScffcF')
    
    if plot_style in ['3d', 'both']:
        # Create 3D plot
        fig_3d = plt.figure(figsize=(12, 8))
        ax = fig_3d.add_subplot(111, projection='3d')
        _plot_rays_on_axis(ax, lenses, best_result, n_plot_rays)
        plt.title(f"Ray Trace: {best_result['lens1']} + {best_result['lens2']}, Coupling: {best_result['coupling']:.4f}, {orientation}")
        plt.tight_layout()
        
        # Save 3D plot
        if method:
            plot_dir = f"./plots/{run_id}/{best_result['lens1']}+{best_result['lens2']}"
            if not __import__("os").path.exists(plot_dir):
                __import__("os").makedirs(plot_dir)
            filename_3d = f"{plot_dir}/C-{best_result['coupling']:.4f}_{method}_3d.png"
        else:
            if not __import__("os").path.exists('./plots/' + run_id):
                __import__("os").makedirs('./plots/' + run_id)
            filename_3d = f"./plots/{run_id}/C-{best_result['coupling']:.4f}_L1-{best_result['lens1']}_L2-{best_result['lens2']}_3d.png"
        plt.savefig(filename_3d)
        plt.close(fig_3d)
    
    if plot_style in ['2d', 'both']:
        # Create 2D dual-view plot
        fig_2d = plt.figure(figsize=(12, 10))
        _plot_rays_2d_dual_view(fig_2d, lenses, best_result, n_plot_rays)
        fig_2d.suptitle(f"Ray Trace (2D): {best_result['lens1']} + {best_result['lens2']}, Coupling: {best_result['coupling']:.4f}, {orientation}", 
                       fontsize=13, fontweight='bold', y=0.995)
        plt.tight_layout(rect=(0, 0, 1, 0.99))
        
        # Save 2D plot
        if method:
            plot_dir = f"./plots/{run_id}/{best_result['lens1']}+{best_result['lens2']}"
            if not __import__("os").path.exists(plot_dir):
                __import__("os").makedirs(plot_dir)
            filename_2d = f"{plot_dir}/C-{best_result['coupling']:.4f}_{method}_2d.png"
        else:
            if not __import__("os").path.exists('./plots/' + run_id):
                __import__("os").makedirs('./plots/' + run_id)
            filename_2d = f"./plots/{run_id}/C-{best_result['coupling']:.4f}_L1-{best_result['lens1']}_L2-{best_result['lens2']}_2d.png"
        plt.savefig(filename_2d)
        plt.close(fig_2d)


def plot_dual_orientation_comparison(lenses, result1, result2, run_id, n_plot_rays=1000, plot_style='both'):
    """
    Plot side-by-side comparison of both lens orientations.
    
    Parameters
    ----------
    lenses : dict
        Dictionary of lens specifications
    result1 : dict
        First orientation result (ScffcF)
    result2 : dict
        Second orientation result (SfccfF)
    run_id : str
        Run identifier for output directory
    n_plot_rays : int
        Number of rays to plot
    plot_style : str
        Plot style: '2d', '3d', or 'both'
    """
    # Ensure result1 is ScffcF and result2 is SfccfF for consistency
    if result1.get('orientation') == 'SfccfF':
        result1, result2 = result2, result1
    
    lens1_name = result1['lens1']
    lens2_name = result1['lens2']
    
    # Create plot directory
    plot_dir = f"./plots/{run_id}"
    if not __import__("os").path.exists(plot_dir):
        __import__("os").makedirs(plot_dir)
    
    if plot_style in ['2d', 'both']:
        # Create 2×2 grid for 2D comparison: ScffcF (top row), SfccfF (bottom row)
        fig = plt.figure(figsize=(20, 16))
        
        # ScffcF orientation - X-Z view (top left)
        ax1 = fig.add_subplot(2, 2, 1)
        _plot_single_2d_view(ax1, lenses, result1, n_plot_rays, 'xz')
        ax1.set_title(f"ScffcF (L1 curved-first) - X-Z View\nCoupling: {result1['coupling']:.4f}", 
                     fontsize=11, fontweight='bold')
        
        # ScffcF orientation - Y-Z view (top right)
        ax2 = fig.add_subplot(2, 2, 2)
        _plot_single_2d_view(ax2, lenses, result1, n_plot_rays, 'yz')
        ax2.set_title(f"ScffcF (L1 curved-first) - Y-Z View\nCoupling: {result1['coupling']:.4f}", 
                     fontsize=11, fontweight='bold')
        
        # SfccfF orientation - X-Z view (bottom left)
        ax3 = fig.add_subplot(2, 2, 3)
        _plot_single_2d_view(ax3, lenses, result2, n_plot_rays, 'xz')
        ax3.set_title(f"SfccfF (L1 flat-first) - X-Z View\nCoupling: {result2['coupling']:.4f}", 
                     fontsize=11, fontweight='bold')
        
        # SfccfF orientation - Y-Z view (bottom right)
        ax4 = fig.add_subplot(2, 2, 4)
        _plot_single_2d_view(ax4, lenses, result2, n_plot_rays, 'yz')
        ax4.set_title(f"SfccfF (L1 flat-first) - Y-Z View\nCoupling: {result2['coupling']:.4f}", 
                     fontsize=11, fontweight='bold')
        
        fig.suptitle(f"Orientation Comparison: {lens1_name} + {lens2_name}", 
                    fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout(rect=(0, 0, 1, 0.99))
        
        filename_2d = f"{plot_dir}/BOTH_L1-{lens1_name}_L2-{lens2_name}_2d.png"
        plt.savefig(filename_2d)
        plt.close(fig)
    
    if plot_style in ['3d', 'both']:
        # Create 2×1 grid for 3D comparison
        fig = plt.figure(figsize=(20, 16))
        
        # ScffcF orientation (top)
        ax1 = fig.add_subplot(2, 1, 1, projection='3d')
        _plot_rays_on_axis(ax1, lenses, result1, n_plot_rays)
        ax1.set_title(f"ScffcF (L1 curved-first): Coupling = {result1['coupling']:.4f}", 
                     fontsize=12, fontweight='bold')
        
        # SfccfF orientation (bottom)
        ax2 = fig.add_subplot(2, 1, 2, projection='3d')
        _plot_rays_on_axis(ax2, lenses, result2, n_plot_rays)
        ax2.set_title(f"SfccfF (L1 flat-first): Coupling = {result2['coupling']:.4f}", 
                     fontsize=12, fontweight='bold')
        
        fig.suptitle(f"Orientation Comparison: {lens1_name} + {lens2_name}", 
                    fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout(rect=(0, 0, 1, 0.99))
        
        filename_3d = f"{plot_dir}/BOTH_L1-{lens1_name}_L2-{lens2_name}_3d.png"
        plt.savefig(filename_3d)
        plt.close(fig)



def plot_combined_methods(lenses, results_by_method, lens1, lens2, run_id, n_plot_rays=500, plot_style='2d'):
    """
    Plot comparison grid of multiple optimization methods.
    
    Parameters
    ----------
    lenses : dict
        Dictionary of lens specifications
    results_by_method : dict
        Dictionary mapping method names to result dictionaries
    lens1 : str
        First lens identifier
    lens2 : str
        Second lens identifier
    run_id : str
        Run identifier for output directory
    n_plot_rays : int
        Number of rays to plot per subplot
    plot_style : str
        Plot style: '2d' or '3d' (default: '2d')
    """
    n_methods = len(results_by_method)
    if n_methods == 0:
        return

    if n_methods == 1:
        nrows, ncols = 1, 1
        figsize = (12, 8) if plot_style == '3d' else (12, 10)
    elif n_methods == 2:
        nrows, ncols = 1, 2
        figsize = (20, 8) if plot_style == '3d' else (20, 10)
    elif n_methods == 3:
        nrows, ncols = 1, 3
        figsize = (24, 8) if plot_style == '3d' else (30, 10)
    elif n_methods == 4:
        nrows, ncols = 2, 2
        figsize = (20, 16) if plot_style == '3d' else (20, 20)
    else:
        nrows, ncols = 2, 3
        figsize = (24, 16) if plot_style == '3d' else (30, 20)

    fig = plt.figure(figsize=figsize)

    if plot_style == '3d':
        # 3D subplot grid
        for idx, (method, result) in enumerate(sorted(results_by_method.items()), 1):
            ax = fig.add_subplot(nrows, ncols, idx, projection='3d')
            _plot_rays_on_axis(ax, lenses, result, n_plot_rays)
            
            time_str = f", {result.get('time_seconds', 0):.1f}s" if 'time_seconds' in result else ""
            ax.set_title(f"{method}\nCoupling: {result['coupling']:.4f}{time_str}", fontsize=10)
    else:
        # 2D subplot grid - each method gets a single combined 2D view
        # For simplicity, we'll just show X-Z projection in combined view
        for idx, (method, result) in enumerate(sorted(results_by_method.items()), 1):
            ax = fig.add_subplot(nrows, ncols, idx)
            
            # Create a temporary figure for 2D plotting, extract the X-Z view
            # For combined methods, we'll create simplified single-panel X-Z views
            _plot_single_2d_view(ax, lenses, result, n_plot_rays, 'xz')
            
            time_str = f", {result.get('time_seconds', 0):.1f}s" if 'time_seconds' in result else ""
            ax.set_title(f"{method}\nCoupling: {result['coupling']:.4f}{time_str}", fontsize=10)

    plt.suptitle(f"Method Comparison: {lens1} + {lens2}", fontsize=14, y=0.995)
    plt.tight_layout(rect=(0, 0, 1, 0.99))

    plot_dir = f"./plots/{run_id}/{lens1}+{lens2}"
    if not __import__("os").path.exists(plot_dir):
        __import__("os").makedirs(plot_dir)
    
    suffix = '_2d' if plot_style == '2d' else ''
    filename = f"{plot_dir}/{lens1}+{lens2}{suffix}.png"
    plt.savefig(filename, dpi=150)
    plt.close(fig)


def plot_spot_diagram(best, lenses, run_id):
    accepted_mask = best['accepted']
    origins = best['origins']
    dirs = best['dirs']
    
    # Parse orientation to determine flipped flags
    orientation = best.get('orientation', 'ScffcF')
    if orientation == 'SfccfF':
        flipped1, flipped2 = True, False
    else:  # Default to 'ScffcF'
        flipped1, flipped2 = False, True
    
    land_x = np.full(origins.shape[0], np.nan)
    land_y = np.full(origins.shape[0], np.nan)
    for i in range(origins.shape[0]):
        o = origins[i].copy()
        d = dirs[i].copy()

        # Create lens1 based on its type
        lens1_data = lenses[best['lens1']]
        if lens1_data.get('lens_type') == 'Bi-Convex':
            out1 = BiConvex(vertex_z_front=best['z_l1'],
                           R_front_mm=lens1_data['R1_mm'],
                           R_back_mm=lens1_data.get('R2_mm', lens1_data['R1_mm']),
                           center_thickness_mm=lens1_data['tc_mm'],
                           edge_thickness_mm=lens1_data['te_mm'],
                           ap_rad_mm=lens1_data['dia']/2.0,
                           flipped=flipped1
                           ).trace_ray(o, d, 1.0)
        elif lens1_data.get('lens_type') == 'Aspheric':
            out1 = Aspheric(vertex_z_front=best['z_l1'],
                           R_front_mm=lens1_data['R1_mm'],
                           R_back_mm=lens1_data.get('R2_mm', lens1_data['R1_mm']),
                           center_thickness_mm=lens1_data['tc_mm'],
                           edge_thickness_mm=lens1_data['te_mm'],
                           ap_rad_mm=lens1_data['dia']/2.0,
                           conic_constant=lens1_data.get('conic_constant', 0.0),
                           flipped=flipped1
                           ).trace_ray(o, d, 1.0)
        else:
            out1 = PlanoConvex(vertex_z_front=best['z_l1'],
                              R_front_mm=_get_r_mm(lens1_data),
                              center_thickness_mm=lens1_data['tc_mm'],
                              edge_thickness_mm=lens1_data['te_mm'],
                              ap_rad_mm=lens1_data['dia']/2.0,
                              flipped=flipped1
                              ).trace_ray(o, d, 1.0)
        if out1[2] is False:
            continue
        o1, d1 = out1[0], out1[1]
        
        # Create lens2 based on its type
        lens2_data = lenses[best['lens2']]
        if lens2_data.get('lens_type') == 'Bi-Convex':
            out2 = BiConvex(vertex_z_front=best['z_l2'],
                           R_front_mm=lens2_data['R1_mm'],
                           R_back_mm=lens2_data.get('R2_mm', lens2_data['R1_mm']),
                           center_thickness_mm=lens2_data['tc_mm'],
                           edge_thickness_mm=lens2_data['te_mm'],
                           ap_rad_mm=lens2_data['dia']/2.0,
                           flipped=flipped2
                           ).trace_ray(o1, d1, 1.0)
        elif lens2_data.get('lens_type') == 'Aspheric':
            out2 = Aspheric(vertex_z_front=best['z_l2'],
                           R_front_mm=lens2_data['R1_mm'],
                           R_back_mm=lens2_data.get('R2_mm', lens2_data['R1_mm']),
                           center_thickness_mm=lens2_data['tc_mm'],
                           edge_thickness_mm=lens2_data['te_mm'],
                           ap_rad_mm=lens2_data['dia']/2.0,
                           conic_constant=lens2_data.get('conic_constant', 0.0),
                           flipped=flipped2
                           ).trace_ray(o1, d1, 1.0)
        else:
            out2 = PlanoConvex(vertex_z_front=best['z_l2'],
                              R_front_mm=_get_r_mm(lens2_data),
                              center_thickness_mm=lens2_data['tc_mm'],
                              edge_thickness_mm=lens2_data['te_mm'],
                              ap_rad_mm=lens2_data['dia']/2.0,
                              flipped=flipped2
                              ).trace_ray(o1, d1, 1.0)
        if out2[2] is False:
            continue
        o2, d2 = out2[0], out2[1]
        if abs(d2[2]) < 1e-9:
            continue
        t = (best['z_fiber'] - o2[2]) / d2[2]
        if t < 0:
            continue
        p = o2 + t * d2
        land_x[i] = p[0]
        land_y[i] = p[1]

    plt.figure(figsize=(6, 6))

    plt.scatter(land_x[~accepted_mask], land_y[~accepted_mask],
                s=1, color='red', alpha=0.3, label='rejected')
    plt.scatter(land_x[accepted_mask], land_y[accepted_mask],
                s=1, color='green', alpha=0.6, label='accepted')
    circle = Circle((0, 0), C.FIBER_CORE_DIAM_MM/2.0, color='blue',
                         fill=False, linewidth=1.5, label='fiber core')
    ax = plt.gca()
    ax.add_patch(circle)
    plt.xlabel('x (mm)')
    plt.ylabel('y (mm)')
    orientation = best.get('orientation', 'ScffcF')
    plt.title(f"Spot diagram: {
              best['lens1']} + {best['lens2']} (coupling={best['coupling']:.4f}, {orientation})")
    plt.axis('equal')
    plt.grid(True)
    plt.legend()

    if not __import__("os").path.exists('./plots/' + run_id):
        __import__("os").makedirs('./plots/' + run_id)
    plt.savefig(f"./plots/{run_id}/spot_C-{best['coupling']:.4f}_L1-{
                best['lens1']}_L2-{best['lens2']}.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_wavelength_coupling(lens1, lens2, methods_data, run_id):
    plt.figure(figsize=(10, 6))
    
    for method, (wavelengths, couplings) in sorted(methods_data.items()):
        if method == 'default':
            plt.plot(wavelengths, couplings, 'o-', linewidth=2, markersize=6)
        else:
            plt.plot(wavelengths, couplings, 'o-', linewidth=2, markersize=6, label=method)
    
    plt.xlabel('Wavelength (nm)', fontsize=12)
    plt.ylabel('Coupling Efficiency', fontsize=12)
    plt.title(f'Coupling vs Wavelength: {lens1} + {lens2}', fontsize=14)
    
    if len(methods_data) > 1 or 'default' not in methods_data:
        plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_dir = f"./plots/{run_id}"
    if not __import__("os").path.exists(plot_dir):
        __import__("os").makedirs(plot_dir)
    
    filename = f"{plot_dir}/wavelength_{lens1}+{lens2}.png"
    
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()


def plot_wavelength_per_lens(lens1, lens2, methods_data, plot_dir, fit_type=None):
    import os
    method_colors = {
        'differential_evolution': '#1f77b4',
        'dual_annealing': '#ff7f0e',
        'nelder_mead': '#2ca02c',
        'powell': '#d62728',
        'grid_search': '#9467bd',
        'bayesian': '#8c564b'
    }
    
    markers = ['o', 's', '^', 'D', 'v', '<']
    
    plt.figure(figsize=(10, 6))
    
    for idx, (method, data) in enumerate(sorted(methods_data.items())):
        wavelengths = np.array(data['wavelengths'])
        couplings = np.array(data['couplings'])
        
        sort_idx = np.argsort(wavelengths)
        wavelengths = wavelengths[sort_idx]
        couplings = couplings[sort_idx]
        
        color = method_colors.get(method, f'C{idx}')
        marker = markers[idx % len(markers)]
        
        plt.plot(wavelengths, couplings, marker=marker, linestyle='-', 
                linewidth=2, markersize=8, color=color, label=method, alpha=0.8)
        
        if fit_type and len(wavelengths) > 2:
            if fit_type == 'polynomial':
                degree = min(3, len(wavelengths) - 1)
                coeffs = np.polyfit(wavelengths, couplings, degree)
                poly = np.poly1d(coeffs)
                wl_fit = np.linspace(wavelengths.min(), wavelengths.max(), 100)
                coupling_fit = poly(wl_fit)
                plt.plot(wl_fit, coupling_fit, '--', color=color, 
                        linewidth=1.5, alpha=0.5, label=f'{method} (fitted)')
            
            elif fit_type == 'spline':
                from scipy.interpolate import UnivariateSpline
                if len(wavelengths) >= 4:
                    k = min(3, len(wavelengths) - 1)
                    spl = UnivariateSpline(wavelengths, couplings, k=k, s=0.001)
                    wl_fit = np.linspace(wavelengths.min(), wavelengths.max(), 100)
                    coupling_fit = spl(wl_fit)
                    plt.plot(wl_fit, coupling_fit, '--', color=color, 
                            linewidth=1.5, alpha=0.5, label=f'{method} (fitted)')
    
    plt.xlabel('Wavelength (nm)', fontsize=12)
    plt.ylabel('Coupling Efficiency', fontsize=12)
    plt.title(f'Coupling vs Wavelength: {lens1} + {lens2}', fontsize=14)
    plt.legend(loc='best', fontsize=10, ncol=2 if len(methods_data) > 4 else 1)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # plt.ylim(-0.25, 0.60)  # Fixed y-axis limits for better comparison
    
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    
    fit_suffix = f'_{fit_type}' if fit_type else ''
    filename = os.path.join(plot_dir, f'{lens1}+{lens2}_coupling_vs_wavelength{fit_suffix}.png')
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()


def plot_wavelength_per_method(method, lens_combos_data, plot_dir, fit_type=None):
    import os
    from matplotlib import cm
    
    n_combos = len(lens_combos_data)
    if n_combos <= 10:
        colors = cm.get_cmap('tab10')(np.linspace(0, 1, 10))
    elif n_combos <= 20:
        colors = cm.get_cmap('tab20')(np.linspace(0, 1, 20))
    else:
        colors = cm.get_cmap('hsv')(np.linspace(0, 0.9, n_combos))
    
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    
    plt.figure(figsize=(12, 8))
    
    for idx, (combo_name, data) in enumerate(sorted(lens_combos_data.items())):
        wavelengths = np.array(data['wavelengths'])
        couplings = np.array(data['couplings'])
        
        sort_idx = np.argsort(wavelengths)
        wavelengths = wavelengths[sort_idx]
        couplings = couplings[sort_idx]
        
        color = colors[idx]
        marker = markers[idx % len(markers)]
        
        plt.plot(wavelengths, couplings, marker=marker, linestyle='-', 
                linewidth=2, markersize=7, color=color, label=combo_name, alpha=0.8)
        
        if fit_type and len(wavelengths) > 2:
            if fit_type == 'polynomial':
                degree = min(3, len(wavelengths) - 1)
                coeffs = np.polyfit(wavelengths, couplings, degree)
                poly = np.poly1d(coeffs)
                wl_fit = np.linspace(wavelengths.min(), wavelengths.max(), 100)
                coupling_fit = poly(wl_fit)
                plt.plot(wl_fit, coupling_fit, '--', color=color, 
                        linewidth=1.5, alpha=0.4)
            
            elif fit_type == 'spline':
                from scipy.interpolate import UnivariateSpline
                if len(wavelengths) >= 4:
                    k = min(3, len(wavelengths) - 1)
                    spl = UnivariateSpline(wavelengths, couplings, k=k, s=0.001)
                    wl_fit = np.linspace(wavelengths.min(), wavelengths.max(), 100)
                    coupling_fit = spl(wl_fit)
                    plt.plot(wl_fit, coupling_fit, '--', color=color, 
                            linewidth=1.5, alpha=0.4)
    
    plt.xlabel('Wavelength (nm)', fontsize=12)
    plt.ylabel('Coupling Efficiency', fontsize=12)
    
    method_display = method.replace('_', ' ').title()
    plt.title(f'Coupling vs Wavelength - {method_display} Optimization', fontsize=14)
    
    ncol = max(1, n_combos // 8)
    plt.legend(loc='best', fontsize=9, ncol=ncol, framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # plt.ylim(-0.25, 0.60)  # Fixed y-axis limits for better comparison
    
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    
    fit_suffix = f'_{fit_type}' if fit_type else ''
    filename = os.path.join(plot_dir, f'{method}_coupling_vs_wavelength{fit_suffix}.png')
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()


def plot_wavelength_per_lens_aggregated(lens1, lens2, methods_data, plot_dir, fit_type=None):
    import os
    
    all_wavelengths = set()
    for data in methods_data.values():
        all_wavelengths.update(data['wavelengths'])
    wavelengths = np.array(sorted(all_wavelengths))
    
    mean_couplings = []
    std_couplings = []
    
    for wl in wavelengths:
        couplings_at_wl = []
        for data in methods_data.values():
            wl_arr = np.array(data['wavelengths'])
            mask = np.isclose(wl_arr, wl)
            if np.any(mask):
                idx = np.where(mask)[0][0]
                couplings_at_wl.append(data['couplings'][idx])
        
        if len(couplings_at_wl) > 0:
            mean_couplings.append(np.mean(couplings_at_wl))
            std_couplings.append(np.std(couplings_at_wl))
        else:
            mean_couplings.append(np.nan)
            std_couplings.append(np.nan)
    
    mean_couplings = np.array(mean_couplings)
    std_couplings = np.array(std_couplings)
    
    plt.figure(figsize=(10, 6))
    
    plt.errorbar(wavelengths, mean_couplings, yerr=std_couplings, 
                fmt='o-', linewidth=2, markersize=8, capsize=5, capthick=2,
                color='#1f77b4', ecolor='#1f77b4', alpha=0.8,
                label=f'Mean ± Std ({len(methods_data)} methods)')
    
    if fit_type and len(wavelengths) > 2:
        valid_mask = ~np.isnan(mean_couplings)
        wl_valid = wavelengths[valid_mask]
        coupling_valid = mean_couplings[valid_mask]
        
        if len(wl_valid) > 2:
            if fit_type == 'polynomial':
                degree = min(3, len(wl_valid) - 1)
                coeffs = np.polyfit(wl_valid, coupling_valid, degree)
                poly = np.poly1d(coeffs)
                wl_fit = np.linspace(wl_valid.min(), wl_valid.max(), 100)
                coupling_fit = poly(wl_fit)
                plt.plot(wl_fit, coupling_fit, '--', color='#d62728', 
                        linewidth=2, alpha=0.7, label=f'Polynomial fit (deg {degree})')
            
            elif fit_type == 'spline':
                from scipy.interpolate import UnivariateSpline
                if len(wl_valid) >= 4:
                    k = min(3, len(wl_valid) - 1)
                    spl = UnivariateSpline(wl_valid, coupling_valid, k=k, s=0.001)
                    wl_fit = np.linspace(wl_valid.min(), wl_valid.max(), 100)
                    coupling_fit = spl(wl_fit)
                    plt.plot(wl_fit, coupling_fit, '--', color='#d62728', 
                            linewidth=2, alpha=0.7, label='Spline fit')
    
    plt.xlabel('Wavelength (nm)', fontsize=12)
    plt.ylabel('Coupling Efficiency', fontsize=12)
    plt.title(f'Coupling vs Wavelength (Aggregated): {lens1} + {lens2}', fontsize=14)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # plt.ylim(-0.25, 0.60)  # Fixed y-axis limits for better comparison
    
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    
    fit_suffix = f'_{fit_type}' if fit_type else ''
    filename = os.path.join(plot_dir, f'{lens1}+{lens2}_aggregated{fit_suffix}.png')
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()


def plot_wavelength_per_method_aggregated(method, lens_combos_data, plot_dir, fit_type=None):
    import os
    
    all_wavelengths = set()
    for data in lens_combos_data.values():
        all_wavelengths.update(data['wavelengths'])
    wavelengths = np.array(sorted(all_wavelengths))
    
    mean_couplings = []
    std_couplings = []
    
    for wl in wavelengths:
        couplings_at_wl = []
        for data in lens_combos_data.values():
            wl_arr = np.array(data['wavelengths'])
            mask = np.isclose(wl_arr, wl)
            if np.any(mask):
                idx = np.where(mask)[0][0]
                couplings_at_wl.append(data['couplings'][idx])
        
        if len(couplings_at_wl) > 0:
            mean_couplings.append(np.mean(couplings_at_wl))
            std_couplings.append(np.std(couplings_at_wl))
        else:
            mean_couplings.append(np.nan)
            std_couplings.append(np.nan)
    
    mean_couplings = np.array(mean_couplings)
    std_couplings = np.array(std_couplings)
    
    plt.figure(figsize=(10, 6))
    
    plt.errorbar(wavelengths, mean_couplings, yerr=std_couplings, 
                fmt='o-', linewidth=2, markersize=8, capsize=5, capthick=2,
                color='#2ca02c', ecolor='#2ca02c', alpha=0.8,
                label=f'Mean ± Std ({len(lens_combos_data)} lens combos)')
    
    if fit_type and len(wavelengths) > 2:
        valid_mask = ~np.isnan(mean_couplings)
        wl_valid = wavelengths[valid_mask]
        coupling_valid = mean_couplings[valid_mask]
        
        if len(wl_valid) > 2:
            if fit_type == 'polynomial':
                degree = min(3, len(wl_valid) - 1)
                coeffs = np.polyfit(wl_valid, coupling_valid, degree)
                poly = np.poly1d(coeffs)
                wl_fit = np.linspace(wl_valid.min(), wl_valid.max(), 100)
                coupling_fit = poly(wl_fit)
                plt.plot(wl_fit, coupling_fit, '--', color='#d62728', 
                        linewidth=2, alpha=0.7, label=f'Polynomial fit (deg {degree})')
            
            elif fit_type == 'spline':
                from scipy.interpolate import UnivariateSpline
                if len(wl_valid) >= 4:
                    k = min(3, len(wl_valid) - 1)
                    spl = UnivariateSpline(wl_valid, coupling_valid, k=k, s=0.001)
                    wl_fit = np.linspace(wl_valid.min(), wl_valid.max(), 100)
                    coupling_fit = spl(wl_fit)
                    plt.plot(wl_fit, coupling_fit, '--', color='#d62728', 
                            linewidth=2, alpha=0.7, label='Spline fit')
    
    plt.xlabel('Wavelength (nm)', fontsize=12)
    plt.ylabel('Coupling Efficiency', fontsize=12)
    
    method_display = method.replace('_', ' ').title()
    plt.title(f'Coupling vs Wavelength (Aggregated) - {method_display}', fontsize=14)
    
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # plt.ylim(-0.25, 0.60)  # Fixed y-axis limits for better comparison
    
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    
    fit_suffix = f'_{fit_type}' if fit_type else ''
    filename = os.path.join(plot_dir, f'{method}_aggregated{fit_suffix}.png')
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()


def plot_tolerance_results(results, run_id, output_dir='./plots'):
    """
    Generate plots for tolerance analysis results.
    
    Parameters:
    -----------
    results : dict
        Results dictionary from analyze_tolerance()
    run_id : str
        Identifier for this run
    output_dir : str
        Base directory for plots
    """
    import os
    from pathlib import Path
    
    params = results['parameters']
    lens_pair = f"{params['lens1']}+{params['lens2']}"
    
    # Create output directory
    output_path = Path(output_dir) / run_id
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Extract data
    baseline = results['baseline']
    z_l1_disp = results['z_l1_sensitivity']['displacements']
    z_l1_coup = results['z_l1_sensitivity']['couplings']
    z_l1_metrics = results['z_l1_sensitivity']['metrics']
    
    z_l2_disp = results['z_l2_sensitivity']['displacements']
    z_l2_coup = results['z_l2_sensitivity']['couplings']
    z_l2_metrics = results['z_l2_sensitivity']['metrics']
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot L1 sensitivity
    ax1.plot(z_l1_disp, z_l1_coup, 'o-', linewidth=2, markersize=6, 
             color='#1f77b4', label='Coupling efficiency')
    ax1.axhline(y=baseline, color='green', linestyle='--', linewidth=1.5, 
                label=f'Baseline ({baseline:.4f})')
    ax1.axhline(y=baseline - 0.01, color='red', linestyle=':', linewidth=1.5, 
                label='1% drop threshold')
    ax1.axvline(x=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
    
    # Mark worst displacement
    ax1.plot(z_l1_metrics['worst_displacement'], 
             baseline - z_l1_metrics['max_drop'],
             'rx', markersize=12, markeredgewidth=2, 
             label=f"Worst case (Δz={z_l1_metrics['worst_displacement']:.3f} mm)")
    
    # Mark tolerance range if available
    if z_l1_metrics['tolerance_1pct'] is not None:
        tol = z_l1_metrics['tolerance_1pct']
        ax1.axvspan(-tol, tol, alpha=0.2, color='green', 
                    label=f'1% tolerance (±{tol:.3f} mm)')
    
    ax1.set_xlabel('L1 Displacement (mm)', fontsize=11)
    ax1.set_ylabel('Coupling Efficiency', fontsize=11)
    ax1.set_title(f'L1 Position Sensitivity\n{params["lens1"]}', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best', fontsize=9)
    
    # Plot L2 sensitivity
    ax2.plot(z_l2_disp, z_l2_coup, 'o-', linewidth=2, markersize=6, 
             color='#ff7f0e', label='Coupling efficiency')
    ax2.axhline(y=baseline, color='green', linestyle='--', linewidth=1.5, 
                label=f'Baseline ({baseline:.4f})')
    ax2.axhline(y=baseline - 0.01, color='red', linestyle=':', linewidth=1.5, 
                label='1% drop threshold')
    ax2.axvline(x=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
    
    # Mark worst displacement
    ax2.plot(z_l2_metrics['worst_displacement'], 
             baseline - z_l2_metrics['max_drop'],
             'rx', markersize=12, markeredgewidth=2, 
             label=f"Worst case (Δz={z_l2_metrics['worst_displacement']:.3f} mm)")
    
    # Mark tolerance range if available
    if z_l2_metrics['tolerance_1pct'] is not None:
        tol = z_l2_metrics['tolerance_1pct']
        ax2.axvspan(-tol, tol, alpha=0.2, color='green', 
                    label=f'1% tolerance (±{tol:.3f} mm)')
    
    ax2.set_xlabel('L2 Displacement (mm)', fontsize=11)
    ax2.set_ylabel('Coupling Efficiency', fontsize=11)
    ax2.set_title(f'L2 Position Sensitivity\n{params["lens2"]}', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best', fontsize=9)
    
    # Add overall title
    fig.suptitle(f'Tolerance Analysis: {lens_pair} ({params["orientation"]})', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    
    # Save plot
    filename = output_path / f"tolerance_{lens_pair}.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Saved tolerance plot: {filename}")
    plt.close(fig)


def plot_tolerance_comparison(summary_df, run_id, output_dir='./plots'):
    """
    Generate comparison plots for batch tolerance analysis results.
    
    This creates bar charts comparing tolerance metrics across multiple
    lens pairs to identify which configurations are most/least sensitive
    to misalignment.
    
    Parameters:
    -----------
    summary_df : pandas.DataFrame
        Summary DataFrame from run_tolerance_batch()
    run_id : str
        Identifier for this run
    output_dir : str
        Base directory for plots
    """
    import os
    from pathlib import Path
    import numpy as np
    
    if summary_df.empty:
        print("Warning: Empty summary DataFrame, skipping comparison plots")
        return
    
    # Create output directory
    output_path = Path(output_dir) / run_id
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Calculate minimum tolerance for each pair (worst-case between L1 and L2)
    summary_df = summary_df.copy()
    summary_df['min_tolerance_mm'] = summary_df[['L1_tolerance_1pct_mm', 'L2_tolerance_1pct_mm']].min(axis=1)
    
    # Filter out rows where tolerance couldn't be calculated
    valid_df = summary_df[summary_df['min_tolerance_mm'].notna()].copy()
    
    if len(valid_df) == 0:
        print("Warning: No valid tolerance values to plot")
        return
    
    # Sort by min_tolerance descending (most tolerant first)
    valid_df = valid_df.sort_values('min_tolerance_mm', ascending=False)
    
    # Limit to top 20 pairs if there are many
    n_pairs = len(valid_df)
    n_total = len(summary_df)
    if n_pairs > 20:
        print(f"Note: Showing top 20 most tolerant configurations (out of {n_pairs} with valid tolerances, {n_total} total)")
        valid_df = valid_df.head(20)
    
    lens_pairs = valid_df['lens_pair'].values
    l1_tol = valid_df['L1_tolerance_1pct_mm'].values
    l2_tol = valid_df['L2_tolerance_1pct_mm'].values
    min_tol = valid_df['min_tolerance_mm'].values
    baseline_coupling = valid_df['baseline_coupling'].values
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Tolerance comparison bar chart
    x = np.arange(len(lens_pairs))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, l1_tol, width, label='L1 Tolerance', 
                    color='#1f77b4', alpha=0.8)
    bars2 = ax1.bar(x + width/2, l2_tol, width, label='L2 Tolerance', 
                    color='#ff7f0e', alpha=0.8)
    
    # Add horizontal line at mean tolerance
    mean_min_tol = min_tol.mean()
    ax1.axhline(y=mean_min_tol, color='red', linestyle='--', linewidth=2, 
                label=f'Mean min tolerance ({mean_min_tol:.3f} mm)')
    
    ax1.set_xlabel('Lens Pair', fontsize=11)
    ax1.set_ylabel('1% Drop Tolerance (mm)', fontsize=11)
    ax1.set_title('Tolerance Comparison: Position Sensitivity', 
                  fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(lens_pairs, rotation=45, ha='right', fontsize=9)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Highlight the best (most tolerant) configuration
    if len(x) > 0 and len(ax1.patches) >= 2:
        # First bar of L1 (left) and first bar of L2 (right)
        ax1.patches[0].set_edgecolor('green')
        ax1.patches[0].set_linewidth(2.5)
        if len(ax1.patches) > len(x):
            ax1.patches[len(x)].set_edgecolor('green')
            ax1.patches[len(x)].set_linewidth(2.5)
    
    # Plot 2: Coupling vs Min Tolerance scatter
    colors = ['#2ca02c' if tol >= mean_min_tol else '#d62728' for tol in min_tol]
    scatter = ax2.scatter(baseline_coupling, min_tol, c=colors, s=150, alpha=0.7, 
                         edgecolors='black', linewidth=1.5)
    
    # Add lens pair labels to points
    for i, (coupling, tol, pair) in enumerate(zip(baseline_coupling, min_tol, lens_pairs)):
        # Only label every other point if too many
        if n_pairs <= 10 or i % 2 == 0:
            ax2.annotate(pair, (coupling, tol), fontsize=7, 
                        xytext=(5, 5), textcoords='offset points', alpha=0.8)
    
    # Add reference lines
    ax2.axhline(y=mean_min_tol, color='gray', linestyle='--', linewidth=1.5, 
                label=f'Mean tolerance ({mean_min_tol:.3f} mm)')
    
    ax2.set_xlabel('Baseline Coupling Efficiency', fontsize=11)
    ax2.set_ylabel('Min 1% Tolerance (mm)', fontsize=11)
    ax2.set_title('Coupling Efficiency vs. Tolerance Trade-off', 
                  fontsize=12, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Add text box with statistics
    stats_text = f'Best: {lens_pairs[0]} (±{min_tol[0]:.3f} mm)\n'
    stats_text += f'Worst: {lens_pairs[-1]} (±{min_tol[-1]:.3f} mm)\n'
    stats_text += f'Mean: ±{mean_min_tol:.3f} mm\n'
    stats_text += f'Std: ±{min_tol.std():.3f} mm'
    
    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, 
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Add overall title
    fig.suptitle(f'Batch Tolerance Analysis Comparison\n{len(lens_pairs)} Lens Pairs', 
                 fontsize=14, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=(0, 0, 1, 0.985))
    
    # Save plot
    filename = output_path / "tolerance_batch_comparison.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Saved tolerance comparison plot: {filename}")
    plt.close(fig)
