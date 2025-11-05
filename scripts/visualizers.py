import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Patch
from matplotlib.lines import Line2D

from scripts.PlanoConvex import PlanoConvex
from scripts.raytrace_helpers import sample_rays
from scripts import consts as C


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
    R = lens_data['R_mm']
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
        # Normal orientation: curved front, flat back
        # Curved surface equation: z = z_pos + R - sqrt(R^2 - y^2)
        z_curve = z_pos + R - np.sqrt(np.maximum(R**2 - y_curve**2, 0))
        z_flat = z_pos + tc
        
        # Build lens outline: curved front (bottom to top) + flat back (top to bottom, reversed) + close
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
    R = lens_data['R_mm']
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

    lens1 = PlanoConvex(z_l1, lens1_data['R_mm'], lens1_data['tc_mm'],
                        lens1_data['te_mm'], lens1_data['dia']/2.0, flipped=flipped1)
    lens2 = PlanoConvex(z_l2, lens2_data['R_mm'], lens2_data['tc_mm'],
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
    _draw_planoconvex_3d(ax, z_l1, lens1_data, flipped=flipped1, alpha=0.2)
    _draw_planoconvex_3d(ax, z_l2, lens2_data, flipped=flipped2, alpha=0.2)

    # Draw fiber face
    theta = np.linspace(0, 2*np.pi, 100)
    r = np.linspace(0, C.FIBER_CORE_DIAM_MM/2.0, 2)
    t, r = np.meshgrid(theta, r)
    x = r * np.cos(t)
    y = r * np.sin(t)
    ax.plot_surface(x, y, z_fiber + np.zeros_like(x), alpha=0.3, color='g')

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

    lens1 = PlanoConvex(z_l1, lens1_data['R_mm'], lens1_data['tc_mm'],
                        lens1_data['te_mm'], lens1_data['dia']/2.0, flipped=flipped1)
    lens2 = PlanoConvex(z_l2, lens2_data['R_mm'], lens2_data['tc_mm'],
                        lens2_data['te_mm'], lens2_data['dia']/2.0, flipped=flipped2)

    # Create two subplots for X-Z and Y-Z views
    ax1 = fig.add_subplot(2, 1, 1)  # X-Z view (top)
    ax2 = fig.add_subplot(2, 1, 2)  # Y-Z view (bottom)
    
    # Trace rays and collect data for both projections
    for i in range(n_plot_rays):
        points_x = []  # X coordinates
        points_y = []  # Y coordinates
        points_z = []  # Z coordinates
        
        o = origins[i].copy()
        d = dirs[i].copy()
        points_x.append(o[0])
        points_y.append(o[1])
        points_z.append(o[2])

        out1 = lens1.trace_ray(o, d, 1.0)
        if out1[2] is False:
            # Ray rejected at L1
            ax1.plot(points_z, points_x, 'r-', alpha=0.2, linewidth=0.5)
            ax2.plot(points_z, points_y, 'r-', alpha=0.2, linewidth=0.5)
            continue
        o1, d1 = out1[0], out1[1]
        points_x.append(o1[0])
        points_y.append(o1[1])
        points_z.append(o1[2])

        out2 = lens2.trace_ray(o1, d1, 1.0)
        if out2[2] is False:
            # Ray rejected at L2
            ax1.plot(points_z, points_x, 'r-', alpha=0.2, linewidth=0.5)
            ax2.plot(points_z, points_y, 'r-', alpha=0.2, linewidth=0.5)
            continue
        o2, d2 = out2[0], out2[1]
        points_x.append(o2[0])
        points_y.append(o2[1])
        points_z.append(o2[2])

        if abs(d2[2]) < 1e-9:
            continue
        t = (z_fiber - o2[2]) / d2[2]
        if t < 0:
            continue
        p_f = o2 + t * d2
        points_x.append(p_f[0])
        points_y.append(p_f[1])
        points_z.append(p_f[2])

        # Check acceptance criteria
        r = __import__("math").hypot(p_f[0], p_f[1])
        theta = __import__("math").acos(abs(d2[2]) / np.linalg.norm(d2))
        color = 'g' if (r <= C.FIBER_CORE_DIAM_MM/2.0 and
                        theta <= C.ACCEPTANCE_HALF_RAD) else 'r'

        # Plot in both views
        ax1.plot(points_z, points_x, color+'-', alpha=0.5, linewidth=0.5)
        ax2.plot(points_z, points_y, color+'-', alpha=0.5, linewidth=0.5)

    # Draw lenses with actual curved profiles
    _draw_planoconvex_2d(ax1, z_l1, lens1_data, flipped=flipped1, alpha=0.2)
    _draw_planoconvex_2d(ax2, z_l1, lens1_data, flipped=flipped1, alpha=0.2)
    
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
        Line2D([0], [0], color='g', linewidth=3, alpha=0.6, label='Fiber core')
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

    lens1 = PlanoConvex(z_l1, lens1_data['R_mm'], lens1_data['tc_mm'],
                        lens1_data['te_mm'], lens1_data['dia']/2.0, flipped=flipped1)
    lens2 = PlanoConvex(z_l2, lens2_data['R_mm'], lens2_data['tc_mm'],
                        lens2_data['te_mm'], lens2_data['dia']/2.0, flipped=flipped2)

    coord_idx = 0 if projection == 'xz' else 1  # X=0, Y=1
    coord_label = 'X' if projection == 'xz' else 'Y'
    
    # Trace rays and collect data
    for i in range(n_plot_rays):
        points_coord = []  # X or Y coordinates
        points_z = []      # Z coordinates
        
        o = origins[i].copy()
        d = dirs[i].copy()
        points_coord.append(o[coord_idx])
        points_z.append(o[2])

        out1 = lens1.trace_ray(o, d, 1.0)
        if out1[2] is False:
            ax.plot(points_z, points_coord, 'r-', alpha=0.2, linewidth=0.5)
            continue
        o1, d1 = out1[0], out1[1]
        points_coord.append(o1[coord_idx])
        points_z.append(o1[2])

        out2 = lens2.trace_ray(o1, d1, 1.0)
        if out2[2] is False:
            ax.plot(points_z, points_coord, 'r-', alpha=0.2, linewidth=0.5)
            continue
        o2, d2 = out2[0], out2[1]
        points_coord.append(o2[coord_idx])
        points_z.append(o2[2])

        if abs(d2[2]) < 1e-9:
            continue
        t = (z_fiber - o2[2]) / d2[2]
        if t < 0:
            continue
        p_f = o2 + t * d2
        points_coord.append(p_f[coord_idx])
        points_z.append(p_f[2])

        # Check acceptance criteria
        r = __import__("math").hypot(p_f[0], p_f[1])
        theta = __import__("math").acos(abs(d2[2]) / np.linalg.norm(d2))
        color = 'g' if (r <= C.FIBER_CORE_DIAM_MM/2.0 and
                        theta <= C.ACCEPTANCE_HALF_RAD) else 'r'

        ax.plot(points_z, points_coord, color+'-', alpha=0.5, linewidth=0.5)

    # Draw lenses with actual curved profiles
    _draw_planoconvex_2d(ax, z_l1, lens1_data, flipped=flipped1, alpha=0.2)
    _draw_planoconvex_2d(ax, z_l2, lens2_data, flipped=flipped2, alpha=0.2)
    
    # Draw fiber as vertical line
    fiber_half_dia = C.FIBER_CORE_DIAM_MM / 2.0
    ax.plot([z_fiber, z_fiber], [-fiber_half_dia, fiber_half_dia], 
            'g-', linewidth=2, alpha=0.6)
    ax.plot(z_fiber, -fiber_half_dia, 'go', markersize=4)
    ax.plot(z_fiber, fiber_half_dia, 'go', markersize=4)
    
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
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        
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
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        
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
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        
        filename_3d = f"{plot_dir}/BOTH_L1-{lens1_name}_L2-{lens2_name}_3d.png"
        plt.savefig(filename_3d)
        plt.close(fig)


def _plot_single_2d_view(ax, lenses, result, n_plot_rays, view='xz'):
    """
    Helper function to plot a single 2D view (either X-Z or Y-Z projection).
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to plot on
    lenses : dict
        Dictionary of lens specifications
    result : dict
        Result dictionary containing z_l1, z_l2, z_fiber, lens1, lens2, orientation
    n_plot_rays : int
        Number of rays to plot
    view : str
        'xz' for X-Z projection or 'yz' for Y-Z projection
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

    lens1 = PlanoConvex(z_l1, lens1_data['R_mm'], lens1_data['tc_mm'],
                        lens1_data['te_mm'], lens1_data['dia']/2.0, flipped=flipped1)
    lens2 = PlanoConvex(z_l2, lens2_data['R_mm'], lens2_data['tc_mm'],
                        lens2_data['te_mm'], lens2_data['dia']/2.0, flipped=flipped2)

    # Trace rays and collect data
    for i in range(n_plot_rays):
        points_perp = []  # Perpendicular coordinate (X or Y)
        points_z = []     # Z coordinates
        
        o = origins[i].copy()
        d = dirs[i].copy()
        points_perp.append(o[0] if view == 'xz' else o[1])
        points_z.append(o[2])

        out1 = lens1.trace_ray(o, d, 1.0)
        if out1[2] is False:
            # Ray rejected at L1
            ax.plot(points_z, points_perp, 'r-', alpha=0.2, linewidth=0.5)
            continue
        o1, d1 = out1[0], out1[1]
        points_perp.append(o1[0] if view == 'xz' else o1[1])
        points_z.append(o1[2])

        out2 = lens2.trace_ray(o1, d1, 1.0)
        if out2[2] is False:
            # Ray rejected at L2
            ax.plot(points_z, points_perp, 'r-', alpha=0.2, linewidth=0.5)
            continue
        o2, d2 = out2[0], out2[1]
        points_perp.append(o2[0] if view == 'xz' else o2[1])
        points_z.append(o2[2])

        if abs(d2[2]) < 1e-9:
            continue
        t = (z_fiber - o2[2]) / d2[2]
        if t < 0:
            continue
        p_f = o2 + t * d2
        points_perp.append(p_f[0] if view == 'xz' else p_f[1])
        points_z.append(p_f[2])

        # Check acceptance criteria
        r = __import__("math").hypot(p_f[0], p_f[1])
        theta = __import__("math").acos(abs(d2[2]) / np.linalg.norm(d2))
        color = 'g' if (r <= C.FIBER_CORE_DIAM_MM/2.0 and
                        theta <= C.ACCEPTANCE_HALF_RAD) else 'r'

        ax.plot(points_z, points_perp, color+'-', alpha=0.5, linewidth=0.5)

    # Draw lenses with actual curved profiles
    _draw_planoconvex_2d(ax, z_l1, lens1_data, flipped=flipped1, alpha=0.2)
    _draw_planoconvex_2d(ax, z_l2, lens2_data, flipped=flipped2, alpha=0.2)
    
    # Draw fiber as vertical line
    fiber_half_height = C.FIBER_CORE_DIAM_MM / 2.0
    ax.axvline(x=z_fiber, color='orange', linestyle='--', linewidth=2, label='Fiber')
    ax.plot([z_fiber, z_fiber], [-fiber_half_height, fiber_half_height], 
            'orange', linewidth=3, label='Fiber Core')
    
    # Formatting
    ax.set_xlabel('Z (mm)', fontsize=10)
    ax.set_ylabel('X (mm)' if view == 'xz' else 'Y (mm)', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='datalim')
    ax.legend(loc='upper right', fontsize=8)


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
            _plot_single_2d_view(ax, lenses, result, n_plot_rays, projection='xz')
            
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

        out1 = PlanoConvex(vertex_z_front=best['z_l1'],
                           R_front_mm=lenses[best['lens1']]['R_mm'],
                           center_thickness_mm=lenses[best['lens1']]['tc_mm'],
                           edge_thickness_mm=lenses[best['lens1']]['te_mm'],
                           ap_rad_mm=lenses[best['lens1']]['dia']/2.0,
                           flipped=flipped1
                           ).trace_ray(o, d, 1.0)
        if out1[2] is False:
            continue
        o1, d1 = out1[0], out1[1]
        out2 = PlanoConvex(vertex_z_front=best['z_l2'],
                           R_front_mm=lenses[best['lens2']]['R_mm'],
                           center_thickness_mm=lenses[best['lens2']]['tc_mm'],
                           edge_thickness_mm=lenses[best['lens2']]['te_mm'],
                           ap_rad_mm=lenses[best['lens2']]['dia']/2.0,
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
    
    plt.ylim(-0.25, 0.40)  # Fixed y-axis limits for better comparison
    
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
    
    plt.ylim(-0.25, 0.40)  # Fixed y-axis limits for better comparison
    
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
    
    plt.ylim(-0.25, 0.40)  # Fixed y-axis limits for better comparison
    
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
    
    plt.ylim(-0.25, 0.40)  # Fixed y-axis limits for better comparison
    
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
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save plot
    filename = output_path / f"tolerance_{lens_pair}.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Saved tolerance plot: {filename}")
    plt.close(fig)
