import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from scripts.PlanoConvex import PlanoConvex
from scripts.raytrace_helpers import sample_rays
from scripts import consts as C


def _plot_rays_on_axis(ax, lenses, result, n_plot_rays=1000):
    z_l1 = result['z_l1']
    z_l2 = result['z_l2']
    z_fiber = result['z_fiber']
    lens1_data = lenses[result['lens1']]
    lens2_data = lenses[result['lens2']]

    origins, dirs = sample_rays(n_plot_rays)

    lens1 = PlanoConvex(z_l1, lens1_data['R_mm'], lens1_data['tc_mm'],
                        lens1_data['te_mm'], lens1_data['dia']/2.0)
    lens2 = PlanoConvex(z_l2, lens2_data['R_mm'], lens2_data['tc_mm'],
                        lens2_data['te_mm'], lens2_data['dia']/2.0)

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

    theta = np.linspace(0, 2*np.pi, 100)

    r = np.linspace(0, lens1_data['dia']/2.0, 2)
    t, r = np.meshgrid(theta, r)
    x = r * np.cos(t)
    y = r * np.sin(t)
    ax.plot_surface(x, y, z_l1 + np.zeros_like(x), alpha=0.2, color='b')
    ax.plot_surface(x, y, z_l1 + lens1_data['tc_mm'] + np.zeros_like(x),
                    alpha=0.2, color='b')

    r = np.linspace(0, lens2_data['dia']/2.0, 2)
    t, r = np.meshgrid(theta, r)
    x = r * np.cos(t)
    y = r * np.sin(t)
    ax.plot_surface(x, y, z_l2 + np.zeros_like(x), alpha=0.2, color='b')
    ax.plot_surface(x, y, z_l2 + lens2_data['tc_mm'] + np.zeros_like(x),
                    alpha=0.2, color='b')

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


def plot_system_rays(lenses, best_result, run_id, n_plot_rays=1000, method=None):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    _plot_rays_on_axis(ax, lenses, best_result, n_plot_rays)

    plt.title(f"Ray Trace: {
              best_result['lens1']} + {best_result['lens2']}, Coupling: {best_result['coupling']:.4f}")

    plt.tight_layout()

    # Save plot
    if method:
        plot_dir = f"./plots/{run_id}/{best_result['lens1']}+{best_result['lens2']}"
        if not __import__("os").path.exists(plot_dir):
            __import__("os").makedirs(plot_dir)
        filename = f"{plot_dir}/C-{best_result['coupling']:.4f}_{method}.png"
    else:
        if not __import__("os").path.exists('./plots/' + run_id):
            __import__("os").makedirs('./plots/' + run_id)
        filename = f"./plots/{run_id}/C-{best_result['coupling']:.4f}_L1-{best_result['lens1']}_L2-{best_result['lens2']}.png"
    plt.savefig(filename)
    plt.close(fig)


def plot_combined_methods(lenses, results_by_method, lens1, lens2, run_id, n_plot_rays=500):
    n_methods = len(results_by_method)
    if n_methods == 0:
        return

    if n_methods == 1:
        nrows, ncols = 1, 1
        figsize = (12, 8)
    elif n_methods == 2:
        nrows, ncols = 1, 2
        figsize = (20, 8)
    elif n_methods == 3:
        nrows, ncols = 1, 3
        figsize = (24, 8)
    elif n_methods == 4:
        nrows, ncols = 2, 2
        figsize = (20, 16)
    else:
        nrows, ncols = 2, 3
        figsize = (24, 16)

    fig = plt.figure(figsize=figsize)

    for idx, (method, result) in enumerate(sorted(results_by_method.items()), 1):
        ax = fig.add_subplot(nrows, ncols, idx, projection='3d')
        _plot_rays_on_axis(ax, lenses, result, n_plot_rays)
        
        time_str = f", {result.get('time_seconds', 0):.1f}s" if 'time_seconds' in result else ""
        ax.set_title(f"{method}\nCoupling: {result['coupling']:.4f}{time_str}", fontsize=10)

    plt.suptitle(f"Method Comparison: {lens1} + {lens2}", fontsize=14, y=0.995)
    plt.tight_layout(rect=(0, 0, 1, 0.99))

    plot_dir = f"./plots/{run_id}/{lens1}+{lens2}"
    if not __import__("os").path.exists(plot_dir):
        __import__("os").makedirs(plot_dir)
    filename = f"{plot_dir}/{lens1}+{lens2}.png"
    plt.savefig(filename, dpi=150)
    plt.close(fig)


def plot_spot_diagram(best, lenses, run_id):
    accepted_mask = best['accepted']
    origins = best['origins']
    dirs = best['dirs']
    land_x = np.full(origins.shape[0], np.nan)
    land_y = np.full(origins.shape[0], np.nan)
    for i in range(origins.shape[0]):
        o = origins[i].copy()
        d = dirs[i].copy()

        out1 = PlanoConvex(vertex_z_front=best['z_l1'],
                           R_front_mm=lenses[best['lens1']]['R_mm'],
                           center_thickness_mm=lenses[best['lens1']]['tc_mm'],
                           edge_thickness_mm=lenses[best['lens1']]['te_mm'],
                           ap_rad_mm=lenses[best['lens1']]['dia']/2.0
                           ).trace_ray(o, d, 1.0)
        if out1[2] is False:
            continue
        o1, d1 = out1[0], out1[1]
        out2 = PlanoConvex(vertex_z_front=best['z_l2'],
                           R_front_mm=lenses[best['lens2']]['R_mm'],
                           center_thickness_mm=lenses[best['lens2']]['tc_mm'],
                           edge_thickness_mm=lenses[best['lens2']]['te_mm'],
                           ap_rad_mm=lenses[best['lens2']]['dia']/2.0
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
    plt.title(f"Spot diagram: {
              best['lens1']} + {best['lens2']} (coupling={best['coupling']:.4f})")
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
