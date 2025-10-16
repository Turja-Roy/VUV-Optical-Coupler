import numpy as np
import matplotlib.pyplot as plt

from scripts.PlanoConvex import PlanoConvex
from scripts.raytrace_helpers import sample_rays
from scripts import consts as C


def plot_system_rays(lenses, best_result, run_id, n_plot_rays=1000):
    """
    Plot ray tracing through the optical system.
    """
    # Create figure
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Get system parameters
    z_l1 = best_result['z_l1']
    z_l2 = best_result['z_l2']
    z_fiber = best_result['z_fiber']
    lens1_data = lenses[best_result['lens1']]
    lens2_data = lenses[best_result['lens2']]

    # Create new rays for visualization
    origins, dirs = sample_rays(n_plot_rays)

    # Create lens instances
    lens1 = PlanoConvex(z_l1, lens1_data['R_mm'], lens1_data['tc_mm'],
                        lens1_data['te_mm'], lens1_data['dia']/2.0)
    lens2 = PlanoConvex(z_l2, lens2_data['R_mm'], lens2_data['tc_mm'],
                        lens2_data['te_mm'], lens2_data['dia']/2.0)

    # Plot ray paths
    for i in range(n_plot_rays):
        points = []  # Will store all points along ray path
        o = origins[i].copy()
        d = dirs[i].copy()
        points.append(o)

        # Through first lens
        out1 = lens1.trace_ray(o, d, 1.0)
        if out1[2] is False:
            # Plot failed ray in red
            points = np.array(points)
            ax.plot(points[:, 0], points[:, 1], points[:, 2], 'r-', alpha=0.2)
            continue
        o1, d1 = out1[0], out1[1]
        points.append(o1)

        # Through second lens
        out2 = lens2.trace_ray(o1, d1, 1.0)
        if out2[2] is False:
            # Plot failed ray in red
            points = np.array(points)
            ax.plot(points[:, 0], points[:, 1], points[:, 2], 'r-', alpha=0.2)
            continue
        o2, d2 = out2[0], out2[1]
        points.append(o2)

        # To fiber
        if abs(d2[2]) < 1e-9:
            continue
        t = (z_fiber - o2[2]) / d2[2]
        if t < 0:
            continue
        p_f = o2 + t * d2
        points.append(p_f)

        # Check if accepted
        r = __import__("math").hypot(p_f[0], p_f[1])
        theta = __import__("math").acos(abs(d2[2]) / np.linalg.norm(d2))
        color = 'g' if (r <= C.FIBER_CORE_DIAM_MM/2.0 and
                        theta <= C.ACCEPTANCE_HALF_RAD) else 'r'

        # Plot complete ray path
        points = np.array(points)
        ax.plot(points[:, 0], points[:, 1], points[:, 2],
                color+'-', alpha=0.5)

    # Plot lens surfaces (simplified as disks)
    theta = np.linspace(0, 2*np.pi, 100)

    # Lens 1 surfaces
    r = np.linspace(0, lens1_data['dia']/2.0, 2)
    t, r = np.meshgrid(theta, r)
    x = r * np.cos(t)
    y = r * np.sin(t)
    ax.plot_surface(x, y, z_l1 + np.zeros_like(x), alpha=0.2, color='b')
    ax.plot_surface(x, y, z_l1 + lens1_data['tc_mm'] + np.zeros_like(x),
                    alpha=0.2, color='b')

    # Lens 2 surfaces
    r = np.linspace(0, lens2_data['dia']/2.0, 2)
    t, r = np.meshgrid(theta, r)
    x = r * np.cos(t)
    y = r * np.sin(t)
    ax.plot_surface(x, y, z_l2 + np.zeros_like(x), alpha=0.2, color='b')
    ax.plot_surface(x, y, z_l2 + lens2_data['tc_mm'] + np.zeros_like(x),
                    alpha=0.2, color='b')

    # Plot fiber face
    r = np.linspace(0, C.FIBER_CORE_DIAM_MM/2.0, 2)
    t, r = np.meshgrid(theta, r)
    x = r * np.cos(t)
    y = r * np.sin(t)
    ax.plot_surface(x, y, z_fiber + np.zeros_like(x), alpha=0.3, color='g')

    # Set equal aspect ratio
    ax.set_box_aspect([1, 1, 1])

    # Labels
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    plt.title(f"Ray Trace: {
              best_result['lens1']} + {best_result['lens2']}, Coupling: {best_result['coupling']:.4f}")

    # View angle
    ax.view_init(elev=20, azim=45)

    plt.tight_layout()

    # Save plot
    if not __import__("os").path.exists('./plots/' + run_id):
        __import__("os").makedirs('./plots/' + run_id)
    plt.savefig(f"./plots/{run_id}/C-{best_result['coupling']:.4f}_L1-{
                best_result['lens1']}_L2-{best_result['lens2']}.png")
    plt.close(fig)


def plot_spot_diagram(best, lenses, run_id):
    """
        Spot diagram for best
        (use the origins/dirs that produced the reported best)
    """
    accepted_mask = best['accepted']
    origins = best['origins']
    dirs = best['dirs']
    # compute landing points
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
    circle = plt.Circle((0, 0), C.FIBER_CORE_DIAM_MM/2.0, color='blue',
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

    # Save spot diagram
    if not __import__("os").path.exists('./plots/' + run_id):
        __import__("os").makedirs('./plots/' + run_id)
    plt.savefig(f"./plots/{run_id}/spot_C-{best['coupling']:.4f}_L1-{
                best['lens1']}_L2-{best['lens2']}.png", dpi=300, bbox_inches='tight')
    plt.close()
