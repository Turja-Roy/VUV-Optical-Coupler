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
