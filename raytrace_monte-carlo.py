import math, time, os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
# %matplotlib widget

# Plot style
plt.rcParams.update({'figure.max_open_warning': 0})


############################
# Constants and Parameters #
############################

WAVELENGTH_NM = 200.0                       # 200 nm

# Fiber
FIBER_CORE_DIAM_MM = 1.0                    # 1000 micron
NA = 0.22
ACCEPTANCE_HALF_RAD = np.deg2rad(12.4)

# Source geometry
SOURCE_ARC_DIAM_MM = 3.0                    # Arc diameter
WINDOW_DIAM_MM = 14.3                       # Window diameter
WINDOW_DISTANCE_MM = 8.7                    # Distance from arc to window
MAX_ANGLE_DEG = 33                          # Maximum ray angle at window edge
SOURCE_RADIUS_MM = SOURCE_ARC_DIAM_MM/2.0   # Source radius for ray generation

# Position offset for lenses
SOURCE_TO_LENS_OFFSET = WINDOW_DISTANCE_MM  # Lenses start after the window

# Rays
N_RAYS = 1000

# Date string
DATE_STR = time.strftime("%Y-%m-%d")


################################
# Calculating refractive index #
################################

def fused_silica_n(lambda_nm):
    """Compute fused silica refractive index (Malitson/Sellmeier) -- lambda in nm."""
    lam_um = lambda_nm / 1000.0
    B1 = 0.6961663
    B2 = 0.4079426
    B3 = 0.8974794
    C1 = 0.0684043**2
    C2 = 0.1162414**2
    C3 = 9.896161**2
    lam2 = lam_um*lam_um
    n2 = 1 + B1*lam2/(lam2 - C1) + B2*lam2/(lam2 - C2) + B3*lam2/(lam2 - C3)
    return math.sqrt(n2)

n_glass = fused_silica_n(WAVELENGTH_NM)
print(f"fused silica n({WAVELENGTH_NM} nm) = {n_glass:.6f}")


#################
# Fetching Data #
#################

# l1_candidates = pd.read_csv('./data/l1_candidates.csv')
# l2_candidates = pd.read_csv('./data/l2_candidates.csv')
lenses_candidates = pd.read_csv('./data/ThorLabs_Lenses.csv')

lens1, lens2, lenses = {}, {}, {}

# for _, row in l1_candidates.iterrows():
#     lens1[row['Item #']] = {'dia': row['Diameter (mm)'], 'f_mm': row['Focal Length (mm)'],
#                              'R_mm': row['Radius of Curvature (mm)'], 't_mm': row['Center Thickness (mm)'],
#                              'BFL_mm': row['Back Focal Length (mm)']}
# for _, row in l2_candidates.iterrows():
#     lens2[row['Item #']] = {'dia': row['Diameter (mm)'], 'f_mm': row['Focal Length (mm)'],
#                              'R_mm': row['Radius of Curvature (mm)'], 't_mm': row['Center Thickness (mm)'],
#                              'BFL_mm': row['Back Focal Length (mm)']}
for _, row in lenses_candidates.iterrows():
    lenses[row['Item #']] = {'dia': row['Diameter (mm)'], 'f_mm': row['Focal Length (mm)'],
                             'R_mm': row['Radius of Curvature (mm)'], 't_mm': row['Center Thickness (mm)'],
                             'BFL_mm': row['Back Focal Length (mm)']}

# lenses = lens1 | lens2

# pd.DataFrame(lenses).T.style


################################
# Ray-Tracing Helper Functions #
################################

def sample_rays(n_rays):
    arc_radius = SOURCE_ARC_DIAM_MM / 2.0
    
    r = np.sqrt(np.random.rand(n_rays)) * arc_radius  # radial positions
    phi = np.linspace(0, 2*np.pi, n_rays)  # angular positions around circle
    
    # Source points
    x_source = r * np.cos(phi)
    y_source = r * np.sin(phi)
    origins = np.vstack([x_source, y_source, np.zeros_like(x_source)]).T

    # Calculate coherent ray angles based on radial position
    # Angle increases linearly with radius (0 at center, max_angle_deg at edge)
    ray_angles = np.deg2rad(MAX_ANGLE_DEG * r / arc_radius)
    
    # Calculate ray directions in cylindrical coordinates
    # phi is same as source point (coherent beam)
    # theta=ray_angles is the angle from z-axis, varying with radius
    x_dir = np.sin(ray_angles) * np.cos(phi)
    y_dir = np.sin(ray_angles) * np.sin(phi)
    z_dir = np.cos(ray_angles)
    
    # Stack directions and normalize
    directions = np.vstack([x_dir, y_dir, z_dir]).T
    directions = directions / np.linalg.norm(directions, axis=1)[:, np.newaxis]

    return origins, directions


def intersect_ray_sphere(o, d, c, R):
    oc = o - c
    b = 2 * np.dot(oc, d)
    c0 = np.dot(oc,oc) - R*R
    a = np.dot(d,d)
    disc = b*b - 4*a*c0
    if disc < 0:
        return None
    sqrt_d = math.sqrt(disc)
    t1 = (-b - sqrt_d) / (2*a)
    t2 = (-b + sqrt_d) / (2*a)
    ts = [t for t in (t1,t2) if t>1e-9]
    if not ts:
        return None
    return min(ts)


def refract_vec(n_vec, v_in, n1, n2):
    n_vec = np.array(n_vec); v_in = np.array(v_in)
    n_vec = n_vec / np.linalg.norm(n_vec); v = v_in / np.linalg.norm(v_in)
    cos_i = -np.dot(n_vec, v)
    eta = n1 / n2
    k = 1 - eta*eta * (1 - cos_i*cos_i)
    if k < 0:
        return None, True
    v_out = eta * v + (eta * cos_i - math.sqrt(k)) * n_vec
    v_out = v_out / np.linalg.norm(v_out)
    return v_out, False


############################
# Define PlanoConvex class #
############################

class PlanoConvex:
    """
    A plano-convex lens with spherical front surface.
    Front surface is convex (center of curvature on +z side).
    Back surface is planar.
    """
    def __init__(self, vertex_z_front, R_front_mm, thickness_mm, ap_rad_mm, n_glass):
        """Initialize lens with its parameters."""
        self.vertex_z_front = vertex_z_front
        self.R_front_mm = R_front_mm
        self.thickness_mm = thickness_mm
        self.ap_rad_mm = ap_rad_mm
        self.n_glass = n_glass
        # Derived quantities
        self.vertex_z_back = vertex_z_front + thickness_mm
        self.center_z_front = vertex_z_front + R_front_mm
    
    def trace_ray(self, o, d, n1):
        """
        Trace a ray through the lens.
        
        Parameters:
        - o: 3D origin point
        - d: 3D direction vector (normalized)
        - n1: input refractive index
        
        Returns:
        - (o_out, d_out, success)
        """
        # Front surface (sphere)
        c = np.array([0, 0, self.center_z_front])  # center
        t = intersect_ray_sphere(o, d, c, self.R_front_mm)
        if t is None: return None, None, False
        p = o + t*d  # intersection point
        
        # Check aperture
        if math.hypot(p[0], p[1]) > self.ap_rad_mm:
            return None, None, False
        
        # Surface normal (points out of glass)
        n = (p - c) / self.R_front_mm
        
        # Refract into glass
        d_in, TIR = refract_vec(n, d, n1, self.n_glass)
        if TIR: return None, None, False
        
        # Go to back surface (planar)
        o_back = p + (self.thickness_mm/abs(d_in[2])) * d_in
        
        # Check aperture at back
        if math.hypot(o_back[0], o_back[1]) > self.ap_rad_mm:
            return None, None, False
        
        # Refract out of glass (planar surface, normal = -z)
        d_out, TIR = refract_vec(np.array([0,0,-1]), d_in, self.n_glass, n1)
        if TIR: return None, None, False
        
        return o_back, d_out, True


################################################
# Trace rays through system cand check success #
################################################

def trace_system(origins, dirs, lens1, lens2, z_fiber, fiber_rad, acceptance_half_rad):
    """
    Trace rays through system and check if they make it into the fiber.
    
    Parameters:
    - origins, dirs: Nx3 arrays of ray origins and directions
    - lens1, lens2: PlanoConvex objects for first and second lens
    - z_fiber: z-position of fiber face
    - fiber_rad: fiber core radius
    - acceptance_half_rad: half-acceptance angle in radians
    
    Returns:
    - accepted: boolean array indicating which rays made it into fiber
    """
    n_rays = origins.shape[0]
    accepted = np.zeros(n_rays, dtype=bool)
    
    for i in range(n_rays):
        o = origins[i].copy()
        d = dirs[i].copy()
        
        # Through first lens
        out1 = lens1.trace_ray(o, d, 1.0)
        if out1[2] is False: continue
        o1, d1 = out1[0], out1[1]
        
        # Through second lens
        out2 = lens2.trace_ray(o1, d1, 1.0)
        if out2[2] is False: continue
        o2, d2 = out2[0], out2[1]
        
        # Find intersection with fiber plane
        if abs(d2[2]) < 1e-9: continue  # parallel to fiber face
        t = (z_fiber - o2[2]) / d2[2]
        if t < 0: continue  # going wrong way
        p = o2 + t*d2
        
        # Check if within fiber core
        if math.hypot(p[0], p[1]) > fiber_rad:
            continue
        
        # Check if within acceptance angle
        theta = math.acos(abs(d2[2]) / np.linalg.norm(d2))
        if theta > acceptance_half_rad:
            continue
        
        accepted[i] = True
    
    return accepted

    
############################################
# Visualize ray tracing through the system #
############################################

def plot_system_rays(best_result, n_plot_rays=1000):
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
    lens1 = PlanoConvex(z_l1, lens1_data['R_mm'], lens1_data['t_mm'], 
                        lens1_data['dia']/2.0, n_glass)
    lens2 = PlanoConvex(z_l2, lens2_data['R_mm'], lens2_data['t_mm'], 
                        lens2_data['dia']/2.0, n_glass)
    
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
            ax.plot(points[:,0], points[:,1], points[:,2], 'r-', alpha=0.2)
            continue
        o1, d1 = out1[0], out1[1]
        points.append(o1)
        
        # Through second lens
        out2 = lens2.trace_ray(o1, d1, 1.0)
        if out2[2] is False:
            # Plot failed ray in red
            points = np.array(points)
            ax.plot(points[:,0], points[:,1], points[:,2], 'r-', alpha=0.2)
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
        r = math.hypot(p_f[0], p_f[1])
        theta = math.acos(abs(d2[2]) / np.linalg.norm(d2))
        color = 'g' if (r <= FIBER_CORE_DIAM_MM/2.0 and 
                       theta <= ACCEPTANCE_HALF_RAD) else 'r'
        
        # Plot complete ray path
        points = np.array(points)
        ax.plot(points[:,0], points[:,1], points[:,2], 
                color+'-', alpha=0.5)
    
    # Plot lens surfaces (simplified as disks)
    theta = np.linspace(0, 2*np.pi, 100)
    
    # Lens 1 surfaces
    r = np.linspace(0, lens1_data['dia']/2.0, 2)
    t, r = np.meshgrid(theta, r)
    x = r * np.cos(t)
    y = r * np.sin(t)
    ax.plot_surface(x, y, z_l1 + np.zeros_like(x), alpha=0.2, color='b')
    ax.plot_surface(x, y, z_l1 + lens1_data['t_mm'] + np.zeros_like(x), 
                   alpha=0.2, color='b')
    
    # Lens 2 surfaces
    r = np.linspace(0, lens2_data['dia']/2.0, 2)
    t, r = np.meshgrid(theta, r)
    x = r * np.cos(t)
    y = r * np.sin(t)
    ax.plot_surface(x, y, z_l2 + np.zeros_like(x), alpha=0.2, color='b')
    ax.plot_surface(x, y, z_l2 + lens2_data['t_mm'] + np.zeros_like(x), 
                   alpha=0.2, color='b')
    
    # Plot fiber face
    r = np.linspace(0, FIBER_CORE_DIAM_MM/2.0, 2)
    t, r = np.meshgrid(theta, r)
    x = r * np.cos(t)
    y = r * np.sin(t)
    ax.plot_surface(x, y, z_fiber + np.zeros_like(x), alpha=0.3, color='g')
    
    # Set equal aspect ratio
    ax.set_box_aspect([1,1,1])
    
    # Labels
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    plt.title(f"Ray Trace: {best_result['lens1']} + {best_result['lens2']}, Coupling: {best_result['coupling']:.4f}")
    
    # View angle
    ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()

    # Save plot
    if not os.path.exists('./plots_' + DATE_STR):
        os.makedirs('./plots_' + DATE_STR)
    plt.savefig(f"./plots_{DATE_STR}/C-{best_result['coupling']:.4f}_L1-{best_result['lens1']}_L2-{best_result['lens2']}.png")
    plt.close(fig)
    # plt.show()


################################################################
# Simulation: coarse + refined grid search over lens positions #
################################################################

n_glass = fused_silica_n(WAVELENGTH_NM)
print('Using fused silica n =', n_glass)

combos = []
for a in lenses:
    for b in lenses:
        combos.append((a,b))

# Evaluate a single configuration (given lens vertex positions and a fixed fiber z)
def evaluate_config(z_l1, z_l2, origins, dirs, d1, d2, n_glass, z_fiber, n_rays):
    lens1 = PlanoConvex(vertex_z_front=z_l1, R_front_mm=d1['R_mm'], thickness_mm=d1['t_mm'], ap_rad_mm=d1['dia'], n_glass=n_glass)
    lens2 = PlanoConvex(vertex_z_front=z_l2, R_front_mm=d2['R_mm'], thickness_mm=d2['t_mm'], ap_rad_mm=d2['dia'], n_glass=n_glass)
    accepted = trace_system(origins, dirs, lens1, lens2, z_fiber, FIBER_CORE_DIAM_MM/2.0, ACCEPTANCE_HALF_RAD)
    coupling = np.count_nonzero(accepted) / n_rays
    return coupling, accepted

# Coarse + refine grid search per lens pair
def run_grid(name1, name2, coarse_steps=9, refine_steps=11, n_coarse=3000, n_refine=8000):
    d1 = lenses[name1]; d2 = lenses[name2]
    f1 = d1['f_mm']; f2 = d2['f_mm']
    # Generate ray set once per pair for fair comparison
    origins_coarse, dirs_coarse = sample_rays(n_coarse)
    # coarse search ranges: place lens1 roughly near its focal length, lens2 downstream
    z_l1_min = max(0.5, f1 * 0.5)
    z_l1_max = f1 * 1.5
    best = {'coupling': -1}
    for z_l1 in np.linspace(z_l1_min, z_l1_max, coarse_steps):
        # allow lens2 to vary relative to lens1; keep fiber at z_l2 + f2 (imaging plane assumption)
        z_l2_min = z_l1 + f2 * 0.5
        z_l2_max = z_l1 + f2 * 2.5
        for z_l2 in np.linspace(z_l2_min, z_l2_max, coarse_steps):
            z_fiber = z_l2 + f2
            coupling, accepted = evaluate_config(z_l1, z_l2, origins_coarse, dirs_coarse, d1, d2, n_glass, z_fiber, n_coarse)
            if coupling > best['coupling']:
                best = {'z_l1':z_l1, 'z_l2':z_l2, 'z_fiber':z_fiber, 'coupling':coupling, 'accepted':accepted, 'origins':origins_coarse, 'dirs':dirs_coarse}
    # refine around best
    z1c = best['z_l1']; z2c = best['z_l2']
    dz1 = max(0.05, (z_l1_max - z_l1_min) / (coarse_steps-1) )
    dz2 = max(0.05, ( (z2c - (z1c + f2*0.5)) + ( (z1c + f2*2.5) - z2c) ) / (coarse_steps-1) )
    z1_min = max(0.0, z1c - dz1*2)
    z1_max = z1c + dz1*2
    z2_min = max(z1_min + 0.1, z2c - dz2*2)
    z2_max = z2c + dz2*2
    origins_ref, dirs_ref = sample_rays(n_refine)
    for z_l1 in np.linspace(z1_min, z1_max, refine_steps):
        for z_l2 in np.linspace(z2_min, z2_max, refine_steps):
            z_fiber = z_l2 + f2
            coupling, accepted = evaluate_config(z_l1, z_l2, origins_ref, dirs_ref, d1, d2, n_glass, z_fiber, n_refine)
            if coupling > best['coupling']:
                best = {'z_l1':z_l1, 'z_l2':z_l2, 'z_fiber':z_fiber, 'coupling':coupling, 'accepted':accepted, 'origins':origins_ref, 'dirs':dirs_ref}
    # attach metadata
    best.update({'lens1':name1, 'lens2':name2, 'f1_mm':f1, 'f2_mm':f2, 'total_len_mm':best['z_fiber']})
    
    # Visualize this combination
    plot_system_rays(best)
    
    return best

# run sweep across all combos (coarse+refine)
# Add a progress bar
results = []
print(f"Running coarse+refined grid sweep for {len(combos)} combos (this may take from a few minutes to hours)...")
for (a,b) in tqdm(combos):
    print(f"\nEvaluating {a} + {b} ...")
    res = run_grid(a,b, coarse_steps=7, refine_steps=9, n_coarse=2000, n_refine=6000)
    print(f"best coupling={res['coupling']:.4f} at z_l1={res['z_l1']:.2f}, z_l2={res['z_l2']:.2f}")
    results.append(res)

# build a results DataFrame
rows = [{k:v for k,v in r.items() if k in ['lens1','lens2','f1_mm','f2_mm','z_l1','z_l2','z_fiber','total_len_mm','coupling']} for r in results]
df = pd.DataFrame(rows).sort_values(['coupling','total_len_mm'], ascending=[False, True]).reset_index(drop=True)
print('\nSummary (coarse+refined search):')
print(df.to_string(index=False))

# pick best overall
best = results[np.argmax([r['coupling'] for r in results])]
print('\nBest combo overall:', best['lens1'], best['lens2'], 'coupling =', best['coupling'])

# Spot diagram for best (use the origins/dirs that produced the reported best)
accepted_mask = best['accepted']
origins = best['origins']; dirs = best['dirs']
# compute landing points
land_x = np.full(origins.shape[0], np.nan)
land_y = np.full(origins.shape[0], np.nan)
for i in range(origins.shape[0]):
    o = origins[i].copy(); d = dirs[i].copy()
    out1 = PlanoConvex(vertex_z_front=best['z_l1'], R_front_mm=lenses[best['lens1']]['R_mm'], thickness_mm=lenses[best['lens1']]['t_mm'], ap_rad_mm=lenses[best['lens1']]['dia'], n_glass=n_glass).trace_ray(o,d,1.0)
    if out1[2] is False: continue
    o1,d1 = out1[0], out1[1]
    out2 = PlanoConvex(vertex_z_front=best['z_l2'], R_front_mm=lenses[best['lens2']]['R_mm'], thickness_mm=lenses[best['lens2']]['t_mm'], ap_rad_mm=lenses[best['lens2']]['dia'], n_glass=n_glass).trace_ray(o1,d1,1.0)
    if out2[2] is False: continue
    o2,d2 = out2[0], out2[1]
    if abs(d2[2])<1e-9: continue
    t = (best['z_fiber'] - o2[2]) / d2[2]
    if t < 0: continue
    p = o2 + t * d2
    land_x[i] = p[0]; land_y[i] = p[1]


#####################
# Plot spot diagram #
#####################

plt.figure(figsize=(6,6))
plt.scatter(land_x[~accepted_mask], land_y[~accepted_mask], s=1, color='red', alpha=0.3, label='rejected')
plt.scatter(land_x[accepted_mask], land_y[accepted_mask], s=1, color='green', alpha=0.6, label='accepted')
circle = plt.Circle((0,0), FIBER_CORE_DIAM_MM/2.0, color='blue', fill=False, linewidth=1.5, label='fiber core')
ax = plt.gca(); ax.add_patch(circle)
plt.xlabel('x (mm)'); plt.ylabel('y (mm)'); plt.title(f"Spot diagram: {best['lens1']} + {best['lens2']} (coupling={best['coupling']:.4f})")
plt.axis('equal'); plt.grid(True); plt.legend()

# Save spot diagram
if not os.path.exists('./plots'):
    os.makedirs('./plots')
plt.savefig(f"./plots/spot_C-{best['coupling']:.4f}_L1-{best['lens1']}_L2-{best['lens2']}.png", dpi=300, bbox_inches='tight')
plt.close()

# Save summary table to CSV and latex
if not os.path.exists('./results_' + DATE_STR):
    os.makedirs('./results_' + DATE_STR)
df.to_csv('./results_' + DATE_STR + '/two_lens_coupling_summary.csv', index=False)
df.to_latex('./results_' + DATE_STR + '/two_lens_coupling_summary.tex', index=False)