import numpy as np
import math
from scripts import consts as C
from scripts.calcs import transmission_through_medium


def sample_rays(n_rays):
    arc_radius = C.SOURCE_ARC_DIAM_MM / 2.0

    r = np.sqrt(np.random.rand(n_rays)) * arc_radius  # radial positions
    phi = np.linspace(0, 2*np.pi, n_rays)  # angular positions around circle

    # Source points
    x_source = r * np.cos(phi)
    y_source = r * np.sin(phi)
    origins = np.vstack([x_source, y_source, np.zeros_like(x_source)]).T

    # Calculate coherent ray angles based on radial position
    # Angle increases linearly with radius (0 at center, max_angle_deg at edge)
    ray_angles = np.deg2rad(C.MAX_ANGLE_DEG * r / arc_radius)

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
    c0 = np.dot(oc, oc) - R*R
    a = np.dot(d, d)
    disc = b*b - 4*a*c0

    if disc < 0:
        return None

    sqrt_d = math.sqrt(disc)
    t1 = (-b - sqrt_d) / (2*a)
    t2 = (-b + sqrt_d) / (2*a)
    ts = [t for t in (t1, t2) if t > 1e-9]

    if not ts:
        return None

    return min(ts)


def refract_vec(n_vec, v_in, n1, n2):
    n_vec = np.array(n_vec)
    v_in = np.array(v_in)
    n_vec = n_vec / np.linalg.norm(n_vec)
    v = v_in / np.linalg.norm(v_in)
    cos_i = -np.dot(n_vec, v)
    eta = n1 / n2
    k = 1 - eta*eta * (1 - cos_i*cos_i)

    if k < 0:
        return None

    v_out = eta * v + (eta * cos_i - math.sqrt(k)) * n_vec
    v_out = v_out / np.linalg.norm(v_out)

    return v_out


def trace_system(origins, dirs, lens1, lens2,
                 z_fiber, fiber_rad, acceptance_half_rad, 
                 medium='air', pressure_atm=1.0, temp_k=293.15, humidity_fraction=0.01):
    """
    Trace rays through system and check if they make it into the fiber.

    Parameters:
    - origins, dirs: Nx3 arrays of ray origins and directions
    - lens1, lens2: PlanoConvex objects for first and second lens
    - z_fiber: z-position of fiber face
    - fiber_rad: fiber core radius
    - acceptance_half_rad: half-acceptance angle in radians
    - medium: propagation medium ('air', 'argon', 'helium')
    - pressure_atm: pressure in atmospheres
    - temp_k: temperature in Kelvin
    - humidity_fraction: water vapor fraction (0.0-0.1 typical, 0.01 = 1%)

    Returns:
    - accepted: boolean array indicating which rays made it into fiber
    """
    n_rays = origins.shape[0]
    accepted = np.zeros(n_rays, dtype=bool)
    transmission_factors = np.ones(n_rays)

    for i in range(n_rays):
        o = origins[i].copy()
        d = dirs[i].copy()

        z_l1 = lens1.z_center
        d1 = z_l1 - o[2]
        T1 = transmission_through_medium(d1, C.WAVELENGTH_NM, medium, pressure_atm, temp_k, humidity_fraction)

        out1 = lens1.trace_ray(o, d, 1.0)
        if out1[2] is False:
            continue
        o1, d1_out = out1[0], out1[1]

        z_l2 = lens2.z_center
        d2 = z_l2 - o1[2]
        T2 = transmission_through_medium(d2, C.WAVELENGTH_NM, medium, pressure_atm, temp_k, humidity_fraction)

        out2 = lens2.trace_ray(o1, d1_out, 1.0)
        if out2[2] is False:
            continue
        o2, d2_out = out2[0], out2[1]

        if abs(d2_out[2]) < 1e-9:
            continue
        t = (z_fiber - o2[2]) / d2_out[2]
        if t < 0:
            continue
        p = o2 + t*d2_out

        d3 = abs(z_fiber - o2[2])
        T3 = transmission_through_medium(d3, C.WAVELENGTH_NM, medium, pressure_atm, temp_k, humidity_fraction)

        if math.hypot(p[0], p[1]) > fiber_rad:
            continue

        theta = math.acos(abs(d2_out[2]) / np.linalg.norm(d2_out))
        if theta > acceptance_half_rad:
            continue

        accepted[i] = True
        transmission_factors[i] = T1 * T2 * T3

    return accepted, transmission_factors
