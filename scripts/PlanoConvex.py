import math
import numpy as np
from scripts.consts import N_GLASS
from scripts.raytrace_helpers import intersect_ray_sphere, refract_vec


class PlanoConvex:
    """
    A plano-convex lens with spherical front surface.
    Front surface is convex (center of curvature on +z side).
    Back surface is planar.
    """

    def __init__(self, vertex_z_front, R_front_mm,
                 center_thickness_mm, edge_thickness_mm, ap_rad_mm):
        """Initialize lens with its parameters."""
        self.vertex_z_front = vertex_z_front
        self.R_front_mm = R_front_mm
        self.center_thickness_mm = center_thickness_mm
        self.edge_thickness_mm = edge_thickness_mm
        self.ap_rad_mm = ap_rad_mm
        self.n_glass = N_GLASS
        # Derived quantities
        self.vertex_z_back = vertex_z_front + center_thickness_mm
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
        if t is None:
            return None, None, False
        p = o + t*d  # intersection point

        # Check aperture
        if math.hypot(p[0], p[1]) > self.ap_rad_mm:
            return None, None, False

        # Surface normal (points out of glass)
        n = (p - c) / self.R_front_mm

        # Refract into glass
        d_in = refract_vec(n, d, n1, self.n_glass)
        if d_in is None:
            return None, None, False

        # Calculate local thickness at this radial position
        r = math.hypot(p[0], p[1])
        local_thickness = self.center_thickness_mm - (self.center_thickness_mm - self.edge_thickness_mm) * (r / self.ap_rad_mm)

        # Go to back surface (planar)
        o_back = p + (local_thickness/abs(d_in[2])) * d_in

        # Check aperture at back
        if math.hypot(o_back[0], o_back[1]) > self.ap_rad_mm:
            return None, None, False

        # Refract out of glass (planar surface, normal = -z)
        d_out = refract_vec(np.array([0, 0, -1]), d_in, self.n_glass, n1)
        if d_out is None:
            return None, None, False

        return o_back, d_out, True
