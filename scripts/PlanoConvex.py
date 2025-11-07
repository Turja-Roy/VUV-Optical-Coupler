import math
import numpy as np
from scripts.consts import N_GLASS
from scripts.raytrace_helpers import intersect_ray_sphere, refract_vec


class PlanoConvex:
    """
    A plano-convex lens with spherical front surface.
    
    If flipped=False (default):
        Front surface is convex (center of curvature on +z side).
        Back surface is planar.
        Light path: curved -> flat
        
    If flipped=True:
        Front surface is planar.
        Back surface is convex (center of curvature on -z side).
        Light path: flat -> curved
    """

    def __init__(self, vertex_z_front, R_front_mm,
                 center_thickness_mm, edge_thickness_mm, ap_rad_mm, flipped=False):
        """Initialize lens with its parameters."""
        self.vertex_z_front = vertex_z_front
        self.R_front_mm = R_front_mm
        self.center_thickness_mm = center_thickness_mm
        self.edge_thickness_mm = edge_thickness_mm
        self.ap_rad_mm = ap_rad_mm
        self.n_glass = N_GLASS
        self.flipped = flipped
        self.vertex_z_back = vertex_z_front + center_thickness_mm
        self.z_center = vertex_z_front + center_thickness_mm / 2.0
        
        if not flipped:
            # Normal orientation: curved face first
            self.center_z_front = vertex_z_front + R_front_mm
        else:
            # Flipped orientation: flat face first, curved face at back
            # Center of curvature is on the -z side of the back surface
            self.center_z_back = self.vertex_z_back - R_front_mm

    def trace_ray(self, o, d, n1):
        """
        Trace a ray through the lens.

        Parameters:
        o : array
            3D origin point of the ray
        d : array
            3D direction vector (normalized)
        n1 : float
            Input refractive index

        Returns:
        - (o_out, d_out, success)
            o_out: Exit point
            d_out: Exit direction
            success: True if ray successfully traced
        """
        if not self.flipped:
            # Normal orientation: curved face first, flat face second
            return self._trace_curved_flat(o, d, n1)
        else:
            # Flipped orientation: flat face first, curved face second
            return self._trace_flat_curved(o, d, n1)
    
    def trace_ray_detailed(self, o, d, n1):
        """
        Trace a ray through the lens with detailed intermediate points.
        
        This method returns both entry and exit points for visualization purposes.

        Parameters:
        - o: 3D origin point
        - d: 3D direction vector (normalized)
        - n1: input refractive index

        Returns:
        tuple : (entry_point, dir_in_glass, exit_point, dir_out, success)
          where:
            entry_point: point where ray enters the lens
            dir_in_glass: refracted direction inside the lens
            exit_point: point where ray exits the lens
            dir_out: refracted direction after exiting
            success: True if ray successfully traced
        """
        if not self.flipped:
            return self._trace_curved_flat_detailed(o, d, n1)
        else:
            return self._trace_flat_curved_detailed(o, d, n1)
    
    def _trace_curved_flat(self, o, d, n1):
        """Trace through lens with curved face first (normal orientation)."""
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

        # Go to back surface (planar at vertex_z_back)
        # The back surface is a flat plane at z = vertex_z_back
        if abs(d_in[2]) < 1e-9:
            return None, None, False
        t_back = (self.vertex_z_back - p[2]) / d_in[2]
        if t_back < 0:
            return None, None, False
        o_back = p + t_back * d_in

        # Check aperture at back
        if math.hypot(o_back[0], o_back[1]) > self.ap_rad_mm:
            return None, None, False

        # Refract out of glass (planar surface, normal = -z)
        d_out = refract_vec(np.array([0, 0, -1]), d_in, self.n_glass, n1)
        if d_out is None:
            return None, None, False

        return o_back, d_out, True
    
    def _trace_flat_curved(self, o, d, n1):
        """Trace through lens with flat face first (flipped orientation)."""
        # Front surface (planar)
        # Find intersection with flat face at vertex_z_front
        if abs(d[2]) < 1e-9:
            return None, None, False
        
        t_front = (self.vertex_z_front - o[2]) / d[2]
        if t_front < 0:
            return None, None, False
        
        p_front = o + t_front * d
        
        # Check aperture at front
        if math.hypot(p_front[0], p_front[1]) > self.ap_rad_mm:
            return None, None, False
        
        # Refract into glass (planar surface, normal = +z for incoming ray)
        # Normal points in -z direction (out of glass), but ray is entering
        d_in = refract_vec(np.array([0, 0, -1]), d, n1, self.n_glass)
        if d_in is None:
            return None, None, False
        
        # Back surface (sphere) - center is on -z side of vertex_z_back
        c = np.array([0, 0, self.center_z_back])
        t = intersect_ray_sphere(p_front, d_in, c, self.R_front_mm)
        if t is None:
            return None, None, False
        
        p_back = p_front + t * d_in
        
        # Check aperture at back
        if math.hypot(p_back[0], p_back[1]) > self.ap_rad_mm:
            return None, None, False
        
        # Surface normal (points out of glass toward +z)
        # For a sphere centered at center_z_back (which is < vertex_z_back),
        # the outward normal from the back surface points away from center
        n = -(p_back - c) / self.R_front_mm
        
        # Refract out of glass
        d_out = refract_vec(n, d_in, self.n_glass, n1)
        if d_out is None:
            return None, None, False
        
        return p_back, d_out, True
    
    def _trace_curved_flat_detailed(self, o, d, n1):
        """Trace through lens with curved face first (normal orientation) - detailed version."""
        # Front surface (sphere)
        c = np.array([0, 0, self.center_z_front])  # center
        t = intersect_ray_sphere(o, d, c, self.R_front_mm)
        if t is None:
            return None, None, None, None, False
        p = o + t*d  # intersection point (ENTRY)

        # Check aperture
        if math.hypot(p[0], p[1]) > self.ap_rad_mm:
            return None, None, None, None, False

        # Surface normal (points out of glass)
        n = (p - c) / self.R_front_mm

        # Refract into glass
        d_in = refract_vec(n, d, n1, self.n_glass)
        if d_in is None:
            return None, None, None, None, False

        # Go to back surface (planar at vertex_z_back) - EXIT point
        # The back surface is a flat plane at z = vertex_z_back
        if abs(d_in[2]) < 1e-9:
            return None, None, None, None, False
        t_back = (self.vertex_z_back - p[2]) / d_in[2]
        if t_back < 0:
            return None, None, None, None, False
        o_back = p + t_back * d_in

        # Check aperture at back
        if math.hypot(o_back[0], o_back[1]) > self.ap_rad_mm:
            return None, None, None, None, False

        # Refract out of glass (planar surface, normal = -z)
        d_out = refract_vec(np.array([0, 0, -1]), d_in, self.n_glass, n1)
        if d_out is None:
            return None, None, None, None, False

        return p, d_in, o_back, d_out, True
    
    def _trace_flat_curved_detailed(self, o, d, n1):
        """Trace through lens with flat face first (flipped orientation) - detailed version."""
        # Front surface (planar)
        # Find intersection with flat face at vertex_z_front
        if abs(d[2]) < 1e-9:
            return None, None, None, None, False
        
        t_front = (self.vertex_z_front - o[2]) / d[2]
        if t_front < 0:
            return None, None, None, None, False
        
        p_front = o + t_front * d  # ENTRY point
        
        # Check aperture at front
        if math.hypot(p_front[0], p_front[1]) > self.ap_rad_mm:
            return None, None, None, None, False
        
        # Refract into glass (planar surface, normal = +z for incoming ray)
        # Normal points in -z direction (out of glass), but ray is entering
        d_in = refract_vec(np.array([0, 0, -1]), d, n1, self.n_glass)
        if d_in is None:
            return None, None, None, None, False
        
        # Back surface (sphere) - center is on -z side of vertex_z_back
        c = np.array([0, 0, self.center_z_back])
        t = intersect_ray_sphere(p_front, d_in, c, self.R_front_mm)
        if t is None:
            return None, None, None, None, False
        
        p_back = p_front + t * d_in  # EXIT point
        
        # Check aperture at back
        if math.hypot(p_back[0], p_back[1]) > self.ap_rad_mm:
            return None, None, None, None, False
        
        # Surface normal (points out of glass toward +z)
        # For a sphere centered at center_z_back (which is < vertex_z_back),
        # the outward normal from the back surface points away from center
        n = -(p_back - c) / self.R_front_mm
        
        # Refract out of glass
        d_out = refract_vec(n, d_in, self.n_glass, n1)
        if d_out is None:
            return None, None, None, None, False
        
        return p_front, d_in, p_back, d_out, True
