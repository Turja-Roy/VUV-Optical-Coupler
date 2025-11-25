import math
import numpy as np
from scripts.raytrace_helpers import intersect_ray_sphere, refract_vec


class BiConvex:
    """
    A bi-convex lens with spherical surfaces on both sides.
    
    Convention:
        If flipped=False (default):
            Front surface (R1): uses R_front_mm, center of curvature on +z side
            Back surface (R2): uses R_back_mm, center of curvature on -z side
        
        If flipped=True:
            Front surface (R1): uses R_back_mm, center of curvature on +z side
            Back surface (R2): uses R_front_mm, center of curvature on -z side
            
    The flipped parameter allows testing both orientations of an asymmetric bi-convex lens,
    which affects optical aberrations and focusing behavior.
    """

    def __init__(self, vertex_z_front, R_front_mm, R_back_mm,
                 center_thickness_mm, edge_thickness_mm, ap_rad_mm, flipped=False):
        """Initialize bi-convex lens with its parameters."""
        self.vertex_z_front = vertex_z_front
        self.center_thickness_mm = center_thickness_mm
        self.edge_thickness_mm = edge_thickness_mm
        self.ap_rad_mm = ap_rad_mm
        self.flipped = flipped
        self.vertex_z_back = vertex_z_front + center_thickness_mm
        self.z_center = vertex_z_front + center_thickness_mm / 2.0
        
        # If flipped, swap the radii
        if not flipped:
            self.R_front_mm = abs(R_front_mm)  # Always positive for our convention
            self.R_back_mm = abs(R_back_mm)    # Always positive for our convention
        else:
            # Swap the radii when flipped
            self.R_front_mm = abs(R_back_mm)
            self.R_back_mm = abs(R_front_mm)
        
        # Center of curvature for front surface (on +z side)
        self.center_z_front = vertex_z_front + self.R_front_mm
        # Center of curvature for back surface (on -z side of back vertex)
        self.center_z_back = self.vertex_z_back - self.R_back_mm
    
    @property
    def n_glass(self):
        """
        Glass refractive index calculated dynamically based on current wavelength.
        This ensures wavelength-dependent refraction is handled correctly.
        """
        from scripts import consts as C
        from scripts.calcs import fused_silica_n
        return fused_silica_n(C.WAVELENGTH_NM)

    def trace_ray(self, o, d, n1):
        """
        Trace a ray through the bi-convex lens.

        Parameters:
        -----------
        o : array
            3D origin point of the ray
        d : array
            3D direction vector (normalized)
        n1 : float
            Input refractive index

        Returns:
        --------
        tuple : (o_out, d_out, success)
            o_out: Exit point
            d_out: Exit direction
            success: True if ray successfully traced
        """
        # Front surface (convex sphere)
        c_front = np.array([0, 0, self.center_z_front])
        t = intersect_ray_sphere(o, d, c_front, self.R_front_mm)
        if t is None:
            return None, None, False
        
        p_front = o + t * d  # Entry point
        
        # Check aperture at front
        if math.hypot(p_front[0], p_front[1]) > self.ap_rad_mm:
            return None, None, False
        
        # Surface normal at front (points outward from glass)
        n_front = (p_front - c_front) / self.R_front_mm
        
        # Refract into glass
        d_in = refract_vec(n_front, d, n1, self.n_glass)
        if d_in is None:
            return None, None, False
        
        # Back surface (convex sphere, center on -z side)
        c_back = np.array([0, 0, self.center_z_back])
        t = intersect_ray_sphere(p_front, d_in, c_back, self.R_back_mm)
        if t is None:
            return None, None, False
        
        p_back = p_front + t * d_in  # Exit point
        
        # Check aperture at back
        if math.hypot(p_back[0], p_back[1]) > self.ap_rad_mm:
            return None, None, False
        
        # Surface normal at back (points outward from glass, toward +z)
        # Since center is on -z side, outward normal points away from center
        n_back = -(p_back - c_back) / self.R_back_mm
        
        # Refract out of glass
        d_out = refract_vec(n_back, d_in, self.n_glass, n1)
        if d_out is None:
            return None, None, False
        
        return p_back, d_out, True
    
    def trace_ray_detailed(self, o, d, n1):
        """
        Trace a ray through the lens with detailed intermediate points.
        
        Returns entry and exit points for visualization purposes.

        Parameters:
        -----------
        - o: 3D origin point
        - d: 3D direction vector (normalized)
        - n1: input refractive index

        Returns:
        --------
        tuple : (entry_point, dir_in_glass, exit_point, dir_out, success)
          where:
            entry_point: point where ray enters the lens
            dir_in_glass: refracted direction inside the lens
            exit_point: point where ray exits the lens
            dir_out: refracted direction after exiting
            success: True if ray successfully traced
        """
        # Front surface (convex sphere)
        c_front = np.array([0, 0, self.center_z_front])
        t = intersect_ray_sphere(o, d, c_front, self.R_front_mm)
        if t is None:
            return None, None, None, None, False
        
        p_front = o + t * d  # ENTRY point
        
        # Check aperture at front
        if math.hypot(p_front[0], p_front[1]) > self.ap_rad_mm:
            return None, None, None, None, False
        
        # Surface normal at front (points outward from glass)
        n_front = (p_front - c_front) / self.R_front_mm
        
        # Refract into glass
        d_in = refract_vec(n_front, d, n1, self.n_glass)
        if d_in is None:
            return None, None, None, None, False
        
        # Back surface (convex sphere, center on -z side)
        c_back = np.array([0, 0, self.center_z_back])
        t = intersect_ray_sphere(p_front, d_in, c_back, self.R_back_mm)
        if t is None:
            return None, None, None, None, False
        
        p_back = p_front + t * d_in  # EXIT point
        
        # Check aperture at back
        if math.hypot(p_back[0], p_back[1]) > self.ap_rad_mm:
            return None, None, None, None, False
        
        # Surface normal at back (points outward from glass, toward +z)
        n_back = -(p_back - c_back) / self.R_back_mm
        
        # Refract out of glass
        d_out = refract_vec(n_back, d_in, self.n_glass, n1)
        if d_out is None:
            return None, None, None, None, False
        
        return p_front, d_in, p_back, d_out, True
