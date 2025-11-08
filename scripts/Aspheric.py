import math
import numpy as np
from scripts.consts import N_GLASS
from scripts.raytrace_helpers import intersect_ray_sphere, refract_vec


class Aspheric:
    """
    An aspheric condenser lens with an aspheric front surface and spherical back surface.
    
    IMPORTANT LIMITATION (Current Implementation):
    -----------------------------------------------
    This implementation uses a k=0 (spherical) approximation for the aspheric front surface.
    The front surface is traced as a sphere with radius R1_mm (the base radius).
    
    This is because the Edmund Optics condenser lens data does NOT include:
    - Conic constant (k)
    - Aspheric polynomial coefficients (A₂, A₄, A₆, A₈, ...)
    
    Full aspheric surface equation:
        z(r) = (c*r²)/(1 + sqrt(1 - (1+k)*c²*r²)) + A₂*r² + A₄*r⁴ + A₆*r⁶ + A₈*r⁸ + ...
    
    where:
        c = 1/R (curvature)
        k = conic constant (k=0 for sphere, k=-1 for parabola, etc.)
        r = sqrt(x² + y²) (radial distance from optical axis)
        A₂, A₄, ... = aspheric polynomial coefficients
    
    Current approximation (k=0, all A_i=0):
        z(r) = R - sqrt(R² - r²)  (simple sphere)
    
    FUTURE EXPANSION:
    -----------------
    When aspheric coefficients become available:
    1. Replace intersect_ray_sphere() for front surface with intersect_ray_aspheric()
    2. Update normal vector calculation for aspheric surface
    3. Update visualization code to show true aspheric profile
    
    Convention:
    -----------
    If flipped=False (default):
        Front surface (R1): Aspheric with base radius R_front_mm (currently traced as sphere)
        Back surface (R2): Spherical with radius R_back_mm
        
    If flipped=True:
        Front surface: Spherical with radius R_back_mm
        Back surface: Aspheric with base radius R_front_mm (currently traced as sphere)
        
    The flipped parameter allows testing both orientations of the lens.
    """

    def __init__(self, vertex_z_front, R_front_mm, R_back_mm,
                 center_thickness_mm, edge_thickness_mm, ap_rad_mm, 
                 conic_constant=0.0, aspheric_coeffs=None, flipped=False):
        """
        Initialize aspheric lens with its parameters.
        
        Parameters:
        -----------
        vertex_z_front : float
            Z position of the front vertex (mm)
        R_front_mm : float
            Base radius of aspheric front surface (mm) - currently used as sphere radius
        R_back_mm : float
            Radius of spherical back surface (mm)
        center_thickness_mm : float
            Thickness at optical axis (mm)
        edge_thickness_mm : float
            Thickness at edge (mm)
        ap_rad_mm : float
            Aperture radius (mm)
        conic_constant : float, optional
            Conic constant k (default: 0.0 = sphere)
            Currently NOT USED - reserved for future implementation
        aspheric_coeffs : dict, optional
            Dictionary of aspheric coefficients {'A2': val, 'A4': val, ...}
            Currently NOT USED - reserved for future implementation
        flipped : bool, optional
            If True, swap front and back orientations (default: False)
        """
        self.vertex_z_front = vertex_z_front
        self.center_thickness_mm = center_thickness_mm
        self.edge_thickness_mm = edge_thickness_mm
        self.ap_rad_mm = ap_rad_mm
        self.n_glass = N_GLASS
        self.flipped = flipped
        self.vertex_z_back = vertex_z_front + center_thickness_mm
        self.z_center = vertex_z_front + center_thickness_mm / 2.0
        
        # Store aspheric parameters for future use
        self.conic_constant = conic_constant
        self.aspheric_coeffs = aspheric_coeffs if aspheric_coeffs is not None else {}
        
        # If flipped, swap the radii (PRESERVE SIGN for back surface!)
        if not flipped:
            self.R_front_mm = abs(R_front_mm)  # Base radius of aspheric surface (always positive for front)
            self.R_back_mm = R_back_mm         # Radius of spherical back (KEEP SIGN: +convex, -concave)
        else:
            # Swap the radii when flipped
            self.R_front_mm = abs(R_back_mm)
            self.R_back_mm = R_front_mm
        
        # Detect back surface type based on radius
        # Threshold for plano: if |R| > 1000mm, treat as flat surface
        if abs(self.R_back_mm) > 1000.0:
            self.back_surface_type = 'plano'
        elif self.R_back_mm > 0:
            self.back_surface_type = 'convex'
        else:  # R_back_mm < 0
            self.back_surface_type = 'concave'
        
        # Center of curvature for front surface (on +z side)
        # For aspheric: this is the center of the BASE SPHERE (k=0 approximation)
        self.center_z_front = vertex_z_front + self.R_front_mm
        
        # Center of curvature for back surface depends on surface type
        if self.back_surface_type == 'plano':
            # No center for planar surface
            self.center_z_back = None
        elif self.back_surface_type == 'convex':
            # Convex: center on -z side of back vertex (R > 0)
            self.center_z_back = self.vertex_z_back - self.R_back_mm
        else:  # concave
            # Concave: center on +z side of back vertex (R < 0, so subtract negative = add)
            self.center_z_back = self.vertex_z_back - self.R_back_mm

    def trace_ray(self, o, d, n1):
        """
        Trace a ray through the aspheric lens.
        
        NOTE: Currently uses spherical approximation (k=0) for aspheric front surface.
        
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
        # Front surface (aspheric, but currently approximated as sphere)
        # TODO: Replace with true aspheric intersection when k≠0 or coefficients available
        c_front = np.array([0, 0, self.center_z_front])
        t = intersect_ray_sphere(o, d, c_front, self.R_front_mm)
        if t is None:
            return None, None, False
        
        p_front = o + t * d  # Entry point
        
        # Check aperture at front
        if math.hypot(p_front[0], p_front[1]) > self.ap_rad_mm:
            return None, None, False
        
        # Surface normal at front (points outward from glass)
        # TODO: Replace with true aspheric normal when k≠0 or coefficients available
        n_front = (p_front - c_front) / self.R_front_mm
        
        # Refract into glass
        d_in = refract_vec(n_front, d, n1, self.n_glass)
        if d_in is None:
            return None, None, False
        
        # Back surface - handle three types: plano, convex, concave
        if self.back_surface_type == 'plano':
            # Planar back surface at vertex_z_back
            if abs(d_in[2]) < 1e-9:
                return None, None, False
            t_back = (self.vertex_z_back - p_front[2]) / d_in[2]
            if t_back < 0:
                return None, None, False
            p_back = p_front + t_back * d_in
            
            # Check aperture at back
            if math.hypot(p_back[0], p_back[1]) > self.ap_rad_mm:
                return None, None, False
            
            # Surface normal for plane (points outward from glass, toward -z)
            n_back = np.array([0, 0, -1])
            
        elif self.back_surface_type == 'convex':
            # Convex spherical back surface (center on -z side)
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
            
        else:  # concave
            # Concave spherical back surface (center on +z side, R_back_mm < 0)
            c_back = np.array([0, 0, self.center_z_back])
            t = intersect_ray_sphere(p_front, d_in, c_back, abs(self.R_back_mm))
            if t is None:
                return None, None, False
            
            p_back = p_front + t * d_in  # Exit point
            
            # Check aperture at back
            if math.hypot(p_back[0], p_back[1]) > self.ap_rad_mm:
                return None, None, False
            
            # Surface normal at back (points outward from glass, toward +z)
            # For concave surface, center is on +z side, so outward normal points toward -z
            n_back = (p_back - c_back) / abs(self.R_back_mm)
        
        # Refract out of glass
        d_out = refract_vec(n_back, d_in, self.n_glass, n1)
        if d_out is None:
            return None, None, False
        
        return p_back, d_out, True
    
    def trace_ray_detailed(self, o, d, n1):
        """
        Trace a ray through the lens with detailed intermediate points.
        
        Returns entry and exit points for visualization purposes.
        
        NOTE: Currently uses spherical approximation (k=0) for aspheric front surface.

        Parameters:
        -----------
        o : array
            3D origin point
        d : array
            3D direction vector (normalized)
        n1 : float
            Input refractive index

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
        # Front surface (aspheric, but currently approximated as sphere)
        # TODO: Replace with true aspheric intersection when k≠0 or coefficients available
        c_front = np.array([0, 0, self.center_z_front])
        t = intersect_ray_sphere(o, d, c_front, self.R_front_mm)
        if t is None:
            return None, None, None, None, False
        
        p_front = o + t * d  # ENTRY point
        
        # Check aperture at front
        if math.hypot(p_front[0], p_front[1]) > self.ap_rad_mm:
            return None, None, None, None, False
        
        # Surface normal at front (points outward from glass)
        # TODO: Replace with true aspheric normal when k≠0 or coefficients available
        n_front = (p_front - c_front) / self.R_front_mm
        
        # Refract into glass
        d_in = refract_vec(n_front, d, n1, self.n_glass)
        if d_in is None:
            return None, None, None, None, False
        
        # Back surface - handle three types: plano, convex, concave
        if self.back_surface_type == 'plano':
            # Planar back surface at vertex_z_back
            if abs(d_in[2]) < 1e-9:
                return None, None, None, None, False
            t_back = (self.vertex_z_back - p_front[2]) / d_in[2]
            if t_back < 0:
                return None, None, None, None, False
            p_back = p_front + t_back * d_in  # EXIT point
            
            # Check aperture at back
            if math.hypot(p_back[0], p_back[1]) > self.ap_rad_mm:
                return None, None, None, None, False
            
            # Surface normal for plane (points outward from glass, toward -z)
            n_back = np.array([0, 0, -1])
            
        elif self.back_surface_type == 'convex':
            # Convex spherical back surface (center on -z side)
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
            
        else:  # concave
            # Concave spherical back surface (center on +z side, R_back_mm < 0)
            c_back = np.array([0, 0, self.center_z_back])
            t = intersect_ray_sphere(p_front, d_in, c_back, abs(self.R_back_mm))
            if t is None:
                return None, None, None, None, False
            
            p_back = p_front + t * d_in  # EXIT point
            
            # Check aperture at back
            if math.hypot(p_back[0], p_back[1]) > self.ap_rad_mm:
                return None, None, None, None, False
            
            # Surface normal at back (points outward from glass, toward +z)
            # For concave surface, center is on +z side, so outward normal points toward -z
            n_back = (p_back - c_back) / abs(self.R_back_mm)
        
        # Refract out of glass
        d_out = refract_vec(n_back, d_in, self.n_glass, n1)
        if d_out is None:
            return None, None, None, None, False
        
        return p_front, d_in, p_back, d_out, True
