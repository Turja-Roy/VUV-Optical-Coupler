from scripts.calcs import fused_silica_n


WAVELENGTH_NM = 200.0                       # 200 nm

# Fiber
FIBER_CORE_DIAM_MM = 1.0                    # 1000 micron
NA = 0.22
ACCEPTANCE_HALF_RAD = __import__("numpy").deg2rad(12.4)

# Source geometry
SOURCE_ARC_DIAM_MM = 3.0                    # Arc diameter
WINDOW_DIAM_MM = 14.3                       # Window diameter
WINDOW_DISTANCE_MM = 8.7                    # Distance from arc to window
MAX_ANGLE_DEG = 33                          # Maximum ray angle at window edge
SOURCE_RADIUS_MM = SOURCE_ARC_DIAM_MM/2.0   # Source radius for ray generation

# Position offset for lenses
SOURCE_TO_LENS_OFFSET = WINDOW_DISTANCE_MM + 1  # Lenses start after the window

# Rays
N_RAYS = 1000
N_GLASS = fused_silica_n(WAVELENGTH_NM)

# Date string
DATE_STR = __import__("time").strftime("%Y-%m-%d")

# Grid run parameters
COARSE_STEPS = 7
REFINE_STEPS = 9
N_COARSE = 500  # 2000
N_REFINE = 1000  # 6000
