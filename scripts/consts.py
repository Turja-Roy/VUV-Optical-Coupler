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

# Performance settings
USE_VECTORIZED_TRACING = True  # Use vectorized ray tracing (10-15x faster)

MEDIUM = 'air'
PRESSURE_ATM = 1.0
TEMPERATURE_K = 293.15
HUMIDITY_FRACTION = 0.01

DATE_STR = __import__("time").strftime("%Y-%m-%d")
