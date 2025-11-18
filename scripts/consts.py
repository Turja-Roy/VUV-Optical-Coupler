from scripts.calcs import fused_silica_n


WAVELENGTH_NM = 200.0                       # 200 nm

# Fiber
FIBER_CORE_DIAM_MM = 1.0                    # 1000 micron
NA = 0.22
ACCEPTANCE_HALF_RAD = __import__("numpy").deg2rad(12.4)

# Source geometry - WITH COOLING JACKET
SOURCE_ARC_DIAM_MM = 3.0                    # Arc diameter
LAMP_WINDOW_DIAM_MM = 14.3                  # Lamp window diameter (at 8.7mm from arc)
COOLING_JACKET_THREAD_DIAM_MM = 23.0        # M23 thread inner diameter
LAMP_WINDOW_DISTANCE_MM = 8.7               # Distance from arc to lamp window
WINDOW_DISTANCE_MM = 26                     # Distance from arc to cooling jacket exit (23.5±1.5 + ~1mm)
MAX_ANGLE_DEG = 22.85                       # Limited by cooling jacket: atan(11.5/26.3) = 22.85°
GEOMETRIC_LOSS_FACTOR = 0.43                # Solid angle ratio due to cooling jacket vignetting (51.4% light loss)
SOURCE_RADIUS_MM = SOURCE_ARC_DIAM_MM/2.0   # Source radius for ray generation

# Legacy constant for backwards compatibility
WINDOW_DIAM_MM = LAMP_WINDOW_DIAM_MM        # Deprecated: use LAMP_WINDOW_DIAM_MM

# Position offset for lenses
SOURCE_TO_LENS_OFFSET = WINDOW_DISTANCE_MM + 1  # Lenses start after the window

# Rays
N_RAYS = 1000
N_GLASS = fused_silica_n(WAVELENGTH_NM)

# Performance settings
USE_VECTORIZED_TRACING = True  # Use vectorized ray tracing (10-15x faster)

# Database settings
USE_DATABASE = True  # Enable SQLite database for storing results
DATABASE_PATH = './results/optimization.db'  # Path to database file
LENS_DATABASE_PATH = './data/lenses.db'  # Path to lens catalog database

MEDIUM = 'air'
PRESSURE_ATM = 1.0
TEMPERATURE_K = 293.15
HUMIDITY_FRACTION = 0.01

DATE_STR = __import__("time").strftime("%Y-%m-%d")
