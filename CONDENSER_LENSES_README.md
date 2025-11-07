# Condenser Lens Processing - Setup Complete

## Overview

This project now includes tools to work with Edmund Optics condenser lenses, including support for bi-convex lenses that weren't previously supported.

## Files Added

### 1. `scripts/process_condenser_lenses.py`
Python script to extract and process condenser lens data from Excel to CSV format.

**Usage:**
```bash
python scripts/process_condenser_lenses.py \
    --input data/Edmund-Optics_condenser-lenses-731b9cef.xlsx \
    --output data/Edmund-Optics_condenser-lenses.csv
```

**Features:**
- Reads Excel file with 102 condenser lenses
- Extracts optical parameters (diameter, focal length, thickness, etc.)
- Cleans numeric data (removes tolerance annotations)
- Calculates front surface radius (R1) using lensmaker's equation
- Classifies lenses by type
- Outputs structured CSV file

### 2. `scripts/BiConvex.py`
New lens class for bi-convex lenses (both surfaces curved).

**Features:**
- Handles raytracing through bi-convex spherical lenses
- Compatible interface with existing `PlanoConvex` class
- Implements `trace_ray(o, d, n1)` method
- Implements `trace_ray_detailed(o, d, n1)` for visualization
- Properly handles refraction at two spherical surfaces

**Usage Example:**
```python
from scripts.BiConvex import BiConvex

lens = BiConvex(
    vertex_z_front=10.0,
    R_front_mm=15.65,
    R_back_mm=15.65,
    center_thickness_mm=7.5,
    edge_thickness_mm=1.86,
    ap_rad_mm=6.35  # half of diameter
)

exit_point, exit_dir, success = lens.trace_ray(origin, direction, n_air)
```

### 3. `data/Edmund-Optics_condenser-lenses.csv`
Processed lens data ready for use in simulations.

**Contents:**
- 102 condenser lenses
- 54 Plano-Convex lenses (use existing `PlanoConvex` class)
- 40 Bi-Convex lenses (use new `BiConvex` class)
- 8 Aspheric lenses (future work - would need `Aspheric` class)

**Specifications:**
- Diameter range: 6.5 - 80.0 mm
- Focal length range: 4.5 - 59.0 mm
- Multiple coatings available (Uncoated, MgF2, VIS-EXT, NIR I)
- Various substrates (Float Glass, H-K51, H-ZK2, H-K9L, B270)

## Data Format

The CSV includes these columns:
- `Item #` - Stock number identifier
- `Title` - Full lens description
- `Diameter (mm)` - Lens diameter
- `Center Thickness (mm)` - Thickness at center
- `Edge Thickness (mm)` - Thickness at edge
- `Focal Length (mm)` - Effective focal length
- `Back Focal Length (mm)` - Distance from back surface to focal point
- `Numerical Aperture` - NA value
- `f/#` - F-number
- `Back Surface Shape` - Plano, Convex, or Aspheric
- `Radius R1 (mm)` - Front surface radius (calculated)
- `Radius R2 (mm)` - Back surface radius
- `Substrate` - Glass material type
- `Coating` - Anti-reflection coating type
- `Wavelength Range (nm)` - Coating wavelength range
- `Lens Type` - Classification (Plano, Convex, Aspheric)

## Integration with Existing Code

To use these lenses in your raytracing, you can load them like this:

```python
import pandas as pd
from scripts.PlanoConvex import PlanoConvex
from scripts.BiConvex import BiConvex

# Load lens data
df = pd.read_csv('data/Edmund-Optics_condenser-lenses.csv')

# Filter for desired lenses
df = df[df['Back Surface Shape'].str.strip() == 'Convex']  # Bi-convex only

# Create lens instance
row = df.iloc[0]
lens = BiConvex(
    vertex_z_front=position_z,
    R_front_mm=row['Radius R1 (mm)'],
    R_back_mm=row['Radius R2 (mm)'],
    center_thickness_mm=row['Center Thickness (mm)'],
    edge_thickness_mm=row['Edge Thickness (mm)'],
    ap_rad_mm=row['Diameter (mm)'] / 2.0
)
```

## Notes on Aspheric Lenses

The 8 aspheric lenses require a more sophisticated class that isn't included yet. They would need:
- Aspheric surface equation implementation
- Surface coefficients (not all available in Excel file)
- More complex intersection algorithms

For now, you can filter them out or approximate them as bi-convex lenses using the provided radii.

## Testing

Both the processing script and BiConvex class have been tested and verified to work correctly:
- Processing script successfully extracts all 102 lenses
- BiConvex class correctly traces rays through bi-convex lenses
- Output CSV format is compatible with existing data structures

## Next Steps

1. Integrate condenser lenses into your optimization routines
2. Test performance with different lens combinations
3. Consider implementing aspheric lens support if needed
4. Update selection criteria to handle bi-convex lenses

---

*Created: November 6, 2025*
