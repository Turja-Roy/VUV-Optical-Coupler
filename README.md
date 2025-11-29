# VUV Flashlamp-to-Fiber Coupling Optimization

A comprehensive ray tracing and optimization framework for designing two-lens optical systems that efficiently couple vacuum ultraviolet (VUV) light from xenon flashlamp sources into optical fibers.

## Table of Contents

- [Project Overview](#project-overview)
  - [Key Results](#key-results)
- [Features](#features)
  - [Six Optimization Algorithms](#six-optimization-algorithms)
  - [Multi-Objective Optimization](#multi-objective-optimization)
  - [Additional Capabilities](#additional-capabilities)
- [Performance](#performance)
  - [Vectorized Ray Tracing](#vectorized-ray-tracing)
  - [Benchmark Your System](#benchmark-your-system)
- [Database Storage](#database-storage)
  - [Two-Database Architecture](#two-database-architecture)
  - [Managing the Lens Database](#managing-the-lens-database)
  - [SQLite Backend for Results](#sqlite-backend-for-results)
- [Web Dashboard](#web-dashboard)
  - [Starting the Dashboard](#starting-the-dashboard)
  - [Features](#features-1)
  - [Usage Examples](#usage-examples)
  - [API Endpoints](#api-endpoints)
- [Installation](#installation)
  - [Requirements](#requirements)
  - [Quick Start](#quick-start)
- [Configuration Files](#configuration-files)
  - [Using Preset Profiles](#using-preset-profiles)
  - [Using Custom Configuration Files](#custom-configuration)
- [Usage Guide](#usage-guide)
  - [Basic Commands](#basic-commands)
  - [Command Reference](#command-reference)
- [File Structure](#file-structure)
- [Output Files](#output-files)
  - [Results CSVs](#results-csvs)
  - [Plot Files](#plot-files)
  - [Log Files](#log-files)
- [Physics and Implementation](#physics-and-implementation)
  - [Ray Tracing](#ray-tracing)
  - [Atmospheric Absorption](#atmospheric-absorption)
  - [Multi-Objective Optimization](#multi-objective-optimization-1)
- [Performance Recommendations](#performance-recommendations)
- [Example Workflows](#example-workflows)
- [Troubleshooting](#troubleshooting)
- [Technical Background](#technical-background)
  - [Research Context](#research-context)
  - [Physical Constants](#physical-constants)
  - [Optimization Implementation](#optimization-implementation)
<!-- - [Citation](#citation) -->
- [Author](#author)
- [Contributing](#contributing)

## Project Overview

This project addresses the challenge of efficiently collecting 200nm light from a xenon flashlamp arc source and coupling it into a small-core optical fiber. The system uses:

- **Source**: 60W Xenon flash lamp with 3mm diameter arc in water-cooled jacket
- **Geometry**: Cooling jacket with 26mm optical path length limits beam divergence to 22.85°
- **Target**: 1mm core optical fiber with NA=0.22 and acceptance angle of 24.8°
- **Approach**: Two-lens plano-convex system optimization via deterministic ray tracing
- **Physics**: Full geometric optics with atmospheric O₂ absorption modeling, geometric losses from cooling jacket vignetting (43% transmission)

### Key Results

- **Coupling Efficiency**: 17-23% achievable with optimized configurations under current geometry constraints
- **System Length**: 40-100mm for high-performing configurations
- **Atmospheric Impact**: 16-24% coupling improvement in argon vs. air due to eliminated O₂ absorption at 200nm
- **Medium Dependence**: Argon shows significantly better performance than air due to lack of UV absorption
- **Constraint Impact**: Cooling jacket geometry (lenses must be positioned ≥27mm from source) significantly constrains optimization space

## Features

### Seven Optimization Algorithms

| Method | Speed | Convergence | Best For |
|--------|-------|-------------|----------|
| **Paraxial** ⚡⚡ | ~7ms | Fast screening | **Ultra-fast initial screening of thousands of combinations** |
| **Powell** ⚡ | ~0.1-0.2s | Good | **Quick optimization, everyday use** |
| **Nelder-Mead** ⚡ | ~0.1-0.2s | Good | **Fast local search** |
| Grid Search | ~0.2-0.3s | Systematic | Baseline comparison, small grids |
| Differential Evolution | ~1-2s | Excellent | Thorough global search when time permits |
| Bayesian | ~2-3s | Excellent | Sample-efficient exploration |
| Dual Annealing | ~4-5s | Excellent | Escaping local minima |

*Timings are per lens pair with 1000 rays using vectorized tracing on typical hardware. Paraxial uses analytical approximation (no ray tracing).*

**⚠️ Important Note on Paraxial Approximation**: The paraxial screening mode uses thin/thick lens equations with ABCD matrix propagation for ultra-fast evaluation (~20,503 combinations in ~3 minutes). However, it has known limitations:
- **Grid search limitations**: Uses coarse 5×5×5 grid (125 samples per pair) which may miss optimal configurations
- **Spacing assumptions**: Assumes lenses should be separated by ≥0.5× focal length, but optimal configurations may have much closer spacing
- **Best use case**: Initial screening to identify promising candidates, followed by full ray-trace optimization
- **Not a replacement**: Should be used to guide full optimization, not replace it

### Multi-Objective Optimization

Balance coupling efficiency against system compactness using the alpha parameter:

```
objective = α × (1 - coupling) + (1 - α) × (normalized_length)
```

- **α = 0.9**: Maximize coupling (90% weight) - laboratory setups
- **α = 0.7**: Balanced (default) - general purpose
- **α = 0.5**: Equal priority - compact designs

### Additional Capabilities

- **Paraxial Approximation**: Ultra-fast screening of all 20,503 lens combinations in ~3 minutes using analytical approximations
- **Tolerance Analysis**: Assess manufacturing sensitivity to longitudinal displacements (single-pair and batch modes)
- **Wavelength Analysis**: Study coupling efficiency across 180-300nm range
- **Web Dashboard**: Interactive browser-based results viewer with filtering and plotting
- **Medium Selection**: Air, argon, or helium propagation
- **Resume Support**: Automatic checkpoint/resume for interrupted batch runs
- **Rich Visualization**: Ray trace diagrams, spot diagrams, wavelength plots, tolerance curves
- **Batch Processing**: Automatic splitting and parallel processing of large lens catalogs
- **Dual Database Architecture**: Separate lens catalog (202 lenses) and optimization results databases

## Performance

### Vectorized Ray Tracing

The codebase uses **vectorized NumPy operations** for ray tracing, achieving **10-15x speedup** over serial implementations:

- **1,000 rays**: ~12ms (vectorized) vs ~120ms (serial) ⚡
- **10,000 rays**: ~19ms (vectorized) vs ~287ms (serial) ⚡

All optimization algorithms automatically use vectorized tracing. To disable (for debugging):

```yaml
# In your config file
rays:
  use_vectorized: false  # Default: true
```

**Implementation details**:
- Batch processing of all rays simultaneously using NumPy broadcasting
- Vectorized ray-sphere intersections, refraction calculations, and transmission factors
- Boolean masks to track ray success/failure through optical pipeline
- Located in `scripts/raytrace_helpers_vectorized.py`

### Benchmark Your System

```bash
python test_vectorization.py --rays 5000 --trials 5
```

Expected output:
```
Serial time:     0.045 ± 0.010 s
Vectorized time: 0.003 ± 0.001 s
Speedup:         13.9x
```

## Database Storage

The system uses two SQLite databases:
- **Lens catalog** (`data/lenses.db`): 202 lenses from ThorLabs and Edmund Optics
- **Optimization results** (`results/optimization.db`): Stores run metadata and results

### Managing Lenses

```bash
# Import lenses from CSV
python raytrace.py import-lenses

# List available lenses
python raytrace.py list-lenses --use-database

# Add new lenses
python -m scripts.add_lenses_from_csv new_lenses.csv --format thorlabs
```

### Querying Results

Enable database storage in `configs/default.yaml`:
```yaml
database:
  enabled: true
  path: ./results/optimization.db
```

Common queries:
```bash
# Show all runs
python -m scripts.db_query list-runs

# Find best results
python -m scripts.db_query best --limit 20 --min-coupling 0.20

# Export specific run
python -m scripts.db_query export-run 2025-10-24_coupling_0_21_air output.csv
```

## Web Dashboard

Interactive browser-based interface for viewing, filtering, and analyzing optimization results.

### Starting the Dashboard

```bash
# Auto-detect database or use CSV files
python raytrace.py dashboard

# Custom port
python raytrace.py dashboard --port 8080

# Specify database file
python raytrace.py dashboard --db results/integration_test.db

# Using config file
python raytrace.py dashboard --config my_config.yaml

# Config with CLI override
python raytrace.py dashboard --config my_config.yaml --port 5000
```

**Config file example** (`my_config.yaml`):
```yaml
dashboard:
  port: 8080
  db_path: results/optimization.db
  auto_open: false  # Future feature
```

The dashboard will start a local web server and display:
```
Dashboard running at http://localhost:5000
Press Ctrl+C to stop
```

Open the URL in your browser to access the interface.

### Features

**Statistics Overview**:
- Total optimization results
- Number of runs
- Best coupling efficiency achieved
- Average coupling across all results

**Interactive Filtering**:
- Filter by coupling efficiency range (min/max)
- Filter by medium (air, argon, helium)
- Filter by specific lens pairs
- Export filtered results as CSV

**Data Visualization**:
- Results table with sorting (click column headers)
- Coupling efficiency distribution histogram
- Lens pair comparison plots (compare different optimization methods)

**Data Sources**:
- Automatically reads from SQLite database if available
- Falls back to CSV files in `results/` directory
- Scans all subdirectories for optimization results
- Skips tolerance and wavelength analysis files

### Usage Examples

```bash
# View results from a specific database
python raytrace.py dashboard --db results/optimization.db

# Run on a different port (e.g., if 5000 is in use)
python raytrace.py dashboard --port 8888

# View CSV results without database
python raytrace.py dashboard  # Auto-detects and loads CSV files
```



## Installation

### Requirements

- Python 3.7+
- Required packages:
  ```bash
  pip install numpy scipy pandas matplotlib tqdm regex flask
  ```
- Optional (for Bayesian optimization):
  ```bash
  pip install scikit-optimize
  ```
- Optional (for web dashboard):
  ```bash
  pip install flask
  ```
  *Note: Flask is included in the required packages above*

### Quick Start

```bash
git clone <repository>
cd raytrace-xe-flashlamp-fiber
python raytrace.py particular LA4001 LA4647 --opt powell
```

## Configuration Files

All modes support YAML configuration files for parameter management. CLI arguments override config values.

### Using Preset Profiles

```bash
# Quick test (100 rays, Powell, fast)
python raytrace.py particular LA4001 LA4647 --profile quick_test

# Argon batch (1000 rays, differential evolution)
python raytrace.py combine --profile argon_batch

# Tolerance test (5000 rays, 41 samples, ±1mm)
python raytrace.py tolerance LA4001 LA4647 --profile tolerance_test
```

Available profiles in `configs/`: `quick_test`, `argon_batch`, `wavelength_study`, `tolerance_test`, `integration_test`

### Custom Configuration

Create `my_config.yaml`:

```yaml
rays:
  n_rays: 1500

medium:
  type: argon

optimization:
  method: powell

analyze:
  coupling_threshold: 0.22
  methods: [differential_evolution, powell]
```

Use it:
```bash
python raytrace.py particular LA4001 LA4647 --config my_config.yaml

# CLI overrides config
python raytrace.py particular LA4001 LA4647 --config my_config.yaml --medium air
```

See `configs/default.yaml` for all available options.

## Usage Guide

### Basic Commands

#### Screen All Lens Combinations with Paraxial Approximation (Ultra-Fast)

```bash
# Screen all 20,503 combinations in ~3 minutes (requires --use-database)
python raytrace.py paraxial --use-database --medium argon

# Results are filtered to coupling ≥ 1% and sorted by predicted coupling
# Outputs CSV with top candidates for full ray-trace optimization
```

**Note**: Paraxial approximation provides fast screening but has limitations (see warnings in Features section). Use results to guide full optimization, not as final values.

**Recommended workflow:**
```bash
# 1. Fast paraxial screening
python raytrace.py paraxial --use-database --medium argon

# 2. Review top candidates
head -20 results/paraxial_2025-11-19_argon/paraxial_results_2025-11-19.csv

# 3. Full optimization on top candidates
python raytrace.py particular 84-281 89-414 --opt differential_evolution --medium argon
```

#### Optimize a Specific Lens Pair (Recommended Starting Point)

```bash
# Fast optimization with Powell's method (recommended)
python raytrace.py particular LA4001 LA4647 --opt powell

# Thorough optimization with differential evolution (slower but more thorough)
python raytrace.py particular LA4001 LA4647 --opt differential_evolution

# Try in argon (no UV absorption)
python raytrace.py particular LA4001 LA4647 --opt powell --medium argon
```

#### Optimize All Combinations

```bash
# Strategic scan with pre-selected L1/L2 candidates (924 combinations, recommended)
python raytrace.py select --opt powell

# Exhaustive scan of all possible combinations (24,336 combinations)
python raytrace.py combine --opt powell

# Prioritize coupling efficiency (90% weight)
python raytrace.py select --opt powell --alpha 0.9

# Prioritize compact systems (50/50 weight)
python raytrace.py select --opt powell --alpha 0.5
```

**Select vs Combine modes:**
- **select**: Tests 21 L1 candidates × 44 L2 candidates = 924 strategically chosen lens pairs
- **combine**: Tests all 156 lenses × 156 lenses = 24,336 possible combinations (26× more)
- **Recommendation**: Use `select` for efficient exploration; use `combine` only when exhaustive search is needed

#### Compare All Methods

```bash
# Test all 6 optimization methods on a single lens pair
python raytrace.py compare LA4001 LA4647
```

This runs all available methods and reports:
- Coupling efficiency achieved
- Total system length
- Computation time
- Best method recommendation

#### Resume Interrupted Batch Runs

```bash
# Resume from checkpoint
python raytrace.py combine --opt powell continue 2025-10-14
```

#### Analyze High-Coupling Results

Re-optimize promising results with multiple methods to find absolute best configurations.

```bash
# CLI-only usage
python raytrace.py analyze \
  --results-file results/2025-10-17/results_combine_*.csv \
  --coupling-threshold 0.20

# Using config file to specify methods and parameters
python raytrace.py analyze \
  --config my_config.yaml \
  --results-file results/2025-10-17/results_combine_*.csv

# Config with CLI override
python raytrace.py analyze \
  --config my_config.yaml \
  --results-file results/2025-10-17/results_combine_*.csv \
  --coupling-threshold 0.18 \
  --n-rays 2000
```

**Config file example** (`my_config.yaml`):
```yaml
analyze:
  n_rays: 1000
  coupling_threshold: 0.22
  methods:
    - differential_evolution
    - dual_annealing
    - powell
```

The `methods` list controls which optimizers to test on each lens pair. Results are saved separately for each method and combined into a single best-of-all file.

#### Wavelength Dependence Analysis

Study how coupling varies with wavelength.

```bash
# CLI-only usage
python raytrace.py wavelength-analyze \
  --results-file results/2025-10-17/36-681+LA4647.csv \
  --wl-start 180 --wl-end 300 --wl-step 10

# Using config file
python raytrace.py wavelength-analyze \
  --config my_config.yaml \
  --results-file results/2025-10-17/36-681+LA4647.csv

# Config with CLI override for wavelength range
python raytrace.py wavelength-analyze \
  --config my_config.yaml \
  --results-file results/2025-10-17/36-681+LA4647.csv \
  --wl-start 190 --wl-end 210 --wl-step 2
```

**Config file example** (`my_config.yaml`):
```yaml
wavelength:
  wl_start: 180
  wl_end: 300
  wl_step: 10
  n_rays: 2000
  methods:
    - differential_evolution
    - powell
```

The `methods` list controls which optimizers calibrate the geometry at 200nm before wavelength sweeping.

**Plotting results:**
```bash
# Generate plots from wavelength analysis
python raytrace.py wavelength-analyze-plot \
  --results-dir results/wavelength_analyze_2025-10-18

# With polynomial curve fitting
python raytrace.py wavelength-analyze-plot \
  --results-dir results/wavelength_analyze_2025-10-18 \
  --fit polynomial

# Aggregated plots with error bars
python raytrace.py wavelength-analyze-plot \
  --results-dir results/wavelength_analyze_2025-10-18 \
  --aggregate
```

#### Tolerance Analysis

Assess manufacturing sensitivity by analyzing how coupling efficiency varies with small displacements in lens positions.

##### Single Lens Pair Mode

```bash
# Analyze tolerance for a specific lens pair
python raytrace.py tolerance LA4001 LA4647 --opt powell

# Use high-resolution tolerance profile from config
python raytrace.py tolerance LA4001 LA4647 --profile tolerance_test

# Custom tolerance parameters
python raytrace.py tolerance LA4001 LA4647 \
  --z-range 1.0 \
  --n-samples 41 \
  --n-rays 5000

# Test in argon environment
python raytrace.py tolerance LA4001 LA4647 --opt powell --medium argon
```

##### Batch Mode

Analyze tolerance for multiple lens pairs from optimization results:

```bash
# Batch tolerance analysis on high-coupling results
python raytrace.py tolerance \
  --results-file results/2025-11-07_combine_powell_argon/batch_combine_1.csv \
  --coupling-threshold 0.35 \
  --z-range 1.0 \
  --n-samples 41 \
  --n-rays 5000 \
  --medium argon \
  --use-database

# Use config file with defaults (results_file and coupling_threshold from config)
python raytrace.py tolerance --profile tolerance_test --use-database

# Or override specific config values via CLI
python raytrace.py tolerance \
  --profile tolerance_test \
  --coupling-threshold 0.35 \
  --use-database
```

**Batch mode benefits:**
- Test tolerance on multiple optimized configurations at once
- Compare tolerance characteristics across different lens pairs
- Identify most manufacturable designs (highest tolerance to misalignment)
- Uses pre-optimized positions from results file (no re-optimization needed)

**What it does**:
- First optimizes the lens pair to find best configuration
- Systematically perturbs each lens position (z_l1, z_l2) independently
- Measures coupling efficiency at each position
- Generates tolerance curves showing sensitivity to misalignment

**Output files (single-pair mode)**:
- `tolerance_L1_<lens1>+<lens2>.csv` - L1 position sweep data
- `tolerance_L2_<lens1>+<lens2>.csv` - L2 position sweep data
- `tolerance_summary_<lens1>+<lens2>.csv` - Summary statistics
- `tolerance_<lens1>+<lens2>.png` - Tolerance curve plot

**Output files (batch mode)**:
- `tolerance_L1_<lens1>+<lens2>.csv` - L1 sweep data for each pair
- `tolerance_L2_<lens1>+<lens2>.csv` - L2 sweep data for each pair
- `tolerance_summary_<lens1>+<lens2>.csv` - Individual pair summary
- `tolerance_batch_summary.csv` - Comparison across all pairs
- `tolerance_batch_comparison.png` - Bar chart comparing all configurations

**Configuration options** (in YAML config files):
```yaml
tolerance:
  z_range_mm: 1.0      # ±1.0mm range around optimal position
  n_samples: 41        # Number of positions to test (41 = 0.05mm steps)
  n_rays: 5000         # Rays per position (higher = more accurate)

batch_tolerance:
  results_file: results/combine_*/batch_combine_1.csv  # Results CSV for batch mode
  coupling_threshold: 0.35    # Minimum coupling to include in batch analysis
```

**Interpreting results**:
- Steep slopes indicate high sensitivity to that lens position
- Flat regions indicate tolerance to small displacements
- Asymmetric curves reveal directional sensitivity
- Use results to set manufacturing tolerances

**Example workflow**:
```bash
# 1. Find best configuration
python raytrace.py particular LA4001 LA4647 --opt differential_evolution

# 2. Analyze tolerance with high resolution
python raytrace.py tolerance LA4001 LA4647 --profile tolerance_test

# 3. Review tolerance curves
ls plots/*tolerance*/tolerance_LA4001+LA4647.png

# 4. Check numerical results
cat results/*tolerance*/tolerance_summary_LA4001+LA4647.csv
```

### Command Reference

```
Commands:
  particular <lens1> <lens2>    Optimize specific lens pair
  compare <lens1> <lens2>       Compare all optimization methods
  select                        Optimize L1×L2 candidate combinations (924 pairs, recommended)
  combine                       Optimize all lens combinations (24,336 pairs, exhaustive)
  paraxial                      Ultra-fast screening of all 20,503 combinations using analytical approximation (~3min)
  analyze                       Re-analyze high-coupling results
  wavelength-analyze            Study wavelength dependence
  wavelength-analyze-plot       Create plots from wavelength data
  tolerance <lens1> <lens2>     Analyze manufacturing tolerance sensitivity
  dashboard                     Start interactive web dashboard
  import-lenses                 Import lenses from CSV files into database
  list-lenses                   List available lenses from database or CSV

Options:
  --opt <method>                Optimization method (default: powell)
                                Options: powell, nelder_mead, grid_search,
                                         differential_evolution, bayesian,
                                         dual_annealing
  --alpha <0-1>                 Coupling vs. length trade-off (default: 0.7)
  --medium <type>               Propagation medium (default: air)
                                Options: air, argon, helium
  --n-rays <count>              Number of rays per trace (default: 1000)
  --wl-start <nm>               Wavelength analysis start (default: 180)
  --wl-end <nm>                 Wavelength analysis end (default: 300)
  --wl-step <nm>                Wavelength analysis step (default: 10)
  --z-range <mm>                Tolerance analysis range (default: 0.5)
  --n-samples <count>           Tolerance analysis samples (default: 21)
  --coupling-threshold <value>  Minimum coupling for analyze mode
  --results-file <path>         Input CSV file for analyze/wavelength modes
  --results-dir <path>          Results directory for plotting
  --fit <type>                  Curve fit type: polynomial, spline
  --aggregate                   Generate aggregated plots with error bars
  --port <number>               Dashboard port (default: 5000)
  --db <path>                   Database file for dashboard
  --profile <name>              Use preset configuration profile
  --config <path>               Use custom YAML configuration file
  continue                      Resume incomplete batch run
  <YYYY-MM-DD>                  Specify run date for continue mode
```

## File Structure

```
raytrace-xe-flashlamp-fiber/
├── raytrace.py                    # Main entry point
│
├── scripts/
│   ├── optimization/              # Optimization algorithms
│   │   ├── __init__.py
│   │   ├── powell.py              # Powell's method (fast)
│   │   ├── nelder_mead.py         # Nelder-Mead simplex (fast)
│   │   ├── grid_search.py         # Systematic grid search
│   │   ├── differential_evolution.py  # Global optimizer (thorough)
│   │   ├── bayesian.py            # Bayesian optimization
│   │   ├── dual_annealing.py      # Simulated annealing
│   │   ├── optimization_runner.py # Unified interface
│   │   └── fiber_position_optimizer.py  # Fiber position optimization helper
│   │
│   ├── analysis.py                # Analysis and wavelength studies
│   ├── Aspheric.py                # Aspheric lens class
│   ├── BiConvex.py                # Bi-convex lens class
│   ├── calcs.py                   # Physics calculations (refraction, absorption)
│   ├── cli.py                     # Command-line interface
│   ├── config_loader.py           # YAML configuration file handler
│   ├── consts.py                  # Physical constants and system parameters
│   ├── data_io.py                 # File I/O and result management
│   ├── database.py                # SQLite database backend (optimization results)
│   ├── db_query.py                # Database query CLI
│   ├── hitran_data.py             # HITRAN absorption data handling
│   ├── lens_database.py           # Lens catalog database management
│   ├── lens_factory.py            # Lens object factory
│   ├── paraxial_approximation.py  # Ultra-fast paraxial screening (~7ms/pair)
│   ├── PlanoConvex.py             # Plano-convex lens class
│   ├── raytrace_helpers.py        # Core ray tracing functions
│   ├── raytrace_helpers_vectorized.py  # Vectorized ray tracing (10-15x faster)
│   ├── runner.py                  # Batch processing and continuation
│   ├── tolerance_analysis.py      # Manufacturing tolerance testing
│   ├── visualizers.py             # Plotting and visualization
│   └── web_dashboard.py           # Flask-based interactive web dashboard
│
├── configs/                       # YAML configuration files
│   ├── default.yaml               # Default parameters
│   ├── quick_test.yaml            # Fast testing profile
│   ├── argon_batch.yaml           # Argon medium batch profile
│   ├── wavelength_study.yaml      # Wavelength analysis profile
│   ├── tolerance_test.yaml        # High-resolution tolerance profile
│   └── integration_test.yaml      # Database integration testing
│
├── data/                          # Lens catalogs
│   ├── lenses.db                  # SQLite lens catalog database (202 lenses)
│   ├── Combined_Lenses.csv        # Full lens catalog CSV (legacy, 156 lenses)
│   ├── l1_candidates.csv          # Candidate L1 lenses
│   ├── l1_candidates_ext.csv      # Extended L1 candidates
│   ├── l2_candidates.csv          # Candidate L2 lenses
│   ├── l2_candidates_ext.csv      # Extended L2 candidates
│   └── *.csv, *.xlsx              # Vendor lens data (ThorLabs, Edmund Optics)
│
├── results/                       # Output directory
│   ├── <run_id>/                  # Results organized by run ID
│   │   ├── batch_*.csv            # Intermediate batch results
│   │   ├── results_*.csv          # Final combined results
│   │   ├── tolerance_*.csv        # Tolerance analysis data
│   │   └── temp_*.json            # Checkpoint files
│   └── *.db                       # SQLite database files
│
├── plots/                         # Generated plots
│   └── <run_id>/
│       ├── C-<coupling>_L1-*_L2-*.png      # Ray trace diagrams
│       ├── spot_*.png                       # Spot diagrams
│       ├── tolerance_*.png                  # Tolerance curves
│       └── wavelength_analyze_*/            # Wavelength analysis plots
│
├── logs/                          # Execution logs
│   └── *.log                      # Detailed execution logs
│
└── doc/                           # Documentation
    ├── technical_report.pdf       # Full technical report
    └── *.tex, *.bib               # LaTeX source files
```



## Output Files

### Results CSVs

All optimization results are saved as CSV files with columns:

- `lens1`, `lens2`: Lens identifiers
- `f1_mm`, `f2_mm`: Focal lengths (mm)
- `z_l1`, `z_l2`, `z_fiber`: Positions along optical axis (mm)
- `total_len_mm`: Total system length (mm)
- `coupling`: Coupling efficiency (0-1)

Example: `results/2025-10-18_combine_powell_air/results_combine_2025-10-18.csv`

### Plot Files

#### Ray Trace Diagrams
Shows full ray paths through the optical system with lens positions and fiber location.

Filename format: `C-<coupling>_L1-<lens1>_L2-<lens2>.png`

#### Spot Diagrams
Shows ray impact positions at the fiber face, indicating coupling quality.

Filename format: `spot_C-<coupling>_L1-<lens1>_L2-<lens2>.png`

#### Wavelength Analysis Plots
- Per-lens plots: Coupling vs. wavelength for each lens combination
- Per-method plots: Comparing all lens pairs for each optimization method
- Aggregated plots: Mean curves with error bars across multiple runs

### Log Files

Detailed execution logs including:
- Timestamp for each optimization
- Method being used
- Coupling efficiency and system length
- Computation time
- Error messages and warnings

Example: `logs/2025-10-18_combine_powell_air.log`

## Physics and Implementation

Uses deterministic ray tracing with atmospheric absorption modeling (O₂ absorption in air). Multi-objective optimization balances coupling efficiency against system length using the `alpha` parameter. See `doc/technical_report.pdf` for detailed physics and implementation.

## Performance Recommendations

### For Ultra-Fast Initial Screening
**Use Paraxial Approximation** - Screen all 20,503 combinations in ~3 minutes:
```bash
python raytrace.py paraxial --use-database --medium argon
# Then full ray-trace on top candidates
```

### For Everyday Use
**Use Powell's method** - Fast and reliable:
```bash
python raytrace.py combine --opt powell
```

### For Best Possible Results
**Use Differential Evolution** - Slower but more thorough:
```bash
python raytrace.py combine --opt differential_evolution
```

### For Method Comparison
**Use compare mode** - Test all methods on a sample:
```bash
python raytrace.py compare LA4001 LA4647
```

### For Large Catalogs (Recommended Workflow)
**Multi-stage approach for best results**:
```bash
# Stage 1: Ultra-fast paraxial screening (~3 min for all 20,503 pairs)
python raytrace.py paraxial --use-database --medium argon

# Stage 2: Fast ray-trace on top 100 from paraxial (~10 min with Powell)
# Extract top lens pairs from paraxial results, then:
python raytrace.py particular <lens1> <lens2> --opt powell --medium argon

# Stage 3: Thorough optimization on top 20 (~3-5 min with DE)
python raytrace.py particular <lens1> <lens2> --opt differential_evolution --medium argon

# Total time: ~15-20 minutes vs. ~8 hours for full DE on all pairs
```

### Alternative: Traditional Approach (No Paraxial)
**Start with Powell, verify with DE**:
```bash
# 1. Fast scan with Powell (~40 min for all 24,336 pairs)
python raytrace.py combine --opt powell

# 2. Re-optimize top candidates with differential evolution
python raytrace.py analyze \
  --results-file results/2025-10-18/results_*.csv \
  --coupling-threshold 0.20 \
  --opt differential_evolution
```

## Example Workflows

### Workflow 1: Ultra-Fast Screening with Paraxial Approximation

**Best for**: Quickly identifying promising lens pairs from full catalog

```bash
# Step 1: Fast paraxial screening of all 20,503 combinations (~3 minutes)
python raytrace.py paraxial --use-database --medium argon

# Step 2: Review top candidates
head -30 results/paraxial_2025-11-19_argon/paraxial_results_2025-11-19.csv

# Step 3: Full ray-trace optimization on top 10 candidates
# Example: Top result is 84-281 + 89-414 with predicted 35.5% coupling
python raytrace.py particular 84-281 89-414 --opt differential_evolution --medium argon

# Step 4: Batch optimize multiple promising pairs
for pair in "84-281 89-414" "48-278 LA4194" "84-282 89-411"; do
    python raytrace.py particular $pair --opt differential_evolution --medium argon
done

# Step 5: Compare results
grep "Coupling=" logs/particular_*_differential_evolution_argon.log

# Step 6: Tolerance analysis on best performer
python raytrace.py tolerance 84-281 89-414 --profile tolerance_test --medium argon
```

**Important Notes**:
- Paraxial provides screening only - actual coupling may differ significantly
- Some lens pairs may perform better than paraxial predicts (e.g., if optimal spacing is closer than paraxial assumes)
- Always validate with full ray tracing before finalizing design

### Workflow 2: Find Best Lens Pair for Maximum Coupling

```bash
# Step 1: Quick scan prioritizing coupling
python raytrace.py combine --opt powell --alpha 0.9

# Step 2: Re-optimize top candidates thoroughly
python raytrace.py analyze \
  --results-file results/2025-10-18_combine_powell_air/results_combine_2025-10-18.csv \
  --coupling-threshold 0.22

# Step 3: View results interactively
python raytrace.py dashboard
```



### Workflow 3: Manufacturing Tolerance Assessment

```bash
# Step 1: Optimize a promising lens pair
python raytrace.py particular LA4001 LA4647 --opt differential_evolution

# Step 2: Run high-resolution tolerance analysis
python raytrace.py tolerance LA4001 LA4647 --profile tolerance_test

# Step 3: View tolerance curves
ls plots/2025-10-24_None_powell_air/tolerance_LA4001+LA4647.png

# Step 4: Check tolerance summary statistics
cat results/2025-10-24_None_powell_air/tolerance_summary_LA4001+LA4647.csv

# Step 5: Compare tolerances in different media
python raytrace.py tolerance LA4001 LA4647 --opt powell --medium argon \
  --z-range 1.0 --n-samples 41
```

### Workflow 4: Interactive Results Analysis with Dashboard

```bash
# Step 1: Run batch optimization with database enabled
# (Ensure database.enabled: true in configs/default.yaml)
python raytrace.py select --opt powell --alpha 0.9

# Step 2: Start the dashboard with config
python raytrace.py dashboard --config dashboard_config.yaml

# Step 3: Open browser to http://localhost:8080 (or configured port)
# - Filter results by coupling > 0.22
# - View coupling distribution histogram
# - Compare different lens pairs
# - Export filtered results as CSV

# Alternative: Use CSV files directly (no database)
python raytrace.py dashboard  # Auto-detects CSV files

# Step 4: Export high-performing results via API
curl "http://localhost:8080/api/export?min_coupling=0.22" -o best_results.csv
```

**dashboard_config.yaml:**
```yaml
dashboard:
  port: 8080
  db_path: results/optimization.db
```



## Troubleshooting

### Slow Performance

**Problem**: Optimization takes too long  
**Solutions**:
- Use Powell or Nelder-Mead instead of Differential Evolution
- Reduce number of rays: `--n-rays 500`
- Process specific lens pairs instead of full catalog

### Poor Coupling Results

**Problem**: Low coupling efficiency  
**Solutions**:
- Try different optimization method: `--opt differential_evolution`
- Increase alpha to prioritize coupling: `--alpha 0.9`
- Verify lens data in `data/Combined_Lenses.csv`
- Check if lenses are physically compatible

### Import Errors

**Problem**: `ModuleNotFoundError: No module named 'skopt'`  
**Solution**: Install scikit-optimize or use different method:
```bash
pip install scikit-optimize
# OR
python raytrace.py combine --opt powell
```

### Interrupted Batch Runs

**Problem**: Run stopped midway through catalog  
**Solution**: Resume from checkpoint:
```bash
python raytrace.py combine --opt powell continue 2025-10-18
```

### Memory Issues

**Problem**: Out of memory with large catalogs  
**Solutions**:
- Batch processing is automatic for >100 combinations
- Reduce rays per trace: `--n-rays 500`
- Process smaller subsets using `particular` mode

### Paraxial Results Don't Match Full Ray Tracing

**Problem**: Lens pair performs well in full ray tracing but missing or poor in paraxial results  
**Root Causes**:
1. **Grid search limitations**: Paraxial uses 5×5×5 coarse grid (125 samples) which may miss optimal positions
2. **Conservative spacing assumptions**: Assumes z_l2 ≥ z_l1 + tc + 0.5×f1, but optimal may be much closer
3. **Same-lens pairs**: May require tighter spacing than paraxial grid searches

**Example**: 48-274 + 48-274 achieves 17-23% in full ray tracing (z_l1=27mm, z_l2=33mm) but paraxial predicts <1% because the grid doesn't search z_l2 < 58mm

**Solutions**:
- Use paraxial for initial screening only, not final values
- Always validate promising pairs with full ray-trace optimization
- For specific pairs of interest, run full optimization directly with `particular` mode
- Consider paraxial results as conservative lower bounds

**Recommended workflow**: Paraxial screening → Full optimization of top 20-50 candidates → Tolerance analysis of best results

## Technical Background

This project optimizes two-lens optical systems for coupling 200nm VUV light from xenon flashlamps into optical fibers. For research context, physical constants, optimization algorithms, and implementation details, see `doc/technical_report.pdf`.

<!-- ## Citation -->
<!---->
<!-- If you use this code in your research, please cite: -->
<!---->
<!-- ``` -->
<!-- Turja Roy, "Optimization of Two-Lens Coupling Systems for VUV Flashlamp  -->
<!-- to Fiber Applications Using Ray Tracing and Multi-Algorithm Comparison", -->
<!-- University of Texas at Arlington (2025) -->
<!-- ``` -->
<!---->
<!-- Technical report: `doc/technical_report.pdf` -->

## Author

**Turja Roy**  
Department of Physics  
University of Texas at Arlington

## Contributing

For questions, issues, or contributions, please contact the author or open an issue in the repository.
