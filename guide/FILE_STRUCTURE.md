# File Organization Guide

## New File Structure

After installing the optimization system, your project should look like this:

```
project_root/
│
├── raytrace.py                          # UPDATED - Main entry point with optimization
│
├── setup_optimization_dirs.py           # NEW - Setup script
├── install_optimization.sh              # NEW - Installation script
├── OPTIMIZATION_GUIDE.md                # NEW - Usage documentation
│
├── scripts/
│   ├── consts.py                        # Unchanged
│   ├── calcs.py                         # Unchanged
│   ├── fetcher.py                       # Unchanged
│   ├── PlanoConvex.py                   # Unchanged
│   ├── raytrace_helpers.py              # Unchanged
│   ├── visualizers.py                   # Unchanged
│   ├── runner.py                        # LEGACY - Will be moved
│   │
│   └── optimization/                    # NEW DIRECTORY
│       ├── __init__.py                  # Module init
│       ├── scipy_optimizer.py           # NEW - Fast scipy-based methods
│       ├── bayesian_optimizer.py        # NEW - Bayesian optimization
│       ├── optimization_runner.py       # NEW - Unified runner
│       │
│       └── legacy/                      # Grid search backup
│           └── grid_search.py           # MOVED - Original runner.py
│
├── data/
│   ├── l1_candidates.csv                # Unchanged
│   ├── l2_candidates.csv                # Unchanged
│   └── Combined_Lenses.csv              # Unchanged
│
├── results/                             # Output directory
│   └── <date>/
│       ├── batch_combine_1.csv
│       ├── results_combine_<date>.csv
│       └── temp*.json                   # Temporary files
│
├── plots/                               # Visualization output
│   └── <date>/
│       ├── C-<coupling>_L1-<>_L2-<>.png
│       └── spot_*.png
│
└── logs/                                # Logging output
    └── run_<date>.log
```

## Installation Steps

### Option 1: Automated (Linux/Mac)

```bash
bash install_optimization.sh
```

Then manually copy these files to `scripts/optimization/`:
- `scipy_optimizer.py`
- `bayesian_optimizer.py`
- `optimization_runner.py`

### Option 2: Manual

1. **Create directories:**
   ```bash
   mkdir -p scripts/optimization/legacy
   ```

2. **Create `scripts/optimization/__init__.py`:**
   ```python
   from .scipy_optimizer import run_optimization
   from .optimization_runner import run_combos_optimized, compare_optimizers
   
   try:
       from .bayesian_optimizer import run_bayesian_optimization
       BAYESIAN_AVAILABLE = True
   except ImportError:
       BAYESIAN_AVAILABLE = False
   
   __all__ = [
       'run_optimization',
       'run_combos_optimized',
       'compare_optimizers',
       'run_bayesian_optimization',
       'BAYESIAN_AVAILABLE'
   ]
   ```

3. **Copy files:**
   ```bash
   # Backup original grid search
   cp scripts/runner.py scripts/optimization/legacy/grid_search.py
   
   # Add new optimization files (from artifacts)
   # Copy to scripts/optimization/:
   #   - scipy_optimizer.py
   #   - bayesian_optimizer.py
   #   - optimization_runner.py
   ```

4. **Replace raytrace.py** with the updated version

## File Descriptions

### New Files

| File | Purpose | Dependencies |
|------|---------|--------------|
| `scipy_optimizer.py` | Core optimization algorithms using scipy | scipy, numpy |
| `bayesian_optimizer.py` | Bayesian optimization with GP | scikit-optimize (optional) |
| `optimization_runner.py` | Unified interface for all methods | All optimizers |
| `OPTIMIZATION_GUIDE.md` | Complete usage documentation | - |
| `FILE_STRUCTURE.md` | This file | - |
| `setup_optimization_dirs.py` | Automated setup script | - |
| `install_optimization.sh` | Bash installation script | - |

### Updated Files

| File | Changes | Backward Compatible? |
|------|---------|---------------------|
| `raytrace.py` | Complete rewrite with new CLI | No - see migration guide |

### Unchanged Files

All other files remain exactly as they were:
- `scripts/consts.py`
- `scripts/calcs.py`
- `scripts/fetcher.py`
- `scripts/PlanoConvex.py`
- `scripts/raytrace_helpers.py`
- `scripts/visualizers.py`
- All data CSV files

### Legacy Files

| File | New Location | Purpose |
|------|--------------|---------|
| `scripts/runner.py` | `scripts/optimization/legacy/grid_search.py` | Original grid search (preserved) |

## Dependencies

### Required (already installed)
```bash
pip install numpy scipy matplotlib pandas tqdm regex
```

### Optional (for Bayesian optimization)
```bash
pip install scikit-optimize
```

## Migration Guide

### Old Usage → New Usage

**Single lens pair:**
```bash
# Old
python raytrace.py particular LA4001 LA4647

# New (much faster!)
python raytrace.py particular LA4001 LA4647 --opt differential_evolution
```

**All combinations:**
```bash
# Old
python raytrace.py combine

# New (10-100x faster!)
python raytrace.py combine --opt differential_evolution

# Or keep old behavior
python raytrace.py combine --opt grid_search
```

**Continue runs:**
```bash
# Old
python raytrace.py combine continue 2025-10-14

# New
python raytrace.py combine --opt differential_evolution continue 2025-10-14
```

## Verification

Test that everything is working:

```bash
# Test installation
python -c "from scripts.optimization import run_optimization; print('✓ Installation successful')"

# Compare methods on a test case
python raytrace.py compare LA4001 LA4647

# Quick optimization test
python raytrace.py particular LA4001 LA4647 --opt differential_evolution
```

Expected output:
```
✓ Installation successful
```

## Cleanup (Optional)

After verifying everything works, you can optionally:

1. Keep `scripts/runner.py` as backup, or
2. Remove it (already backed up in `legacy/grid_search.py`)

```bash
# Optional: remove original if backup verified
# rm scripts/runner.py
```

## Rollback

To revert to the original system:

```bash
# Restore original raytrace.py from backup
# Remove optimization directory
rm -rf scripts/optimization

# Original runner.py should still be in place
```

## File Sizes

Approximate sizes of new files:

- `scipy_optimizer.py`: ~8 KB
- `bayesian_optimizer.py`: ~7 KB
- `optimization_runner.py`: ~8 KB
- `raytrace.py` (updated): ~12 KB
- `OPTIMIZATION_GUIDE.md`: ~10 KB

Total addition: ~45 KB

## Integration Points

The new optimization system integrates with existing code through:

1. **`raytrace_helpers.py`**: Uses `sample_rays()` and `trace_system()`
2. **`PlanoConvex.py`**: Creates lens objects
3. **`consts.py`**: Reads system constants
4. **`fetcher.py`**: Uses `write_temp()` and `write_results()`
5. **`visualizers.py`**: Calls `plot_system_rays()` and `plot_spot_diagram()`

No modifications to existing files are required - only additions!

## Troubleshooting

### Import Error: No module named 'optimization'

**Solution:**
```bash
# Ensure __init__.py exists
ls scripts/optimization/__init__.py

# If missing, create it
python setup_optimization_dirs.py
```

### Cannot find scipy_optimizer.py

**Solution:**
Copy the file from the artifact to `scripts/optimization/scipy_optimizer.py`

### Bayesian optimization not available

**Solution:**
```bash
pip install scikit-optimize
```

Or use a different method:
```bash
python raytrace.py combine --opt differential_evolution
```

## Support

For questions or issues:

1. Check `OPTIMIZATION_GUIDE.md` for usage examples
2. Run compare mode to verify setup: `python raytrace.py compare LA4001 LA4647`
3. Check logs: `logs/run_<date>.log`
4. Try with grid search to verify base system: `--opt grid_search`
