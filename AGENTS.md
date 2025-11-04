# Agent Guidelines for raytrace-xe-flashlamp-fiber

## Build/Test/Run Commands
- **Run main script**: `python raytrace.py <command> [options]` (see README.md for full CLI)
- **Quick test**: `python raytrace.py particular LA4001 LA4647 --opt powell`
- **No formal test suite**: Verify changes by running specific lens pairs with known expected results
- **No linting configured**: Follow existing code style (see below)

## Code Style

**Imports**: Standard library first, then third-party (numpy, scipy, pandas, matplotlib), then local modules (`from scripts import`)

**Formatting**: 4-space indentation, no trailing whitespace, Unix line endings (LF)

**Types**: No type hints used in this codebase (pure Python 3.7+ compatible)

**Naming**: `snake_case` for functions/variables, `PascalCase` for classes, `UPPER_CASE` for constants (see `scripts/consts.py`)

**Comments**: Minimal inline comments; rely on docstrings for function documentation

**Error handling**: Direct returns or None for failures; no exceptions raised in ray tracing code

## Project-Specific Conventions
- Physical units always in mm, nm, radians (not degrees except in input constants)
- All optimization functions return dict with keys: `lens1`, `lens2`, `z_l1`, `z_l2`, `z_fiber`, `coupling`, `total_len_mm`, `f1_mm`, `f2_mm`, `origins`, `dirs`, `accepted`
- Ray tracing uses deterministic stratified sampling (not pure Monte Carlo despite legacy naming)
- Medium parameter: `'air'`, `'argon'`, or `'helium'` (lowercase string)
