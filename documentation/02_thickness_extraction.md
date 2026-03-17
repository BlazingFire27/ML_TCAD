# Phase 2: Thickness Extraction — Documentation

**Date**: 2026-03-18  
**Author**: Vaiebhav (via Antigravity)

## What Was Done

Created `extract_thickness.py` — extracts oxide thickness from every spatial concentration profile in the new cleaned CSVs (with pressure).

## Why

The grey-box model needs a dataset of thickness vs. process conditions. Thickness must be computed accurately from the raw spatial profiles using sub-grid interpolation to avoid TCAD grid aliasing.

## How

For each `(file, Step(n))` group:
1. `logY = log10(clip(Y, 1e-9, ∞))`
2. Identify last point where `logY > 2.0` (oxide)
3. Sub-grid linear interpolation between last oxide point and first bulk point → `x_interface`
4. `x_surface = min(X)` (gas-oxide surface, always negative)
5. `Thickness = x_interface - x_surface`

## Results

| Metric | Value |
|--------|-------|
| Thicknesses extracted | 79,181 |
| Skipped (no oxide) | 0 |
| Skipped (bad data) | 0 |
| Errors | 0 |
| Runtime | 40.5 seconds |

### Thickness Statistics

| Stat | Value (μm) |
|------|-----------|
| Min | 0.00152 |
| Max | 0.13043 |
| Mean | 0.01895 |
| Median | 0.01260 |
| Std | 0.01909 |

### Parameter Ranges

| Parameter | Unique | Min | Max |
|-----------|--------|-----|-----|
| Pressure | 8 | 0.1563 | 0.6056 |
| O2 Flow | 10 | 0.6111 | 5.5895 |
| N2 Flow | 10 | 7.1126 | 24.2477 |
| Temperature | 10 | 800.59 | 1074.57 |
| Time | 10 | 38.01 | 348.77 |

### Interface Positions

- `X_surface` range: [−0.0737, −0.0015] (all negative — oxide extends above original Si surface)  
- `X_interface` range: [0.00001, 0.0567] (all positive — oxide extends into Si)

## Files Created

- **NEW**: `src/Vaiebhav/extract_thickness.py`
- **NEW**: `src/Vaiebhav/thickness_values.csv` (79,181 rows × 8 columns)
