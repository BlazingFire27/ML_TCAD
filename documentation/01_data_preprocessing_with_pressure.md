# Phase 1: Data Preprocessing with Pressure — Documentation

**Date**: 2026-03-18  
**Author**: Vaiebhav (via Antigravity)

## What Was Done

Created `data_preprocessing_v2.py` — a new preprocessing script that extracts the **Pressure** field from raw TCAD column headers and includes it in the output CSVs.

## Why

The original `data_preprocessing.py` parsed the `Pres_` field from column headers but **dropped it** from the output. Data exploration revealed **8 distinct pressure values** (0.156–0.606 atm), not a single fixed value as previously assumed. Pressure is a critical physical variable that directly affects oxidation kinetics through the Deal-Grove model's partial pressure calculation.

## How

- **Regex**: `n(\d+)_Pres_([\d\.]+)_O2_([\d\.]+)_N2_([\d\.]+)_Temp_([\d\.]+)_time_([\d\.]+)`
- **Data sources**: Both `data/old/` (oxi3–11) and `data/23rdJan2026/` (Oxi_12–90)
- **Output format**: `Step (n), Pressure, O2 Flow, N2 Flow, Temperature, Time, X, Y`
- **Output location**: `data/new processed (with pressure)/`
- **Original data**: Untouched in `data/processed/`

## Results

| Metric | Value |
|--------|-------|
| Groups processed | 88 (oxi3 through oxi90) |
| Groups skipped | 0 |
| Total rows | 16,571,846 |
| Raw files read | 108 |
| Unique pressures | 8: [0.156, 0.194, 0.225, 0.434, 0.445, 0.490, 0.536, 0.606] atm |

### Pressure distribution across oxi groups

| Pressure (atm) | Oxi groups |
|-----------------|------------|
| 0.444858 | oxi3–13 (old data + first new set) |
| 0.490281 | oxi14–25 |
| 0.605645 | oxi25–36 (oxi25 has both 0.490 and 0.606) |
| 0.433543 | oxi36–47 (oxi36 has both 0.434 and 0.606) |
| 0.535863 | oxi47–58 (oxi47 has both 0.434 and 0.536) |
| 0.193821 | oxi58–69 (oxi58 has both 0.194 and 0.536) |
| 0.224535 | oxi69–80 (oxi69 has both 0.194 and 0.225) |
| 0.156258 | oxi80–90 (oxi80 has both 0.225 and 0.156) |

> **Note**: Some boundary oxi files (25, 36, 47, 58, 69, 80) contain data from **two** pressure regimes — this indicates the TCAD simulation transitioned between pressure settings within those runs.

## Files Created/Modified

- **NEW**: `src/Vaiebhav/data_preprocessing_v2.py`
- **NEW**: 88 files in `data/new processed (with pressure)/Cleaned_oxi{3..90}.csv`
- **NO** modifications to original `data/processed/` or any existing code

## Issues

None — all 88 groups processed cleanly with 0 errors.
