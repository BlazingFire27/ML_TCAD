# Phase 3: 44% Rule Validation — Documentation

**Date**: 2026-03-18  
**Author**: Vaiebhav (via Antigravity)

## What Was Done

Validated the 44/56 oxide growth partition rule against extracted thickness data.

## Physics

When SiO2 grows from Si:
- Si molar volume: V_Si = M_Si / ρ_Si = 28.09 / 2.33 = 12.06 cm³/mol
- SiO2 molar volume: V_SiO2 = M_SiO2 / ρ_SiO2 = 60.08 / 2.21 = 27.18 cm³/mol
- Fraction of oxide INTO silicon: V_Si / V_SiO2 = 12.06 / 27.18 ≈ **44.35%**
- Fraction of oxide OUTSIDE silicon: 1 - 0.4435 ≈ **55.65%**

## Results

| Metric | Value |
|--------|-------|
| Total records validated | 79181 |
| Mean % inside Si | 33.26% |
| Median % inside Si | 38.76% |
| Std deviation | 11.03% |
| Min | 0.60% |
| Max | 48.34% |

### Deviation from expected 44.35%

| Within | Count | Percentage |
|--------|-------|------------|
| ±1% | 69 | 0.1% |
| ±2% | 10682 | 13.5% |
| ±5% | 36926 | 46.6% |
| >±5% | 42255 | 53.4% |

### Per-Pressure Breakdown

| Pressure (atm) | Mean % inside | Std | n |
|-----------------|---------------|-----|---|
| 0.1563 | 29.86% | 12.53% | 9200 |
| 0.1938 | 30.85% | 12.12% | 10000 |
| 0.2245 | 31.49% | 11.83% | 10000 |
| 0.4335 | 34.19% | 10.35% | 10000 |
| 0.4449 | 34.30% | 10.29% | 9981 |
| 0.4903 | 34.66% | 10.05% | 10000 |
| 0.5359 | 35.00% | 9.83% | 10000 |
| 0.6056 | 35.45% | 9.54% | 10000 |

## Interpretation

The data shows a mean of 33.26% inside, which deviates from the expected 44.35%. This may indicate issues with interface detection or TCAD simulation settings.
