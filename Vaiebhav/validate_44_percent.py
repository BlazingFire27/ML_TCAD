"""
validate_44_percent.py — Validate the 44/56 oxide growth partition rule.

WHAT:   Checks whether ~44% of the total oxide extends INTO silicon (positive X)
        and ~56% extends OUTSIDE (negative X), as predicted by the molar volume
        ratio of Si to SiO2.
WHY:    This is a fundamental physical law that validates our thickness extraction.
        If the data consistently violates this rule, our interface detection or 
        thickness computation may be wrong.
HOW:    From thickness_values.csv:
        - inside_thickness  = X_interface  (oxide into Si, positive)
        - outside_thickness = |X_surface|  (oxide above original surface, negative → abs)
        - total_thickness   = inside + outside = Thickness column
        - pct_inside        = 100 * inside / total

PHYSICS:
  Si density: 2.33 g/cm³, molar mass: 28.09 g/mol → V_Si = 12.06 cm³/mol
  SiO2 density: 2.21 g/cm³, molar mass: 60.08 g/mol → V_SiO2 = 27.18 cm³/mol
  For every unit of Si consumed, SiO2 occupies V_SiO2/V_Si ≈ 2.252× more volume.
  The fraction consumed INTO Si = V_Si / V_SiO2 = 12.06/27.18 ≈ 0.4435 ≈ 44%
  The remainder 56% extends above the original surface.

Author: Vaiebhav Shreevarshan R (2024AAPS1427G)
Date:   2026-03-18
"""

import pandas as pd
import numpy as np
import os

# ─── Configuration ───────────────────────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
thickness_file = os.path.join(script_dir, 'thickness_values.csv')
doc_dir = os.path.join(script_dir, '..', 'documentation')
output_doc = os.path.join(doc_dir, '03_44_percent_validation.md')

# Physical constant
EXPECTED_PCT_INSIDE = 44.35  # Based on molar volume derivation


# ─── Main ────────────────────────────────────────────────────────────────────
def validate():
    df = pd.read_csv(thickness_file)
    print(f"Loaded {len(df)} thickness records from {thickness_file}")
    
    # inside_thickness = X_interface (positive: how far oxide extends into Si)
    # outside_thickness = |X_surface| (X_surface is negative: how far oxide extends outward)
    df['inside_thickness'] = df['X_interface']
    df['outside_thickness'] = df['X_surface'].abs()
    df['total_thickness'] = df['inside_thickness'] + df['outside_thickness']
    df['pct_inside'] = 100.0 * df['inside_thickness'] / df['total_thickness']
    
    # Filter out rows where total_thickness is essentially zero
    valid = df[df['total_thickness'] > 1e-8].copy()
    print(f"Valid records (total_thickness > 1e-8): {len(valid)}")
    
    # ─── Statistics ──────────────────────────────────────────────────────
    pct = valid['pct_inside']
    
    print()
    print("=" * 60)
    print(f"44% RULE VALIDATION (expected ~{EXPECTED_PCT_INSIDE:.1f}% inside silicon)")
    print("=" * 60)
    print(f"  Mean:    {pct.mean():.2f}%")
    print(f"  Median:  {pct.median():.2f}%")
    print(f"  Std:     {pct.std():.2f}%")
    print(f"  Min:     {pct.min():.2f}%")
    print(f"  Max:     {pct.max():.2f}%")
    
    # Deviation from expected
    dev_from_44 = (pct - EXPECTED_PCT_INSIDE).abs()
    within_1 = (dev_from_44 <= 1.0).sum()
    within_2 = (dev_from_44 <= 2.0).sum()
    within_5 = (dev_from_44 <= 5.0).sum()
    outside_5 = (dev_from_44 > 5.0).sum()
    
    print()
    print("DEVIATION FROM EXPECTED 44.35%:")
    print(f"  Within ±1%: {within_1:6d}  ({100*within_1/len(valid):.1f}%)")
    print(f"  Within ±2%: {within_2:6d}  ({100*within_2/len(valid):.1f}%)")
    print(f"  Within ±5%: {within_5:6d}  ({100*within_5/len(valid):.1f}%)")
    print(f"  Beyond ±5%: {outside_5:6d}  ({100*outside_5/len(valid):.1f}%)")
    
    # Distribution histogram
    print()
    print("DISTRIBUTION OF % INSIDE SILICON:")
    bins = [0, 30, 35, 40, 42, 43, 44, 45, 46, 48, 50, 55, 60, 100]
    labels = ['<30', '30-35', '35-40', '40-42', '42-43', '43-44', '44-45', 
              '45-46', '46-48', '48-50', '50-55', '55-60', '>60']
    counts = pd.cut(pct, bins=bins).value_counts().sort_index()
    for label, count in zip(labels, counts):
        bar = '#' * min(count // 100, 60)
        print(f"  {label:7s}: {count:6d}  {bar}")
    
    # Per-pressure breakdown
    print()
    print("PER-PRESSURE BREAKDOWN:")
    for pressure, group in valid.groupby('Pressure'):
        gpct = group['pct_inside']
        print(f"  P={pressure:.4f}: mean={gpct.mean():.2f}%, "
              f"std={gpct.std():.2f}%, n={len(group)}")
    
    # Worst deviations
    print()
    print("10 WORST DEVIATIONS FROM 44.35%:")
    valid['deviation'] = (valid['pct_inside'] - EXPECTED_PCT_INSIDE).abs()
    worst = valid.nlargest(10, 'deviation')
    print(worst[['Pressure', 'Temperature', 'Time', 'Thickness', 
                  'pct_inside', 'deviation']].to_string(index=False))
    
    # ─── Save documentation ─────────────────────────────────────────────
    doc_content = f"""# Phase 3: 44% Rule Validation — Documentation

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
| Total records validated | {len(valid)} |
| Mean % inside Si | {pct.mean():.2f}% |
| Median % inside Si | {pct.median():.2f}% |
| Std deviation | {pct.std():.2f}% |
| Min | {pct.min():.2f}% |
| Max | {pct.max():.2f}% |

### Deviation from expected 44.35%

| Within | Count | Percentage |
|--------|-------|------------|
| ±1% | {within_1} | {100*within_1/len(valid):.1f}% |
| ±2% | {within_2} | {100*within_2/len(valid):.1f}% |
| ±5% | {within_5} | {100*within_5/len(valid):.1f}% |
| >±5% | {outside_5} | {100*outside_5/len(valid):.1f}% |

### Per-Pressure Breakdown

| Pressure (atm) | Mean % inside | Std | n |
|-----------------|---------------|-----|---|
"""
    for pressure, group in valid.groupby('Pressure'):
        gpct = group['pct_inside']
        doc_content += f"| {pressure:.4f} | {gpct.mean():.2f}% | {gpct.std():.2f}% | {len(group)} |\n"
    
    doc_content += "\n## Interpretation\n\n"
    if pct.mean() > 42 and pct.mean() < 47:
        doc_content += "The data **closely follows** the expected 44% rule, confirming that our thickness extraction (sub-grid interpolation) is physically consistent.\n"
    else:
        doc_content += f"The data shows a mean of {pct.mean():.2f}% inside, which deviates from the expected 44.35%. This may indicate issues with interface detection or TCAD simulation settings.\n"
    
    os.makedirs(os.path.dirname(output_doc), exist_ok=True)
    with open(output_doc, 'w', encoding='utf-8') as f:
        f.write(doc_content)
    
    print(f"\nDocumentation saved to: {os.path.abspath(output_doc)}")
    
    return valid


if __name__ == "__main__":
    validate()
