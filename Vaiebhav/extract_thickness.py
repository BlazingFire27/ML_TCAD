"""
extract_thickness.py — Extract oxide thickness from spatial concentration profiles.

WHAT:   Reads cleaned CSVs (with Pressure), computes oxide thickness for each 
        (file, step) group using sub-grid linear interpolation of the Si-SiO2 
        interface, and saves the result as thickness_values.csv.
WHY:    We need a dataset of thickness vs. process conditions for the grey-box model.
        Thickness is the critical output of Stage 1.
HOW:    For each spatial profile:
        1. Compute logY = log10(Y) where Y is oxygen concentration
        2. Identify the oxide region where logY > threshold (default 2.0)
        3. Find the Si-SiO2 interface via sub-grid linear interpolation between
           the last "reactive" point and first "bulk" point
        4. x_surface = min(X) is the gas-oxide surface
        5. thickness = x_interface - x_surface

INPUT:  data/new processed (with pressure)/Cleaned_oxi*.csv
OUTPUT: src/Vaiebhav/thickness_values.csv

Author: Vaiebhav Shreevarshan R (2024AAPS1427G)
Date:   2026-03-18
"""

import pandas as pd
import numpy as np
import glob
import os
import time as time_module

# ─── Configuration ───────────────────────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, '..', '..', 'data', 'new processed (with pressure)')
output_file = os.path.join(script_dir, 'thickness_values.csv')

REACTIVE_THRESHOLD = 2.0   # logY > this means "oxide" (not bulk Si)
Y_CLAMP_MIN = 1e-9          # clamp Y to avoid log(0)


# ─── Sub-grid interpolation ─────────────────────────────────────────────────
def interpolate_interface(X_sorted, logY_sorted, threshold=REACTIVE_THRESHOLD):
    """
    Find the Si-SiO2 interface position using sub-grid linear interpolation.
    
    The interface is where logY crosses the threshold from above (oxide) to 
    below (bulk silicon). We interpolate between the last point above threshold
    and the first point below it.
    
    Args:
        X_sorted: 1D array of spatial positions, sorted ascending
        logY_sorted: 1D array of log10(concentration), same order as X_sorted
        threshold: logY value that separates oxide from bulk Si
        
    Returns:
        x_interface: interpolated interface position (float)
        
    Raises:
        ValueError: if no reactive (oxide) point is found
    """
    mask = logY_sorted > threshold
    if not np.any(mask):
        raise ValueError("No reactive (oxide) point found in profile.")
    
    # Find the last index where logY > threshold (the deepest oxide point)
    i1 = np.where(mask)[0].max()
    i2 = i1 + 1
    
    # If the last reactive point is at the edge of the array, return its X
    if i2 >= len(X_sorted):
        return float(X_sorted[i1])
    
    X1, X2 = X_sorted[i1], X_sorted[i2]
    Y1, Y2 = logY_sorted[i1], logY_sorted[i2]
    
    # If the two logY values are equal, return X1
    if np.isclose(Y2, Y1):
        return float(X1)
    
    # Linear interpolation: find X where logY = threshold
    x_interface = X1 + (threshold - Y1) * (X2 - X1) / (Y2 - Y1)
    return float(x_interface)


# ─── Main extraction ────────────────────────────────────────────────────────
def extract_all_thicknesses():
    """
    Process all cleaned CSVs and extract thickness for every (file, step) group.
    """
    file_paths = sorted(glob.glob(os.path.join(data_dir, "Cleaned_oxi*.csv")))
    if not file_paths:
        print(f"ERROR: No Cleaned_oxi*.csv files found in {data_dir}")
        return
    
    print(f"Found {len(file_paths)} cleaned CSV files")
    
    results = []
    errors = 0
    skipped_no_reactive = 0
    skipped_bad_data = 0

    start_time = time_module.time()

    for file_idx, fpath in enumerate(file_paths):
        fname = os.path.basename(fpath)
        df = pd.read_csv(fpath)
        
        # Convert X and Y to numeric, drop non-numeric rows
        df['X'] = pd.to_numeric(df['X'], errors='coerce')
        df['Y'] = pd.to_numeric(df['Y'], errors='coerce')
        df = df.dropna(subset=['X', 'Y'])
        
        if len(df) == 0:
            print(f"  WARNING: {fname} has no valid X/Y data after cleaning")
            continue
        
        # Compute logY
        df['logY'] = np.log10(np.clip(df['Y'].values, Y_CLAMP_MIN, None))
        
        # Group by Step (n) — each step is one spatial profile at one time
        for step_val, step_df in df.groupby('Step (n)', sort=False):
            step_df = step_df.sort_values('X')
            
            X_arr = step_df['X'].values
            logY_arr = step_df['logY'].values
            
            if len(X_arr) < 3:
                skipped_bad_data += 1
                continue
            
            try:
                x_interface = interpolate_interface(X_arr, logY_arr, REACTIVE_THRESHOLD)
                x_surface = float(np.min(X_arr))
                thickness = x_interface - x_surface
                
                # Extract process conditions (constant within a step)
                pressure = float(step_df['Pressure'].iloc[0])
                o2_flow = float(step_df['O2 Flow'].iloc[0])
                n2_flow = float(step_df['N2 Flow'].iloc[0])
                temperature = float(step_df['Temperature'].iloc[0])
                time_val = float(step_df['Time'].iloc[0])
                
                results.append({
                    'Pressure':     pressure,
                    'O2 Flow':      o2_flow,
                    'N2 Flow':      n2_flow,
                    'Temperature':  temperature,
                    'Time':         time_val,
                    'X_surface':    round(x_surface, 8),
                    'X_interface':  round(x_interface, 8),
                    'Thickness':    round(thickness, 8),
                })
                
            except ValueError:
                skipped_no_reactive += 1
                continue
            except Exception as e:
                errors += 1
                if errors <= 5:
                    print(f"  ERROR in {fname} step {step_val}: {e}")
                continue
        
        if (file_idx + 1) % 10 == 0:
            elapsed = time_module.time() - start_time
            print(f"  Processed {file_idx + 1}/{len(file_paths)} files "
                  f"({len(results)} thicknesses extracted, {elapsed:.1f}s)")

    elapsed = time_module.time() - start_time

    if not results:
        print("ERROR: No thickness values extracted!")
        return

    # Create DataFrame and save
    result_df = pd.DataFrame(results)
    result_df.to_csv(output_file, index=False)
    
    # Print summary
    print()
    print("=" * 70)
    print("THICKNESS EXTRACTION RESULTS")
    print("=" * 70)
    print(f"  Total extracted:       {len(results):,}")
    print(f"  Skipped (no oxide):    {skipped_no_reactive:,}")
    print(f"  Skipped (bad data):    {skipped_bad_data:,}")
    print(f"  Errors:                {errors:,}")
    print(f"  Time elapsed:          {elapsed:.1f}s")
    print()
    print("THICKNESS STATISTICS:")
    print(f"  Min:     {result_df['Thickness'].min():.8f}")
    print(f"  Max:     {result_df['Thickness'].max():.8f}")
    print(f"  Mean:    {result_df['Thickness'].mean():.8f}")
    print(f"  Median:  {result_df['Thickness'].median():.8f}")
    print(f"  Std:     {result_df['Thickness'].std():.8f}")
    print()
    print("PARAMETER RANGES IN OUTPUT:")
    for col in ['Pressure', 'O2 Flow', 'N2 Flow', 'Temperature', 'Time']:
        vals = result_df[col].unique()
        print(f"  {col:15s}: {len(vals):4d} unique values, range [{min(vals):.4f}, {max(vals):.4f}]")
    print()
    print("INTERFACE POSITIONS:")
    print(f"  X_surface  range: [{result_df['X_surface'].min():.6f}, {result_df['X_surface'].max():.6f}]")
    print(f"  X_interface range: [{result_df['X_interface'].min():.6f}, {result_df['X_interface'].max():.6f}]")
    print()
    print(f"Saved to: {os.path.abspath(output_file)}")
    print("=" * 70)
    
    return result_df


if __name__ == "__main__":
    extract_all_thicknesses()
