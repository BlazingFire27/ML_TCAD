"""
lhs_augmentation.py — Generate synthetic thickness data via LHS + calibrated Deal-Grove.

WHAT:   Uses Latin Hypercube Sampling to generate uniformly distributed points
        in the 5D input space (P, O2, N2, T, t), then predicts thickness at
        each point using a Deal-Grove model calibrated to real data.
WHY:    The TCAD data has only 8 pressures × 10 O2 × 10 N2 × 10 temps × 10 times.
        LHS fills the gaps for smoother ML training.
HOW:    1. Load thickness_values.csv (real data)
        2. Calibrate Deal-Grove to real thicknesses
        3. Generate N LHS samples within observed parameter bounds
        4. Predict thickness at LHS points using calibrated DG
        5. Add calibrated Gaussian noise
        6. Save combined dataset

Author: Vaiebhav Shreevarshan R (2024AAPS1427G)
Date:   2026-03-18
"""

import numpy as np
import pandas as pd
import os
import sys

# Add project root to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from src.physics.deal_grove import DealGrove


# ─── Latin Hypercube Sampling ────────────────────────────────────────────────
def latin_hypercube_sample(n_samples, n_dims, seed=42):
    """
    Generate Latin Hypercube Samples in [0, 1]^n_dims.
    
    For each dimension:
    1. Divide [0, 1] into n_samples equal intervals
    2. Take one random point from each interval
    3. Randomly permute the order
    
    Returns: (n_samples, n_dims) array in [0, 1]
    """
    rng = np.random.RandomState(seed)
    samples = np.zeros((n_samples, n_dims))
    
    for dim in range(n_dims):
        # Create interval midpoints
        intervals = np.arange(n_samples) / n_samples
        # Add random offset within each interval
        points = intervals + rng.uniform(0, 1.0 / n_samples, n_samples)
        # Random permutation
        rng.shuffle(points)
        samples[:, dim] = points
    
    return samples


def scale_lhs_to_bounds(lhs_samples, bounds_min, bounds_max):
    """Scale LHS samples from [0,1] to [bounds_min, bounds_max]."""
    return lhs_samples * (bounds_max - bounds_min) + bounds_min


# ─── Main augmentation pipeline ─────────────────────────────────────────────
def augment_thickness_data(
    n_synthetic=5000,
    noise_scale=0.5,  # fraction of residual std to use as noise
    seed=42,
):
    """
    Generate augmented thickness data using LHS + calibrated Deal-Grove.
    
    Args:
        n_synthetic: Number of synthetic LHS points to generate
        noise_scale: Fraction of calibration residual std to add as noise
        seed: Random seed for reproducibility
    """
    # ─── Load real data ──────────────────────────────────────────────
    real_file = os.path.join(script_dir, 'thickness_values.csv')
    df_real = pd.read_csv(real_file)
    print(f"Loaded {len(df_real)} real thickness records")
    
    # Extract arrays
    pressures = df_real['Pressure'].values
    o2_flows = df_real['O2 Flow'].values
    n2_flows = df_real['N2 Flow'].values
    temps = df_real['Temperature'].values
    times = df_real['Time'].values
    thicknesses = df_real['Thickness'].values
    
    # ─── Calibrate Deal-Grove ────────────────────────────────────────
    print("\nCalibrating Deal-Grove model to real data...")
    dg = DealGrove(total_pressure_atm=0.44)  # default fallback; actual pressure passed per-point
    
    # Calibrate with per-point pressure
    fitted = dg.fit(
        temps, times, o2_flows, n2_flows, thicknesses,
        pressure_arr=pressures,
        verbose=True
    )
    
    print(f"\nFitted parameters:")
    for k, v in fitted.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.6e}")
        else:
            print(f"  {k}: {v}")
    
    # ─── Compute residuals ───────────────────────────────────────────
    dg_predictions = dg.predict_with_fitted(temps, times, o2_flows, n2_flows, fitted,
                                              pressure=pressures)
    residuals = thicknesses - dg_predictions
    residual_std = np.std(residuals)
    residual_mean = np.mean(residuals)
    
    print(f"\nCalibration residuals:")
    print(f"  Mean:  {residual_mean:.6e}")
    print(f"  Std:   {residual_std:.6e}")
    print(f"  MAE:   {np.mean(np.abs(residuals)):.6e}")
    print(f"  RMSE:  {np.sqrt(np.mean(residuals**2)):.6e}")
    
    # ─── Generate LHS samples ───────────────────────────────────────
    print(f"\nGenerating {n_synthetic} LHS samples in 5D...")
    
    # Parameter bounds from real data
    param_names = ['Pressure', 'O2 Flow', 'N2 Flow', 'Temperature', 'Time']
    bounds_min = np.array([
        pressures.min(), o2_flows.min(), n2_flows.min(), temps.min(), times.min()
    ])
    bounds_max = np.array([
        pressures.max(), o2_flows.max(), n2_flows.max(), temps.max(), times.max()
    ])
    
    print("Parameter bounds:")
    for name, lo, hi in zip(param_names, bounds_min, bounds_max):
        print(f"  {name:15s}: [{lo:.4f}, {hi:.4f}]")
    
    # Generate LHS samples
    lhs_unit = latin_hypercube_sample(n_synthetic, 5, seed=seed)
    lhs_scaled = scale_lhs_to_bounds(lhs_unit, bounds_min, bounds_max)
    
    # Extract individual parameters
    syn_pres = lhs_scaled[:, 0]
    syn_o2 = lhs_scaled[:, 1]
    syn_n2 = lhs_scaled[:, 2]
    syn_temp = lhs_scaled[:, 3]
    syn_time = lhs_scaled[:, 4]
    
    # ─── Predict thickness at LHS points ─────────────────────────────
    # Predict thickness at LHS points using calibrated DG (with per-point pressure)
    syn_dg_thickness = dg.predict_with_fitted(
        syn_temp, syn_time, syn_o2, syn_n2, fitted,
        pressure=syn_pres
    )
    
    # Add calibrated noise
    rng = np.random.RandomState(seed + 1)
    noise = rng.normal(0, residual_std * noise_scale, n_synthetic)
    syn_thickness = np.maximum(syn_dg_thickness + noise, 1e-8)  # Clamp positive
    
    # ─── Combine and save ────────────────────────────────────────────
    print(f"\nSynthetic thickness statistics:")
    print(f"  Min:    {syn_thickness.min():.6e}")
    print(f"  Max:    {syn_thickness.max():.6e}")
    print(f"  Mean:   {syn_thickness.mean():.6e}")
    print(f"  Std:    {syn_thickness.std():.6e}")
    
    # Create synthetic DataFrame
    df_synthetic = pd.DataFrame({
        'Pressure':    syn_pres,
        'O2 Flow':     syn_o2,
        'N2 Flow':     syn_n2,
        'Temperature': syn_temp,
        'Time':        syn_time,
        'Thickness':   syn_thickness,
        'Source':       'LHS_synthetic',
    })
    
    # Add source column to real data
    df_real_with_source = df_real[['Pressure', 'O2 Flow', 'N2 Flow', 
                                   'Temperature', 'Time', 'Thickness']].copy()
    df_real_with_source['Source'] = 'real'
    
    # Combine
    df_combined = pd.concat([df_real_with_source, df_synthetic], ignore_index=True)
    
    # Save files
    synthetic_file = os.path.join(script_dir, 'thickness_synthetic_lhs.csv')
    combined_file = os.path.join(script_dir, 'thickness_combined.csv')
    
    df_synthetic.to_csv(synthetic_file, index=False)
    df_combined.to_csv(combined_file, index=False)
    
    print(f"\n{'='*60}")
    print("AUGMENTATION SUMMARY")
    print(f"{'='*60}")
    print(f"  Real data points:      {len(df_real_with_source):,}")
    print(f"  Synthetic LHS points:  {len(df_synthetic):,}")
    print(f"  Combined total:        {len(df_combined):,}")
    print(f"  Noise scale:           {noise_scale} × residual_std = {noise_scale * residual_std:.6e}")
    print(f"\n  Saved:")
    print(f"    Synthetic only: {os.path.abspath(synthetic_file)}")
    print(f"    Combined:       {os.path.abspath(combined_file)}")
    print(f"{'='*60}")
    
    return df_combined, fitted


if __name__ == "__main__":
    augment_thickness_data(n_synthetic=5000, noise_scale=0.5, seed=42)
