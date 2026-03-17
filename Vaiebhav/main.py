"""
main.py — End-to-end grey-box thickness prediction pipeline.

WHAT:   Orchestrates the full pipeline from data loading through Deal-Grove
        calibration to grey-box model training and evaluation.
WHY:    This is the Stage 1 thickness predictor. Its output feeds into 
        Stage 2's coordinate transformation for concentration field PINNs.
HOW:    1. Load thickness values (real + LHS augmented)
        2. Compute Deal-Grove baseline with per-point pressure
        3. Compute log-ratio residuals: delta = log(e_real / e_DG)
        4. Build input arrays: [Pressure, Temperature, Time, O2 Flow, N2 Flow, e_DG]
        5. Train grey-box MLP to predict delta
        6. Evaluate: DG-only vs DG+grey predictions

INPUT:  thickness_combined.csv (from lhs_augmentation.py)
OUTPUT: best_grey_model.pt (checkpoint with model + scaler)

Author: Vaiebhav Shreevarshan R (2024AAPS1427G)
Date:   2026-03-18 (v2 — with pressure, improved model, LHS data)
"""

import numpy as np
import pandas as pd
import torch
import os
import sys
import time as time_module

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from src.physics.deal_grove import DealGrove
from src.model.grey_thickness_model import GreyThicknessModel, train_grey_model


# ─── Device ──────────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
    print("Using MPS (Apple Silicon GPU)")
else:
    device = torch.device('cpu')
    print("Using CPU")


def main():
    print("=" * 70)
    print("GREY-BOX THICKNESS PREDICTION PIPELINE (v2)")
    print("=" * 70)
    
    # ─── Phase 1: Load data ──────────────────────────────────────────
    print("\n[1/5] Loading thickness data...")
    
    combined_file = os.path.join(script_dir, 'thickness_combined.csv')
    if os.path.exists(combined_file):
        df = pd.read_csv(combined_file)
        n_real = (df['Source'] == 'real').sum()
        n_synth = (df['Source'] == 'LHS_synthetic').sum()
        print(f"  Combined dataset: {len(df)} points ({n_real} real + {n_synth} synthetic)")
    else:
        # Fall back to real-only
        real_file = os.path.join(script_dir, 'thickness_values.csv')
        df = pd.read_csv(real_file)
        df['Source'] = 'real'
        print(f"  Real-only dataset: {len(df)} points")
    
    pressures = df['Pressure'].values
    o2_flows = df['O2 Flow'].values
    n2_flows = df['N2 Flow'].values
    temps = df['Temperature'].values
    times = df['Time'].values
    thicknesses = df['Thickness'].values
    
    print(f"  Thickness range: [{thicknesses.min():.6f}, {thicknesses.max():.6f}]")
    print(f"  Pressure range:  [{pressures.min():.4f}, {pressures.max():.4f}]")
    
    # ─── Phase 2: Deal-Grove calibration ─────────────────────────────
    print("\n[2/5] Calibrating Deal-Grove model...")
    
    dg = DealGrove(total_pressure_atm=0.44)
    
    # Only fit on REAL data (not synthetic — those already come from DG)
    real_mask = df['Source'].values == 'real'
    fitted = dg.fit(
        temps[real_mask], times[real_mask],
        o2_flows[real_mask], n2_flows[real_mask],
        thicknesses[real_mask],
        pressure_arr=pressures[real_mask],
        verbose=False,
    )
    
    print(f"  B0  = {fitted['B0']:.6e}")
    print(f"  EB  = {fitted['EB']:.4f} eV")
    print(f"  BA0 = {fitted['BA0']:.6e}")
    print(f"  EBA = {fitted['EBA']:.4f} eV")
    print(f"  Fit cost: {fitted['cost']:.6e}, success: {fitted['success']}")
    
    # ─── Phase 3: Generate DG predictions & compute residuals ────────
    print("\n[3/5] Computing Deal-Grove baseline and residuals...")
    
    e_dg = dg.predict_with_fitted(
        temps, times, o2_flows, n2_flows, fitted,
        pressure=pressures,
    )
    
    # Clamp DG predictions to be positive
    e_dg = np.maximum(e_dg, 1e-12)
    
    # DG-only metrics
    dg_errors = np.abs(thicknesses - e_dg)
    print(f"  DG-only MAE:  {dg_errors.mean():.6e}")
    print(f"  DG-only RMSE: {np.sqrt(np.mean(dg_errors**2)):.6e}")
    print(f"  DG-only MaxE: {dg_errors.max():.6e}")
    
    # Compute log-ratio residual
    delta_e = np.log(thicknesses / e_dg)
    print(f"  Delta (log-ratio) range: [{delta_e.min():.4f}, {delta_e.max():.4f}]")
    print(f"  Delta mean: {delta_e.mean():.6f}, std: {delta_e.std():.6f}")
    
    # ─── Phase 4: Build ML dataset ──────────────────────────────────
    print("\n[4/5] Preparing ML dataset and training...")
    
    # Input features: [Pressure, Temperature, Time, O2 Flow, N2 Flow, e_DG]
    X = np.column_stack([pressures, temps, times, o2_flows, n2_flows, e_dg])
    y = delta_e
    
    # Split: 80% train, 10% val, 10% test
    np.random.seed(42)
    n = len(X)
    indices = np.random.permutation(n)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    
    print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Build model
    model = GreyThicknessModel(
        input_dim=6,
        hidden_dim=256,
        n_res_blocks=4,
        dropout=0.05,
    )
    
    save_path = os.path.join(script_dir, 'best_grey_model.pt')
    
    t0 = time_module.time()
    model, scaler, history = train_grey_model(
        model, X_train, y_train, X_val, y_val,
        epochs=3000,
        batch_size=2048,
        lr=2e-3,
        patience=300,
        device=str(device),
        save_path=save_path,
        e_dg_col_idx=5,
    )
    elapsed = time_module.time() - t0
    print(f"\n  Training completed in {elapsed:.1f}s")
    
    # ─── Phase 5: Final evaluation ──────────────────────────────────
    print("\n[5/5] Evaluating on test set...")
    
    model.eval()
    model = model.to(device)
    
    # Scale test data
    X_test_s = scaler.transform_x(X_test)
    X_test_t = torch.tensor(X_test_s, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        delta_pred_s = model(X_test_t).cpu().numpy().squeeze()
    
    # Unscale predictions
    delta_pred = scaler.inverse_transform_y(delta_pred_s)
    
    # Recover predicted thickness: e_pred = e_DG * exp(delta_pred)
    e_dg_test = X_test[:, 5]  # e_DG column
    e_pred = e_dg_test * np.exp(delta_pred)
    e_real = e_dg_test * np.exp(y_test)
    e_dg_only = e_dg_test  # DG prediction (uncorrected)
    
    # Metrics
    grey_errors = np.abs(e_real - e_pred)
    dg_only_errors = np.abs(e_real - e_dg_only)
    
    grey_mae = grey_errors.mean()
    grey_rmse = np.sqrt(np.mean(grey_errors**2))
    grey_max = grey_errors.max()
    
    dg_mae = dg_only_errors.mean()
    dg_rmse = np.sqrt(np.mean(dg_only_errors**2))
    dg_max = dg_only_errors.max()
    
    # Percentage errors
    pct_errors = 100.0 * grey_errors / np.maximum(e_real, 1e-12)
    pct_within_1 = (pct_errors < 1.0).mean() * 100
    pct_within_5 = (pct_errors < 5.0).mean() * 100
    pct_within_10 = (pct_errors < 10.0).mean() * 100
    
    print()
    print("=" * 70)
    print("FINAL TEST SET RESULTS")
    print("=" * 70)
    print(f"{'Metric':<25s} {'DG-Only':>15s} {'Grey-Box':>15s} {'Improvement':>12s}")
    print("-" * 70)
    print(f"{'MAE':<25s} {dg_mae:>15.6e} {grey_mae:>15.6e} {(1-grey_mae/dg_mae)*100:>11.1f}%")
    print(f"{'RMSE':<25s} {dg_rmse:>15.6e} {grey_rmse:>15.6e} {(1-grey_rmse/dg_rmse)*100:>11.1f}%")
    print(f"{'Max Error':<25s} {dg_max:>15.6e} {grey_max:>15.6e} {(1-grey_max/dg_max)*100:>11.1f}%")
    print()
    print(f"  Grey-box % error within  1%: {pct_within_1:.1f}%")
    print(f"  Grey-box % error within  5%: {pct_within_5:.1f}%")
    print(f"  Grey-box % error within 10%: {pct_within_10:.1f}%")
    print(f"  Mean % error:                {pct_errors.mean():.2f}%")
    print(f"  Median % error:              {np.median(pct_errors):.2f}%")
    print("=" * 70)
    
    # Save results for documentation
    results = {
        'dg_mae': dg_mae, 'dg_rmse': dg_rmse, 'dg_max': dg_max,
        'grey_mae': grey_mae, 'grey_rmse': grey_rmse, 'grey_max': grey_max,
        'pct_within_1': pct_within_1, 'pct_within_5': pct_within_5,
        'pct_within_10': pct_within_10,
        'mean_pct_error': pct_errors.mean(),
        'fitted_params': fitted,
    }
    
    return results


if __name__ == "__main__":
    results = main()
