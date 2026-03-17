import os
import glob
import numpy as np
import torch
from sklearn.model_selection import train_test_split

# Import project modules
from src.data.loader import OxidationDataLoader
from src.physics.deal_grove import DealGrove
from src.model.grey_thickness_model import GreyThicknessModel, train_grey_model

def get_processed_files(processed_dir):
    """Find all Cleaned_oxi*.csv in the processed directory"""
    return glob.glob(os.path.join(processed_dir, "Cleaned_oxi*.csv"))

def main():
    print("=== Starting Grey Box Model Pipeline ===")
    
    # 1. Setup paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    processed_dir = os.path.join(base_dir, "processed")
    file_paths = get_processed_files(processed_dir)
    
    if not file_paths:
        print(f"Error: No Cleaned_oxi*.csv files found in {processed_dir}")
        return
        
    print(f"Found {len(file_paths)} cleaned files to process.")

    # 2. Extract e_real from Data Loader
    print("\n--- Phase 1: Data Extraction ---")
    loader = OxidationDataLoader(file_paths, reactive_threshold=2.0, extra_points=10)
    df_raw = loader.df  # The processed DataFrame
    
    dg_engine = DealGrove(total_pressure_atm=0.44, reactive_threshold_logY=2.0)
    
    cache_path = os.path.join(processed_dir, "extracted_thicknesses.npz")
    if os.path.exists(cache_path):
        print(f"Loading cached extracted thicknesses from {cache_path}...")
        data = np.load(cache_path)
        features_arr = data['features']
        e_real_arr = data['e_real']
    else:
        print("Extracting thicknesses across all files... This may take a few minutes.")
        features_arr, e_real_arr = loader.get_all_thicknesses(dg_engine)
        print(f"Caching exactly {len(e_real_arr)} thicknesses into {cache_path}...")
        np.savez(cache_path, features=features_arr, e_real=e_real_arr)
    
    print(f"Successfully extracted {len(e_real_arr)} total sequential thickness observations.")
    
    if len(e_real_arr) == 0:
        print("Error: No valid thicknesses extracted. Exiting.")
        return

    # 3. Fit Deal-Grove Kinetics
    print("\n--- Phase 2: Deal-Grove Kinetics Calibration ---")
    
    T_C  = features_arr[:, 0]
    tmin = features_arr[:, 1]
    o2f  = features_arr[:, 2]
    n2f  = features_arr[:, 3]
    
    print("Fitting kinetics parameters (this might take a moment)...")
    fitted = dg_engine.fit(T_C, tmin, o2f, n2f, e_real_arr, verbose=True)
    
    print("\nFitted Parameters:")
    for k, v in fitted.items():
        if isinstance(v, float):
             print(f"  {k}: {v:.6e}")
        else:
             print(f"  {k}: {v}")
             
    if not fitted['success']:
        print("Warning: Optimization did not converge optimally.")

    # 4. Generate Baseline Predictions (e_DG)
    print("\n--- Phase 3: Baseline Prediction & Residual Calculation ---")
    e_dg_arr = dg_engine.predict_with_fitted(T_C, tmin, o2f, n2f, fitted)
    
    # Calculate Residuals (Delta Epsilon as a Multiplier to avoid vanishing gradients)
    # We use log ratio: delta_e_arr = log(e_real / e_dg)
    # So e_final = e_dg * exp(delta_e_arr)
    delta_e_arr = np.log(np.maximum(e_real_arr, 1e-12) / np.maximum(e_dg_arr, 1e-12))
    
    # 5. Prepare ML Dataset
    print("\n--- Phase 4: Grey Model Residual Training ---")
    # ML Input: [Temp, Time, O2, N2, e_DG]
    X_ml = np.column_stack((features_arr, e_dg_arr))
    y_ml = delta_e_arr
    
    # Split: 80% Train, 10% Val, 10% Test
    X_train_val, X_test, y_train_val, y_test = train_test_split(X_ml, y_ml, test_size=0.10, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.1111, random_state=42) # 0.1111 of 0.9 is ~ 0.1
        
    print(f"Data Shapes - Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
    
    # 6. Train Grey Model
    print("Initializing PyTorch MLP Corrector...")
    model = GreyThicknessModel(input_dim=5, hidden_dims=[128, 128, 64, 32])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training on device: {device}")
    
    trained_model, scaler = train_grey_model(
        model=model,
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
        epochs=3000, 
        batch_size=2048, 
        lr=2e-3, 
        patience=200,
        device=device,
        save_path='best_grey_model.pt'
    )
    
    # 7. Evaluate Pipeline
    print("\n--- Phase 5: Pipeline Evaluation ---")
    trained_model.eval()
    
    # Full dataset evaluation on TRUE TEST SET (unseen)
    with torch.no_grad():
        X_test_scaled = scaler.transform_x(X_test)
        X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
        
        # Predict delta_e (which is log ratio)
        pred_delta_e_scaled = trained_model(X_test_tensor).cpu().numpy().flatten()
        pred_delta_e = scaler.inverse_transform_y(pred_delta_e_scaled)
        
        # Calculate e_final using ratio multiplier
        e_dg_test = X_test[:, 4]  # 5th column is e_DG
        pred_e_final = e_dg_test * np.exp(pred_delta_e)
        
        # Actual e_real for test set is obtained similarly
        actual_e_real = e_dg_test * np.exp(y_test)
        
        # Metrics
        dg_error = np.abs(actual_e_real - e_dg_test)
        dg_mae = np.mean(dg_error)
        dg_rmse = np.sqrt(np.mean(dg_error**2))
        dg_accuracy = np.mean(100.0 - (dg_error / np.maximum(actual_e_real, 1e-12)) * 100.0)
        
        grey_error = np.abs(actual_e_real - pred_e_final)
        grey_mae = np.mean(grey_error)
        grey_rmse = np.sqrt(np.mean(grey_error**2))
        grey_accuracy = np.mean(100.0 - (grey_error / np.maximum(actual_e_real, 1e-12)) * 100.0)
        
        print(f"Results on TEST SET (n={len(X_test)}):")
        print(f"  Deal-Grove Baseline MAE : {dg_mae:.6f} (Acc: {dg_accuracy:.2f}%)")
        print(f"  Deal-Grove Baseline RMSE: {dg_rmse:.6f}")
        print(f"  Grey Model Final MAE    : {grey_mae:.6f} (Acc: {grey_accuracy:.2f}%)")
        print(f"  Grey Model Final RMSE   : {grey_rmse:.6f}")
        
    print("\nPipeline execution complete! Model saved to 'best_grey_model.pt'")


if __name__ == "__main__":
    main()
