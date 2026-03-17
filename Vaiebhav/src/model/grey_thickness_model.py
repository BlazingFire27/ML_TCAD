"""
grey_thickness_model.py — Improved grey-box MLP for residual thickness prediction.

WHAT:   MLP that predicts Delta-epsilon (log-ratio residual between observed
        thickness and Deal-Grove predicted thickness).
WHY:    The Deal-Grove model captures ~80-90% of the physics. This ML corrector
        learns the remaining residual so the combined pipeline achieves near-zero
        prediction error.
HOW:    - Input:  6D [Pressure, Temperature, Time, O2 Flow, N2 Flow, e_DG]
        - Target: log(e_real / e_DG)  (multiplicative correction in log-space)
        - Output: e_final = e_DG * exp(predicted_delta)
        
Architecture:
    - SiLU activations (smooth, better gradient flow than LeakyReLU for physics)
    - Residual/skip connections between blocks
    - BatchNorm + Dropout for regularization
    - Kaiming initialization

Author: Vaiebhav Shreevarshan R (2024AAPS1427G)
Date:   2026-03-18 (v2 — with pressure, improved architecture)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import copy
import os


# ─── Scaler ──────────────────────────────────────────────────────────────────
class GreyThicknessScaler:
    """StandardScaler equivalent for features and targets."""
    def __init__(self):
        self.mean_x = None
        self.std_x = None
        self.mean_y = None
        self.std_y = None

    def fit(self, X, y=None):
        self.mean_x = np.mean(X, axis=0)
        self.std_x = np.std(X, axis=0)
        self.std_x[self.std_x == 0] = 1.0

        if y is not None:
            self.mean_y = np.mean(y, axis=0)
            self.std_y = np.std(y, axis=0)
            if isinstance(self.std_y, np.ndarray):
                self.std_y[self.std_y == 0] = 1.0
            elif self.std_y == 0:
                self.std_y = 1.0

    def transform_x(self, X):
        return (X - self.mean_x) / self.std_x

    def transform_y(self, y):
        return (y - self.mean_y) / self.std_y

    def inverse_transform_x(self, X_scaled):
        return X_scaled * self.std_x + self.mean_x

    def inverse_transform_y(self, y_scaled):
        return y_scaled * self.std_y + self.mean_y


# ─── Dataset ─────────────────────────────────────────────────────────────────
class GreyThicknessDataset(Dataset):
    """
    PyTorch Dataset wrapper.
    Inputs: [Pressure, Temperature, Time, O2 Flow, N2 Flow, e_DG]
    Target: log(e_real / e_DG)
    """
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ─── Residual Block ──────────────────────────────────────────────────────────
class ResidualBlock(nn.Module):
    """Two-layer block with skip connection, BatchNorm, SiLU, and Dropout."""
    def __init__(self, dim, dropout=0.05):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.activation(self.block(x) + x))


# ─── Grey-Box Model ─────────────────────────────────────────────────────────
class GreyThicknessModel(nn.Module):
    """
    Improved MLP for predicting Delta-epsilon.
    Uses SiLU activations, residual connections, and deeper architecture.

    Input dim = 6: [Pressure, Temperature, Time, O2 Flow, N2 Flow, e_DG]
    """
    def __init__(self, input_dim=6, hidden_dim=256, n_res_blocks=4, dropout=0.05):
        super().__init__()

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
        )

        # Residual blocks
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim, dropout=dropout) for _ in range(n_res_blocks)]
        )

        # Output head with gradual dimension reduction
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.res_blocks(x)
        return self.output_head(x)


# ─── Accuracy ────────────────────────────────────────────────────────────────
def calculate_accuracy(preds_delta_e, targets_delta_e, e_dg):
    """
    Calculate accuracy: percentage error compared to total thickness e_real.
    e_real = e_dg * exp(target_delta)
    e_pred = e_dg * exp(pred_delta)
    """
    e_real = e_dg * torch.exp(targets_delta_e)
    e_pred = e_dg * torch.exp(preds_delta_e)
    e_real = torch.clamp(e_real, min=1e-12)
    error = torch.abs(e_real - e_pred)
    percentage_error = (error / e_real) * 100.0
    accuracy = 100.0 - percentage_error
    return torch.mean(accuracy).item()


# ─── Training Loop ───────────────────────────────────────────────────────────
def train_grey_model(model, X_train, y_train, X_val, y_val,
                     epochs=3000, batch_size=2048, lr=2e-3,
                     patience=300, device='cpu', save_path='best_grey_model.pt',
                     e_dg_col_idx=5):
    """
    Training loop with:
    - CosineAnnealingWarmRestarts scheduler
    - Gradient clipping
    - Early stopping on validation loss
    - Comprehensive metrics logging
    
    Args:
        e_dg_col_idx: Index of the e_DG column in the input features (default 5 for 6D input)
    """
    model = model.to(device)

    # Fit scaler
    scaler = GreyThicknessScaler()
    scaler.fit(X_train, y_train)

    # Scale data
    X_train_s = scaler.transform_x(X_train)
    y_train_s = scaler.transform_y(y_train)
    X_val_s = scaler.transform_x(X_val)
    y_val_s = scaler.transform_y(y_val)

    # Create DataLoaders
    train_ds = GreyThicknessDataset(X_train_s, y_train_s)
    val_ds = GreyThicknessDataset(X_val_s, y_val_s)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=200, T_mult=2, eta_min=1e-6)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None
    history = {'train_loss': [], 'val_loss': [], 'val_mae': [], 'val_rmse': []}

    print(f"\nTraining for up to {epochs} epochs, patience={patience}")
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}")
    print(f"  Batch size: {batch_size}, LR: {lr}")
    print()

    for epoch in range(epochs):
        # ─── Train ────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)
        train_loss /= len(train_ds)
        scheduler.step()

        # ─── Validate ─────────────────────────────────────────
        model.eval()
        val_loss_scaled = 0.0
        all_val_preds_unscaled = []
        all_val_targets_unscaled = []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                preds = model(X_batch)
                val_loss_scaled += criterion(preds, y_batch).item() * X_batch.size(0)
                
                # Unscale for real metrics
                preds_us = preds.cpu().numpy() * scaler.std_y + scaler.mean_y
                targets_us = y_batch.cpu().numpy() * scaler.std_y + scaler.mean_y
                all_val_preds_unscaled.append(preds_us)
                all_val_targets_unscaled.append(targets_us)
                
        val_loss_scaled /= len(val_ds)
        
        vp = np.concatenate(all_val_preds_unscaled, axis=0).squeeze()
        vt = np.concatenate(all_val_targets_unscaled, axis=0).squeeze()
        val_mae = np.mean(np.abs(vp - vt))
        val_rmse = np.sqrt(np.mean((vp - vt) ** 2))

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss_scaled)
        history['val_mae'].append(val_mae)
        history['val_rmse'].append(val_rmse)

        # ─── Early stopping ──────────────────────────────────
        if val_loss_scaled < best_val_loss:
            best_val_loss = val_loss_scaled
            epochs_no_improve = 0
            best_model_state = copy.deepcopy(model.state_dict())
            if save_path:
                torch.save({
                    'model_state_dict': best_model_state,
                    'scaler': scaler,
                    'epoch': epoch,
                    'val_loss': best_val_loss,
                    'val_mae': val_mae,
                    'val_rmse': val_rmse,
                }, save_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch}. Best val loss: {best_val_loss:.8f}")
                break

        # ─── Logging ──────────────────────────────────────────
        current_lr = optimizer.param_groups[0]['lr']
        if epoch % 100 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:4d} | Train: {train_loss:.6f} | "
                  f"Val: {val_loss_scaled:.6f} | "
                  f"MAE: {val_mae:.6e} | RMSE: {val_rmse:.6e} | "
                  f"LR: {current_lr:.2e} | No-imp: {epochs_no_improve}")

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, scaler, history
