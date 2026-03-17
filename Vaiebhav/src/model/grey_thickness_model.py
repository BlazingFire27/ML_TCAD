import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import copy
import os

class GreyThicknessScaler:
    """
    StandardScaler equivalent to normalize features and targets for the MLP.
    """
    def __init__(self):
        self.mean_x = None
        self.std_x = None
        self.mean_y = None
        self.std_y = None

    def fit(self, X, y=None):
        self.mean_x = np.mean(X, axis=0)
        self.std_x = np.std(X, axis=0)
        self.std_x[self.std_x == 0] = 1.0  # Avoid division by zero
        
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


class GreyThicknessDataset(Dataset):
    """
    PyTorch Dataset wrapper for the Grey box inputs and residual target.
    Inputs: [Temperature, Time, O2 Flow, N2 Flow, e_DG]
    Target: [\Delta\epsilon]
    """
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class GreyThicknessModel(nn.Module):
    """
    Minimal, disciplined MLP to predict Delta Epsilon (the residual between
    observed thickness and Deal-Grove baseline).
    """
    def __init__(self, input_dim=5, hidden_dims=[256, 128, 64, 32]):
        super().__init__()
        
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.LeakyReLU(0.1))
            layers.append(nn.Dropout(0.1))
            in_dim = h_dim
        
        layers.append(nn.Linear(in_dim, 1)) # Output is \Delta\epsilon
        
        self.mlp = nn.Sequential(*layers)
        
        # Kaiming Normal Initialization for SiLU
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.mlp(x)

def calculate_accuracy(preds_delta_e, targets_delta_e, e_dg):
    """
    Calculate accuracy: percentage error compared to total thickness e_real.
    e_real = e_dg * exp(\Delta\epsilon_target)
    e_pred = e_dg * exp(\Delta\epsilon_pred)
    """
    e_real = e_dg * torch.exp(targets_delta_e)
    e_pred = e_dg * torch.exp(preds_delta_e)
    
    # Avoid division by zero (though thickness should be positive)
    e_real = torch.clamp(e_real, min=1e-12)
    
    error = torch.abs(e_real - e_pred)
    percentage_error = (error / e_real) * 100.0
    accuracy = 100.0 - percentage_error
    
    return torch.mean(accuracy).item()

def train_grey_model(model, X_train, y_train, X_val, y_val, X_test, y_test,
                     epochs=500, batch_size=64, lr=1e-3, 
                     patience=20, device='cpu', save_path='grey_model_ckpt.pt'):
    """
    Training loop with early stopping for the GreyThicknessModel.
    Also calculates and outputs accuracy explicitly.
    """
    model = model.to(device)
    
    # Initialize and fit scaler
    scaler = GreyThicknessScaler()
    scaler.fit(X_train, y_train)
    
    # Scale data
    X_train_scaled = scaler.transform_x(X_train)
    y_train_scaled = scaler.transform_y(y_train)
    X_val_scaled = scaler.transform_x(X_val)
    y_val_scaled = scaler.transform_y(y_val)
    
    # Create DataLoaders
    train_dataset = GreyThicknessDataset(X_train_scaled, y_train_scaled)
    val_dataset = GreyThicknessDataset(X_val_scaled, y_val_scaled)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            
            # Loss here is MSE on SCALED values. We want to read unscaled MSE for true understanding.
            unscaled_preds_loss = (preds.detach() * scaler.std_y + scaler.mean_y)
            unscaled_y_loss = (y_batch * scaler.std_y + scaler.mean_y)
            true_mse = torch.mean((unscaled_preds_loss - unscaled_y_loss)**2).item()
            
            train_loss += true_mse * X_batch.size(0)
            
            # Unscale for accurate metrics
            unscaled_preds = unscaled_preds_loss.cpu().numpy()
            unscaled_y = unscaled_y_loss.cpu().numpy()
            
            # X_batch is already scaled. We unscale X:
            e_dg_raw = X_batch.cpu().numpy() * scaler.std_x + scaler.mean_x
            e_dg_col = e_dg_raw[:, 4:5]
            
            acc = calculate_accuracy(
                torch.tensor(unscaled_preds), 
                torch.tensor(unscaled_y), 
                torch.tensor(e_dg_col)
            )
            train_acc += acc * X_batch.size(0)
            
        train_loss /= len(train_loader.dataset)
        train_acc /= len(train_loader.dataset)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                preds = model(X_batch)
                
                # Scaled loss for early stopping logic to remain scale invariant
                scaled_loss = criterion(preds, y_batch)
                
                # Unscaled loss for printing and interpretation
                unscaled_preds_loss = (preds * scaler.std_y + scaler.mean_y)
                unscaled_y_loss = (y_batch * scaler.std_y + scaler.mean_y)
                true_mse = torch.mean((unscaled_preds_loss - unscaled_y_loss)**2).item()
                
                val_loss += true_mse * X_batch.size(0)
                
        val_loss /= len(val_loader.dataset)
        
        # Calculate Validation Accuracy
        model.eval()
        with torch.no_grad():
            X_val_scaled_t = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)
            all_val_preds = model(X_val_scaled_t).cpu().numpy()
            
            unscaled_val_preds = scaler.inverse_transform_y(all_val_preds)
            e_dg_val = scaler.inverse_transform_x(X_val_scaled_t.cpu().numpy())[:, 4:5]
            
            val_acc = calculate_accuracy(
                torch.tensor(unscaled_val_preds),
                torch.tensor(y_val.reshape(-1, 1)),
                torch.tensor(e_dg_val)
            )

        scheduler.step(val_loss)

        # Early Stopping using Validation Loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_model_state = copy.deepcopy(model.state_dict())
            if save_path:
                torch.save(best_model_state, save_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch}. Best Val Loss: {best_val_loss:.6f}")
                break
                
        if epoch % 50 == 0:
            print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.6f} | Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.6f} | Val Acc: {val_acc:.2f}%")
            
    # Load best model
    if best_model_state is not None:
         model.load_state_dict(best_model_state)
         
    return model, scaler
