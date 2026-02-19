import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from model import PINN
import os


# -----------------------------------------------------------------------
# Device setup
# -----------------------------------------------------------------------
if torch.backends.mps.is_available():
    device = torch.device('mps')
    print("Using MPS (Apple Silicon GPU)")
elif torch.cuda.is_available():
    device = torch.device('cuda')
    print("Using CUDA GPU")
else:
    device = torch.device('cpu')
    print("Using CPU")


def train():
    # --- load cache ---
    print("Loading cached dataset...")
    cache = torch.load('dataset_cache.pt', weights_only=False)

    train_inputs  = cache['train_inputs']
    train_targets = cache['train_targets']
    val_inputs    = cache['val_inputs']
    val_targets   = cache['val_targets']
    feat_min      = cache['feat_min']
    feat_max      = cache['feat_max']
    log_y_min     = cache['log_y_min']
    log_y_max     = cache['log_y_max']

    print(f"Train: {len(train_inputs)} samples")
    print(f"Val:   {len(val_inputs)} samples")

    train_loader = DataLoader(
        TensorDataset(train_inputs, train_targets),
        batch_size=2048, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        TensorDataset(val_inputs, val_targets),
        batch_size=2048, shuffle=False, num_workers=0
    )

    # --- model ---
    model = PINN(input_dim=6, hidden_dim=256, n_layers=6).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # start fresh — old checkpoint used random split so weights aren't valid
    save_path = 'best_model.pth'
    best_val  = float('inf')
    print("Starting fresh (simulation-level split requires retraining)\n")

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    epochs    = 500
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5
    )
    loss_fn = nn.MSELoss()

    print(f"Training for {epochs} epochs...\n")

    for epoch in range(1, epochs + 1):
        # --- train ---
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        scheduler.step()

        # --- validate ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                val_loss += loss_fn(model(batch_x), batch_y).item()
        val_loss /= len(val_loader)

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                'epoch':                epoch,
                'model_state_dict':     model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss':             best_val,
                'feat_min':             feat_min,
                'feat_max':             feat_max,
                'log_y_min':            log_y_min,
                'log_y_max':            log_y_max,
            }, save_path)

        current_lr = scheduler.get_last_lr()[0]
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{epochs} | "
                  f"Train: {train_loss:.4f} | "
                  f"Val: {val_loss:.4f} | "
                  f"Best: {best_val:.4f} | "
                  f"LR: {current_lr:.2e}")

    print(f"\nTraining complete. Best val loss: {best_val:.4f}")
    print(f"Saved to {save_path}")
    print(f"\nNow run evaluate.py to test on held-out simulations.")


if __name__ == '__main__':
    train()