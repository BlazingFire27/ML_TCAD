import torch
import numpy as np
import pandas as pd
from model import PINN


if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print(f"Using {device}")


def evaluate():
    # --- load held-out test cache ---
    print("Loading test cache (held-out simulations)...")
    cache = torch.load('test_cache.pt', weights_only=False)
    inputs    = cache['test_inputs']
    df        = cache['test_df']
    log_y_min = cache['log_y_min']
    log_y_max = cache['log_y_max']
    print(f"Loaded {len(inputs)} test samples from unseen simulations\n")

    # --- load model ---
    checkpoint = torch.load('best_model.pth', map_location=device, weights_only=False)
    model = PINN(input_dim=6, hidden_dim=256, n_layers=6).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Loaded model from epoch {checkpoint['epoch']}, "
          f"val loss: {checkpoint['val_loss']:.4f}\n")

    # --- run predictions ---
    all_preds  = []
    batch_size = 4096
    with torch.no_grad():
        for i in range(0, len(inputs), batch_size):
            batch = inputs[i:i+batch_size].to(device)
            pred  = model(batch).cpu().numpy()
            all_preds.append(pred)

    all_preds    = np.concatenate(all_preds, axis=0).squeeze()
    preds_log_y  = all_preds * (log_y_max - log_y_min) + log_y_min
    actual_log_y = df['log_y'].values

    # --- metrics ---
    abs_err = np.abs(preds_log_y - actual_log_y)
    mae     = abs_err.mean()
    rmse    = np.sqrt(((preds_log_y - actual_log_y) ** 2).mean())

    print("=" * 55)
    print("TEST METRICS — UNSEEN SIMULATIONS (in log10(Y) space)")
    print("=" * 55)
    print(f"  MAE:   {mae:.4f}")
    print(f"  RMSE:  {rmse:.4f}")
    print(f"  Within 0.5 log10: {(abs_err < 0.5).mean()*100:.1f}%")
    print(f"  Within 1.0 log10: {(abs_err < 1.0).mean()*100:.1f}%")
    print(f"  Within 2.0 log10: {(abs_err < 2.0).mean()*100:.1f}%")

    bulk_mask     = actual_log_y == 0.0
    reactive_mask = actual_log_y > 0.0
    bulk_err      = abs_err[bulk_mask]
    reactive_err  = abs_err[reactive_mask]

    print(f"\n{'=' * 55}")
    print("BULK POINTS (logY = 0)")
    print("=" * 55)
    print(f"  Count: {bulk_mask.sum()}")
    print(f"  MAE:   {bulk_err.mean():.4f}")
    print(f"  Within 0.5 log10: {(bulk_err < 0.5).mean()*100:.1f}%")

    print(f"\n{'=' * 55}")
    print("REACTIVE POINTS (logY > 0)")
    print("=" * 55)
    print(f"  Count: {reactive_mask.sum()}")
    print(f"  MAE:   {reactive_err.mean():.4f}")
    print(f"  Within 0.5 log10: {(reactive_err < 0.5).mean()*100:.1f}%")
    print(f"  Within 1.0 log10: {(reactive_err < 1.0).mean()*100:.1f}%")

    print(f"\n{'=' * 55}")
    print("10 RANDOM EXAMPLES FROM UNSEEN SIMULATIONS")
    print("=" * 55)
    print(f"{'Actual logY':>12} {'Pred logY':>12} {'Error':>8} {'Type':>10}")
    print("-" * 55)
    rng = np.random.default_rng(42)
    idx = rng.choice(len(actual_log_y), size=10, replace=False)
    for i in idx:
        actual = actual_log_y[i]
        pred   = preds_log_y[i]
        err    = abs(pred - actual)
        kind   = "bulk" if actual == 0.0 else "reactive"
        print(f"{actual:>12.3f} {pred:>12.3f} {err:>8.3f} {kind:>10}")

    # --- worst predictions ---
    results_df = df.copy()
    results_df['pred_log_y'] = preds_log_y
    results_df['abs_error']  = abs_err

    print(f"\n{'=' * 55}")
    print("WORST PREDICTIONS")
    print("=" * 55)
    bad  = results_df[results_df['abs_error'] > 5.0]
    bad2 = results_df[results_df['abs_error'] > 10.0]
    print(f"  Error > 5:  {len(bad):6d} ({len(bad)/len(results_df)*100:.2f}%)")
    print(f"  Error > 10: {len(bad2):6d} ({len(bad2)/len(results_df)*100:.2f}%)")
    print(f"\n  5 worst predictions:")
    worst = results_df.nlargest(5, 'abs_error')[
        ['log_y', 'pred_log_y', 'abs_error', 'temp', 'o2', 'time']
    ]
    print(worst.to_string(index=False))

    results_df.to_csv('test_predictions.csv', index=False)
    print(f"\nSaved to test_predictions.csv")


if __name__ == '__main__':
    evaluate()