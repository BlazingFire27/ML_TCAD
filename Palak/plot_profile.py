import torch
import numpy as np
import matplotlib.pyplot as plt
from model import PINN


if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def plot_profile():
    # --- load cache ---
    cache = torch.load('dataset_cache.pt', weights_only=False)
    df        = cache['df']
    feat_min  = cache['feat_min']
    feat_max  = cache['feat_max']
    log_y_min = cache['log_y_min']
    log_y_max = cache['log_y_max']

    # --- load model ---
    checkpoint = torch.load('best_model.pth', map_location=device, weights_only=False)
    model = PINN(input_dim=6, hidden_dim=256, n_layers=6).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # --- pick a random simulation with at least 10 data points ---
    sim_cols = ['pres', 'o2', 'n2', 'temp', 'time']
    sims     = df.groupby(sim_cols)
    sim_keys = [k for k, g in sims if len(g) >= 10]

    if not sim_keys:
        sim_keys = list(sims.groups.keys())

    rng     = np.random.default_rng()
    sim_key = sim_keys[rng.integers(len(sim_keys))]
    sim_df  = sims.get_group(sim_key).copy().sort_values('x')

    pres, o2, n2, temp, time = sim_key
    print(f"Selected simulation:")
    print(f"  Pressure:    {pres:.4f}")
    print(f"  O2 Flow:     {o2:.4f}")
    print(f"  N2 Flow:     {n2:.4f}")
    print(f"  Temperature: {temp:.2f}")
    print(f"  Time:        {time:.2f}")
    print(f"  Data points: {len(sim_df)}")

    # --- dense prediction curve ---
    x_raw   = sim_df['x'].values
    x_dense = np.linspace(x_raw.min(), x_raw.max(), 300)

    feat_dense = np.column_stack([
        x_dense,
        np.full_like(x_dense, pres),
        np.full_like(x_dense, o2),
        np.full_like(x_dense, n2),
        np.full_like(x_dense, temp),
        np.full_like(x_dense, time),
    ]).astype(np.float32)

    feat_range = feat_max - feat_min
    feat_range[feat_range == 0] = 1.0
    feat_dense_norm = (feat_dense - feat_min) / feat_range

    with torch.no_grad():
        inputs_t   = torch.tensor(feat_dense_norm).to(device)
        preds_norm = model(inputs_t).cpu().numpy().squeeze()

    preds_log_y = preds_norm * (log_y_max - log_y_min) + log_y_min

    # --- predict on actual data points ---
    feat_actual = np.column_stack([
        sim_df['x'].values,
        np.full(len(sim_df), pres),
        np.full(len(sim_df), o2),
        np.full(len(sim_df), n2),
        np.full(len(sim_df), temp),
        np.full(len(sim_df), time),
    ]).astype(np.float32)
    feat_actual_norm = (feat_actual - feat_min) / feat_range

    with torch.no_grad():
        inputs_t2    = torch.tensor(feat_actual_norm).to(device)
        preds_actual = model(inputs_t2).cpu().numpy().squeeze()

    preds_actual_log_y = preds_actual * (log_y_max - log_y_min) + log_y_min

    # --- plot ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(x_dense * 1e3, preds_log_y, color='royalblue',
            linewidth=2, label='PINN prediction', zorder=3)
    ax.scatter(sim_df['x'].values * 1e3, sim_df['log_y'].values,
               color='tomato', s=40, zorder=4, label='Simulation data', alpha=0.8)
    ax.set_xlabel('Depth (mm)', fontsize=12)
    ax.set_ylabel('log₁₀(Y)', fontsize=12)
    ax.set_title(f'Oxide Concentration Profile\n'
                 f'T={temp:.0f}K  O2={o2:.2f}  t={time:.1f}s', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.scatter(sim_df['log_y'].values, preds_actual_log_y,
                color='royalblue', s=40, alpha=0.8)
    all_vals = np.concatenate([sim_df['log_y'].values, preds_actual_log_y])
    lims = [all_vals.min() - 0.5, all_vals.max() + 0.5]
    ax2.plot(lims, lims, 'k--', linewidth=1, alpha=0.5, label='Perfect prediction')
    ax2.set_xlim(lims)
    ax2.set_ylim(lims)
    ax2.set_xlabel('Actual log₁₀(Y)', fontsize=12)
    ax2.set_ylabel('Predicted log₁₀(Y)', fontsize=12)
    ax2.set_title('Predicted vs Actual', fontsize=11)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('profile_plot.png', dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to profile_plot.png")
    plt.show()


if __name__ == '__main__':
    plot_profile()