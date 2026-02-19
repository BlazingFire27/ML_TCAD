import os
import re
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


def parse_column_name(col_name):
    pattern = r'Pres_([\d.]+)_O2_([\d.]+)_N2_([\d.]+)_Temp_([\d.]+)_time_([\d.]+)'
    match = re.search(pattern, col_name)
    if not match:
        return None
    return {
        'pres': float(match.group(1)),
        'o2':   float(match.group(2)),
        'n2':   float(match.group(3)),
        'temp': float(match.group(4)),
        'time': float(match.group(5)),
    }


def load_single_file(filepath, max_bulk_per_sim=5):
    try:
        df = pd.read_csv(filepath, header=0)
    except Exception as e:
        print(f"  Warning: could not read {filepath}: {e}")
        return []

    cols = df.columns.tolist()
    records = []
    x_cols = [c for c in cols if c.strip().endswith(' X')]

    for x_col in x_cols:
        y_col = x_col.strip()[:-1] + 'Y'
        matching_y = [c for c in cols if c.strip() == y_col.strip()]
        if not matching_y:
            continue
        y_col = matching_y[0]

        params = parse_column_name(x_col)
        if params is None:
            continue

        x_vals = pd.to_numeric(df[x_col], errors='coerce')
        y_vals = pd.to_numeric(df[y_col], errors='coerce')
        valid = x_vals.notna() & y_vals.notna()
        x_vals = x_vals[valid].values
        y_vals = y_vals[valid].values

        if len(x_vals) == 0:
            continue

        bulk_mask     = (y_vals == 1.0)
        reactive_mask = ~bulk_mask

        for x, y in zip(x_vals[reactive_mask], y_vals[reactive_mask]):
            records.append({**params, 'x': x, 'y': y})

        bulk_x = x_vals[bulk_mask]
        bulk_y = y_vals[bulk_mask]
        if len(bulk_x) > max_bulk_per_sim:
            idx = np.random.choice(len(bulk_x), size=max_bulk_per_sim, replace=False)
            bulk_x = bulk_x[idx]
            bulk_y = bulk_y[idx]
        for x, y in zip(bulk_x, bulk_y):
            records.append({**params, 'x': x, 'y': y})

    return records


class OxidationDataLoader(Dataset):
    def __init__(self, file_paths, max_bulk_per_sim=5, max_total_samples=500000, seed=42):
        np.random.seed(seed)
        print(f"Loading {len(file_paths)} files...")

        all_records = []
        for i, fp in enumerate(file_paths):
            records = load_single_file(fp, max_bulk_per_sim=max_bulk_per_sim)
            all_records.extend(records)
            if (i + 1) % 10 == 0:
                print(f"  Processed {i+1}/{len(file_paths)} files, "
                      f"{len(all_records)} points so far")

        if len(all_records) == 0:
            raise ValueError("No data loaded. Check your file paths and format.")

        df = pd.DataFrame(all_records)
        print(f"\nTotal points loaded: {len(df)}")

        # log-transform Y, drop unphysical negative log values
        epsilon = 1e-9
        df['y'] = df['y'].clip(lower=epsilon)
        df['log_y'] = np.log10(df['y'])
        df = df[df['log_y'] >= 0].reset_index(drop=True)
        print(f"After dropping negative logY: {len(df)} points")

        # ----------------------------------------------------------------
        # Stratified sampling — always keep ALL reactive points,
        # sample bulk evenly across simulations to fill remaining budget.
        # ----------------------------------------------------------------
        if max_total_samples is not None and len(df) > max_total_samples:
            sim_cols = ['pres', 'o2', 'n2', 'temp', 'time']

            df_reactive = df[df['log_y'] > 0].reset_index(drop=True)
            df_bulk     = df[df['log_y'] == 0].reset_index(drop=True)

            n_reactive    = len(df_reactive)
            n_bulk_target = max(0, max_total_samples - n_reactive)

            print(f"\nKeeping all {n_reactive} reactive points")
            print(f"Sampling {n_bulk_target} bulk points from "
                  f"{len(df_bulk)} available")

            groups  = df_bulk.groupby(sim_cols)
            n_sims  = groups.ngroups
            per_sim = max(1, n_bulk_target // n_sims)

            sampled_bulk = [
                g.sample(min(len(g), per_sim), random_state=seed)
                for _, g in groups
            ]
            df_bulk_sampled = pd.concat(sampled_bulk).reset_index(drop=True)

            df = pd.concat([df_reactive, df_bulk_sampled]).reset_index(drop=True)
            print(f"After sampling: {len(df)} points")

        # show final split
        n_bulk     = (df['log_y'] == 0.0).sum()
        n_reactive = (df['log_y'] > 0.0).sum()
        print(f"\n  Bulk (logY=0):     {n_bulk}")
        print(f"  Reactive (logY>0): {n_reactive}")

        # ----------------------------------------------------------------
        # Normalize log_y to [0, 1] range.
        #
        # Without this, MSE operates on a 0-16 scale and the two clusters
        # (bulk at 0, reactive at 14-16) are so far apart that the network
        # can't learn both well simultaneously.
        #
        # After normalization:
        #   bulk silicon  (logY=0)     -> 0.0
        #   reactive zone (logY=14-16) -> 0.87 - 1.0
        #
        # We save log_y_min and log_y_max so predictions can be converted
        # back to real log10(Y) values later.
        # ----------------------------------------------------------------
        self.log_y_min = float(df['log_y'].min())  # 0.0
        self.log_y_max = float(df['log_y'].max())  # ~16.12
        log_y_range    = self.log_y_max - self.log_y_min

        df['log_y_norm'] = (df['log_y'] - self.log_y_min) / log_y_range

        print(f"\nTarget normalization:")
        print(f"  log_y range: [{self.log_y_min:.2f}, {self.log_y_max:.2f}]")
        print(f"  bulk maps to:     {0.0:.3f}")
        print(f"  reactive maps to: {14.0/log_y_range:.3f} - {self.log_y_max/log_y_range:.3f}")

        # build feature matrix — input order: [x, pres, o2, n2, temp, time]
        feature_cols = ['x', 'pres', 'o2', 'n2', 'temp', 'time']
        features = df[feature_cols].values.astype(np.float32)
        targets  = df['log_y_norm'].values.astype(np.float32)

        # normalize inputs to [0, 1] for training stability
        self.feat_min = features.min(axis=0)
        self.feat_max = features.max(axis=0)
        feat_range    = self.feat_max - self.feat_min
        feat_range[feat_range == 0] = 1.0
        features_norm = (features - self.feat_min) / feat_range

        self.inputs  = torch.tensor(features_norm, dtype=torch.float32)
        self.targets = torch.tensor(targets,       dtype=torch.float32).unsqueeze(-1)
        self.df = df

        print(f"\nDataset ready: {len(self.inputs)} samples, "
              f"{self.inputs.shape[1]} input features")
        print(f"Feature order: {feature_cols}")
        print(f"Target: log_y_norm (normalized log10(Y) in [0,1])")

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

    def get_domain_bounds(self):
        return self.feat_min, self.feat_max

    def denormalize_target(self, y_norm):
        """Convert normalized prediction back to real log10(Y) value."""
        return y_norm * (self.log_y_max - self.log_y_min) + self.log_y_min