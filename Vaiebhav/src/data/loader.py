import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

class OxidationDataLoader(Dataset):

    def __init__(self, file_paths, reactive_threshold: float = 2.0, extra_points: int = 10):
        self.file_paths = list(file_paths)
        self.reactive_threshold = reactive_threshold
        self.extra_points = int(extra_points)

        # load and process all files
        self.df = self._load_and_process()

        # prepare numpy arrays for fast __getitem__
        self.features = self.df[['X', 'Temperature', 'O2 Flow', 'N2 Flow', 'Time']].values.astype(np.float32)
        self.targets  = self.df[['logY']].values.astype(np.float32)

    def _load_and_process(self):

        df_list = []
        for f in self.file_paths:
            df = pd.read_csv(f)

            # if 'Step (n)' in df.columns:
            #     df = df.drop(columns=['Step (n)'])

            cols_list = df.columns.tolist()
            # coerce to numeric (cleaning step assumed done upstream but safe to coerce)
            for c in cols_list:
                df[c] = pd.to_numeric(df[c], errors='coerce')

            # drop rows missing essential numeric fields
            df = df.dropna(subset=['X', 'Y', 'Time'])

            # ensure Y > 0 for log; replace zeros/negatives with tiny epsilon
            df.loc[df['Y'] <= 0, 'Y'] = 1e-9

            df_list.append(df)

        if len(df_list) == 0:
            raise RuntimeError("No files read. Check file_paths list.")

        df_all = pd.concat(df_list, ignore_index=True)

        # 2) compute logY only
        df_all['logY'] = np.log10(np.clip(df_all['Y'].values, 1e-9, None))

        # 3) find global interface X_max (deepest reacted X where logY > threshold)
        reactive_mask = df_all['logY'] > self.reactive_threshold
        if reactive_mask.sum() == 0:
            raise ValueError("No reactive points found.")
        
        X_max_interface = float(df_all.loc[reactive_mask, 'X'].max())
        self.X_max_interface = X_max_interface

        # 4) For each Time: sort by X and slice up to the index where X = X_max_interface + extra_points
        kept_blocks = []
        for df in df_list:
            # compute logY per-file (not strictly necessary now, but keep for clarity)
            df['logY'] = np.log10(np.clip(df['Y'].values, 1e-9, None))

            # preserve step occurrence order using drop_duplicates on Step (n) which keeps first appearances in file order
            steps_in_file = df['Step (n)'].unique()
            for s in steps_in_file:
                block = df[df['Step (n)'] == s]   # preserves original row order inside this file block

                # find the last relative index in block where X <= global X_max_interface
                # (per your guarantee, such an index exists)
                le_indices = np.where(block['X'].to_numpy() <= X_max_interface)[0]
                interface_rel_idx = int(le_indices.max())

                cutoff_rel_idx = interface_rel_idx + self.extra_points
                if cutoff_rel_idx > len(block) - 1:
                    cutoff_rel_idx = len(block) - 1

                sliced_block = block.iloc[: cutoff_rel_idx + 1 ].copy()

                # drop Step (n) as requested
                sliced_block = sliced_block.drop(columns=['Step (n)'])

                kept_blocks.append(sliced_block)

        df_final = pd.concat(kept_blocks, ignore_index=True)

        # ensure logY column exists (it does)
        df_final['logY'] = np.log10(np.clip(df_final['Y'].values, 1e-9, None))

        # shuffle to break ordering bias
        df_final = df_final.sample(frac=1.0, random_state=42).reset_index(drop=True)

        print(f"[OxidationDataLoader] global X_max_interface = {self.X_max_interface:.6g}, extra_points = {self.extra_points}")
        return df_final        

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.features[idx])
        y = torch.from_numpy(self.targets[idx])
        return x, y

    # convenience helper
    def get_tensors(self):
        return torch.tensor(self.features, dtype=torch.float32), torch.tensor(self.targets, dtype=torch.float32)