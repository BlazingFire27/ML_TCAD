import os
import numpy as np
from data_loader import OxidationDataLoader

input_dir = "/Users/palakkshetrapal/ML_TCAD/Data"

file_paths = [
    os.path.join(input_dir, f)
    for f in os.listdir(input_dir)
    if f.endswith('.csv')
]

print(f"Found {len(file_paths)} files")

cache = torch.load('dataset_cache.pt', weights_only=False)
# then access cache['inputs'], cache['feat_min'] etc directly

print("log_y stats:")
print(f"  min:    {dataset.df['log_y'].min():.2f}")
print(f"  max:    {dataset.df['log_y'].max():.2f}")
print(f"  mean:   {dataset.df['log_y'].mean():.2f}")
print(f"  median: {dataset.df['log_y'].median():.2f}")

import pandas as pd

bins = [0, 1, 5, 10, 13, 14, 15, 16, 17]
labels = ['0-1', '1-5', '5-10', '10-13', '13-14', '14-15', '15-16', '16+']
counts = pd.cut(dataset.df['log_y'], bins=bins).value_counts().sort_index()
print("\nlog_y distribution:")
for label, count in zip(labels, counts):
    bar = '█' * (count // 5000)
    print(f"  {label:6s}: {count:6d}  {bar}")

# pick one simulation and show all its raw points
sim = dataset.df[
    (dataset.df['temp'].between(920, 922)) &
    (dataset.df['o2'].between(3.5, 3.6))
]
print(f"\nAll points for T≈921K O2≈3.53:")
print(f"Unique simulations found: {sim[['temp','o2','time']].drop_duplicates()}")
print(sim[['x', 'log_y', 'temp', 'o2', 'time']].sort_values('x').to_string())