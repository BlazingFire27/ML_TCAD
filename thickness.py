import os
import re
import pandas as pd
import numpy as np

data_dir = "/Users/palakkshetrapal/ML_TCAD/Data"

file_paths = [
    os.path.join(data_dir, f)
    for f in sorted(os.listdir(data_dir))
    if f.endswith('.csv')
]

print(f"Found {len(file_paths)} CSV files\n")

all_rows = []

for fp in file_paths:
    fname = os.path.basename(fp)
    try:
        df = pd.read_csv(fp, encoding='latin1')
        cols = df.columns.tolist()

        for i in range(0, len(cols) - 1, 2):
            x_col = cols[i]
            y_col = cols[i + 1]

            match = re.search(
                r'Pres_([\d.]+)_O2_([\d.]+)_N2_([\d.]+)_Temp_([\d.]+)_time_([\d.]+)',
                x_col
            )
            if not match:
                continue

            pres, o2, n2, temp, time = [float(v) for v in match.groups()]

            x = pd.to_numeric(df[x_col], errors='coerce')
            y = pd.to_numeric(df[y_col], errors='coerce')
            valid = (~x.isna()) & (~y.isna())
            x = x[valid].values
            y = y[valid].values

            # Oxide region: y > 1
            oxide_mask = y > 1.0
            if oxide_mask.sum() == 0:
                thickness = 0.0
                x_min = 0.0
                x_max = 0.0
            else:
                ox_x = x[oxide_mask]
                x_min = ox_x.min()
                x_max = ox_x.max()
                thickness = x_max - x_min

            all_rows.append({
                'pres':      pres,
                'o2':        o2,
                'n2':        n2,
                'temp':      temp,
                'time':      time,
                'x_min':     round(x_min, 6),
                'x_max':     round(x_max, 6),
                'thickness': round(thickness, 6),
            })

        print(f"  Done: {fname}")

    except Exception as e:
        print(f"  ERROR: {fname} -> {e}")

result_df = pd.DataFrame(all_rows)
result_df = result_df.drop_duplicates()

print(f"\nTotal unique simulation entries: {len(result_df)}")
print(f"\nSample:")
print(result_df.head(10).to_string(index=False))

result_df.to_csv('oxide_thickness.csv', index=False)
print(f"\nSaved to oxide_thickness.csv")