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
print("Checking 44% rule: 44% of oxide grows INTO silicon (x>0), 56% grows OUTSIDE (x<0)\n")

results = []

for fp in file_paths:
    fname = os.path.basename(fp)
    try:
        df = pd.read_csv(fp, encoding='latin1')
        cols = df.columns.tolist()

        n_ok = 0
        # Columns alternate X, Y, X, Y...
        for i in range(0, len(cols) - 1, 2):
            x_col = cols[i]
            y_col = cols[i + 1]

            # Parse simulation params from column name
            match = re.search(
                r'Pres_([\d.]+)_O2_([\d.]+)_N2_([\d.]+)_Temp_([\d.]+)_time_([\d.]+)',
                x_col
            )
            if match:
                pres, o2, n2, temp, time = [float(v) for v in match.groups()]
            else:
                pres, o2, n2, temp, time = None, None, None, None, None

            x = pd.to_numeric(df[x_col], errors='coerce')
            y = pd.to_numeric(df[y_col], errors='coerce')
            valid = (~x.isna()) & (~y.isna())
            x = x[valid].values
            y = y[valid].values

            oxide_mask = y > 1.0
            if oxide_mask.sum() == 0:
                results.append({'file': fname, 'status': 'no_oxide', 'pres': pres, 'o2': o2, 'n2': n2, 'temp': temp, 'time': time})
                continue

            ox_x = x[oxide_mask]
            x_min = ox_x.min()
            x_max = ox_x.max()

            outside_thickness = abs(min(x_min, 0))
            inside_thickness  = max(x_max, 0)
            total_thickness   = outside_thickness + inside_thickness

            if total_thickness == 0:
                results.append({'file': fname, 'status': 'zero_thickness'})
                continue

            pct_inside  = 100 * inside_thickness  / total_thickness
            pct_outside = 100 * outside_thickness / total_thickness
            n_ok += 1

            results.append({
                'file':        fname,
                'status':      'ok',
                'pres':        pres,
                'o2':          o2,
                'n2':          n2,
                'temp':        temp,
                'time':        time,
                'x_min':       round(x_min, 4),
                'x_max':       round(x_max, 4),
                'outside_um':  round(outside_thickness, 4),
                'inside_um':   round(inside_thickness, 4),
                'total_um':    round(total_thickness, 4),
                'pct_inside':  round(pct_inside, 1),
                'pct_outside': round(pct_outside, 1),
            })

        print(f"  Done: {fname} ({n_ok} sims OK)")

    except Exception as e:
        print(f"  ERROR: {fname} -> {e}")
        results.append({'file': fname, 'status': f'error: {e}'})

df_res = pd.DataFrame(results)
ok = df_res[df_res['status'] == 'ok']

print(f"\n{'='*60}")
print(f"SUMMARY")
print(f"{'='*60}")
print(f"Total simulations processed : {len(df_res)}")
print(f"OK                          : {len(ok)}")
print(f"No oxide                    : {(df_res['status'] == 'no_oxide').sum()}")
print(f"Errors                      : {df_res['status'].str.startswith('error').sum()}")

print(f"\n{'='*60}")
print("% OXIDE INSIDE SILICON (44% expected)")
print(f"{'='*60}")
print(f"  Mean   : {ok['pct_inside'].mean():.1f}%")
print(f"  Median : {ok['pct_inside'].median():.1f}%")
print(f"  Std    : {ok['pct_inside'].std():.1f}%")
print(f"  Min    : {ok['pct_inside'].min():.1f}%")
print(f"  Max    : {ok['pct_inside'].max():.1f}%")

print(f"\n  Distribution of % inside silicon:")
bins   = [0, 20, 35, 40, 42, 44, 46, 48, 50, 55, 60, 80, 100]
labels = ['0-20','20-35','35-40','40-42','42-44','44-46','46-48','48-50','50-55','55-60','60-80','80-100']
counts = pd.cut(ok['pct_inside'], bins=bins).value_counts().sort_index()
for label, count in zip(labels, counts):
    bar = '█' * min(count, 60)
    print(f"  {label:8s}: {count:5d}  {bar}")

print(f"\n{'='*60}")
print("SIMULATIONS DEVIATING MORE THAN 5% FROM 44%")
print(f"{'='*60}")
deviating = ok[abs(ok['pct_inside'] - 44) > 5].sort_values('pct_inside')
if len(deviating) == 0:
    print("  None! All simulations follow the 44% rule closely.")
else:
    print(f"  Count: {len(deviating)} ({100*len(deviating)/len(ok):.1f}% of total)")
    print(deviating[['file','pres','temp','time','pct_inside','inside_um','outside_um','total_um']].head(20).to_string(index=False))

df_res.to_csv('44_percent_check.csv', index=False)
print(f"\nFull results saved to 44_percent_check.csv")