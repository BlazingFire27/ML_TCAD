"""Explore all unique pressure, O2, N2, temperature, and time values across all TCAD data files."""
import pandas as pd
import re
import glob
import os

data_dir_new = r"d:\Machine Learning+AI\SOP Fabrication Tech\Latest data\data\23rdJan2026"
data_dir_old = r"d:\Machine Learning+AI\SOP Fabrication Tech\Latest data\data\old"

files_new = sorted(glob.glob(os.path.join(data_dir_new, "*.csv")))
files_old = sorted(glob.glob(os.path.join(data_dir_old, "*.csv")))
print(f"New data files: {len(files_new)}")
print(f"Old data files: {len(files_old)}")

all_params = []
pressure_per_file = {}

for f in files_new + files_old:
    cols = pd.read_csv(f, nrows=0).columns.tolist()
    fname = os.path.basename(f)
    file_pres = set()
    for col in cols:
        m = re.search(r"Pres_([\d.]+)_O2_([\d.]+)_N2_([\d.]+)_Temp_([\d.]+)_time_([\d.]+)", col)
        if m:
            pres = float(m.group(1))
            o2 = float(m.group(2))
            n2 = float(m.group(3))
            temp = float(m.group(4))
            time_val = float(m.group(5))
            all_params.append((pres, o2, n2, temp, time_val))
            file_pres.add(pres)
    pressure_per_file[fname] = file_pres

unique_pres = sorted(set(p[0] for p in all_params))
unique_o2 = sorted(set(p[1] for p in all_params))
unique_n2 = sorted(set(p[2] for p in all_params))
unique_temp = sorted(set(p[3] for p in all_params))
unique_time = sorted(set(p[4] for p in all_params))
recipes = sorted(set((p[0], p[1], p[2], p[3]) for p in all_params))

print(f"\n=== DATA PARAMETER DIVERSITY ===")
print(f"Total data header columns parsed: {len(all_params)}")
print(f"Unique (P, O2, N2, T, t) tuples: {len(set(all_params))}")
print(f"Unique recipes (P, O2, N2, T):    {len(recipes)}")
print(f"\nUnique Pressures ({len(unique_pres)}): {unique_pres}")
print(f"\nUnique O2 Flows  ({len(unique_o2)}):   {unique_o2}")
print(f"\nUnique N2 Flows  ({len(unique_n2)}):   {unique_n2}")
print(f"\nUnique Temps     ({len(unique_temp)}):  {unique_temp}")
print(f"\nUnique Times     ({len(unique_time)}):  total {len(unique_time)}, range [{min(unique_time):.2f}, {max(unique_time):.2f}]")

# Show which files have which pressure
print(f"\n=== PRESSURE PER FILE ===")
for pres_val in unique_pres:
    files_with = [f for f, ps in pressure_per_file.items() if pres_val in ps]
    print(f"  P={pres_val}: {len(files_with)} files")
    for fname in files_with[:5]:
        print(f"    {fname}")
    if len(files_with) > 5:
        print(f"    ...and {len(files_with)-5} more")
