"""
data_preprocessing_v2.py — Reprocess ALL raw TCAD CSVs with Pressure column included.

WHAT:   Reads raw TCAD CSV files, extracts ALL metadata (including Pressure)
        from column headers, and saves cleaned CSVs.
WHY:    The original data_preprocessing.py dropped Pressure from the output.
        We need Pressure as a model feature because 8 distinct pressure values
        exist in the data (0.156 to 0.606 atm), not a single fixed value.
HOW:    Parses regex from column headers: n{step}_Pres_{p}_O2_{o}_N2_{n}_Temp_{T}_time_{t}
        Outputs: Step (n), Pressure, O2 Flow, N2 Flow, Temperature, Time, X, Y
        Saves to: data/new processed (with pressure)/Cleaned_oxi{group}.csv

Author: Vaiebhav Shreevarshan R (2024AAPS1427G)
Date:   2026-03-18
"""

import pandas as pd
import re
import glob
import os

# ─── Configuration ───────────────────────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))

# Both raw data directories
data_dirs = [
    os.path.join(script_dir, '..', '..', 'data', 'old'),              # oxi3–oxi11
    os.path.join(script_dir, '..', '..', 'data', '23rdJan2026'),      # Oxi_12–Oxi_90
]

output_dir = os.path.join(script_dir, '..', '..', 'data', 'new processed (with pressure)')
os.makedirs(output_dir, exist_ok=True)

# Group numbers to process — covers all available data
# Old data: oxi3–oxi11 (raw files named oxi3_*, oxi4_*, etc.)
# New data: Oxi_12–Oxi_90 (raw files named O2_conc_extracted_Oxi_{n}_*.csv)
group_numbers = list(range(3, 91))  # 3 through 90 inclusive


# ─── Header Parser ───────────────────────────────────────────────────────────
def parse_header(header_string):
    """
    Parse metadata from a raw TCAD column header.
    
    Example header:
        O2(n12_Pres_0.444858_O2_1.243631_N2_23.847076_Temp_964.102600_time_142.722137_fps) X
    
    Returns dict with Step(n), Pressure, O2 Flow, N2 Flow, Temperature, Time.
    Returns None if the header doesn't match the expected pattern.
    """
    pattern = r"n(\d+)_Pres_([\d\.]+)_O2_([\d\.]+)_N2_([\d\.]+)_Temp_([\d\.]+)_time_([\d\.]+)"
    match = re.search(pattern, header_string)
    if match:
        return {
            'Step (n)':     int(match.group(1)),
            'Pressure':     float(match.group(2)),
            'O2 Flow':      float(match.group(3)),
            'N2 Flow':      float(match.group(4)),
            'Temperature':  float(match.group(5)),
            'Time':         float(match.group(6)),
        }
    return None


# ─── Group Processor ─────────────────────────────────────────────────────────
def find_files_for_group(group_num):
    """
    Find all raw CSV files that contain data for a given oxi group number.
    Handles both naming conventions:
      - Old: oxi{n}_1_300.csv, oxi{n}_301_600.csv, etc.
      - New: O2_conc_extracted_Oxi_{n}.csv
    """
    all_files = []
    for data_dir in data_dirs:
        if not os.path.isdir(data_dir):
            continue
        # Old naming: oxi{n}_*.csv
        pattern_old = os.path.join(data_dir, f"oxi{group_num}_*.csv")
        all_files.extend(glob.glob(pattern_old))
        # New naming: *_Oxi_{n}.csv  or  *_Oxi_{n}_*.csv
        pattern_new = os.path.join(data_dir, f"*Oxi_{group_num}.csv")
        all_files.extend(glob.glob(pattern_new))
        pattern_new2 = os.path.join(data_dir, f"*Oxi_{group_num}_*.csv")
        all_files.extend(glob.glob(pattern_new2))
    # Deduplicate
    return sorted(set(all_files))


def process_group(group_num):
    """Process all files for a single oxi group and save cleaned CSV."""
    files = find_files_for_group(group_num)
    if not files:
        return None  # No files found for this group — silently skip

    group_data = []
    
    for filepath in files:
        try:
            df_raw = pd.read_csv(filepath)
        except Exception as e:
            print(f"  ERROR reading {os.path.basename(filepath)}: {e}")
            continue

        # Iterate over column pairs (X, Y)
        for i in range(0, len(df_raw.columns), 2):
            if i + 1 >= len(df_raw.columns):
                break

            col_x_header = df_raw.columns[i]
            col_y_header = df_raw.columns[i + 1]

            meta = parse_header(col_x_header)
            if not meta:
                continue

            # Extract (X, Y) data, drop NaN rows
            step_df = df_raw[[col_x_header, col_y_header]].dropna()
            step_df.columns = ['X', 'Y']

            # Add metadata columns (including Pressure)
            step_df['Step (n)']     = meta['Step (n)']
            step_df['Pressure']     = meta['Pressure']
            step_df['O2 Flow']      = meta['O2 Flow']
            step_df['N2 Flow']      = meta['N2 Flow']
            step_df['Temperature']  = meta['Temperature']
            step_df['Time']         = meta['Time']

            group_data.append(step_df)

    if not group_data:
        return None

    # Concatenate all data for this group
    final_df = pd.concat(group_data, ignore_index=True)

    # Reorder columns: metadata first, then spatial data
    desired_order = ['Step (n)', 'Pressure', 'O2 Flow', 'N2 Flow', 'Temperature', 'Time', 'X', 'Y']
    final_df = final_df[desired_order]

    # Sort for consistent ordering
    final_df = final_df.sort_values(by=['Step (n)', 'Time', 'X'])

    # Save
    output_file = os.path.join(output_dir, f"Cleaned_oxi{group_num}.csv")
    final_df.to_csv(output_file, index=False)

    return {
        'group': group_num,
        'files_read': len(files),
        'rows': len(final_df),
        'unique_steps': final_df['Step (n)'].nunique(),
        'unique_pressures': sorted(final_df['Pressure'].unique().tolist()),
        'output_file': output_file,
    }


# ─── Main ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 70)
    print("DATA PREPROCESSING V2 — WITH PRESSURE COLUMN")
    print("=" * 70)
    print(f"Data source directories:")
    for d in data_dirs:
        exists = os.path.isdir(d)
        print(f"  {os.path.abspath(d)} {'✓' if exists else '✗ NOT FOUND'}")
    print(f"Output directory: {os.path.abspath(output_dir)}")
    print()

    total_rows = 0
    total_files = 0
    groups_processed = 0
    groups_skipped = 0
    all_pressures = set()

    for group_num in group_numbers:
        result = process_group(group_num)
        if result is None:
            groups_skipped += 1
            continue

        groups_processed += 1
        total_rows += result['rows']
        total_files += result['files_read']
        all_pressures.update(result['unique_pressures'])

        print(f"  oxi{result['group']:3d}: {result['rows']:7d} rows | "
              f"{result['unique_steps']:4d} steps | "
              f"P={result['unique_pressures']} | "
              f"{result['files_read']} files")

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Groups processed: {groups_processed}")
    print(f"  Groups skipped:   {groups_skipped} (no files found)")
    print(f"  Total rows:       {total_rows:,}")
    print(f"  Total raw files:  {total_files}")
    print(f"  Unique pressures: {sorted(all_pressures)}")
    print(f"  Output directory: {os.path.abspath(output_dir)}")
    print("=" * 70)
