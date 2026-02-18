import pandas as pd
import re
import glob
import os

# groups = ['oxi3', 'oxi4', 'oxi5', 'oxi6', 'oxi7', 'oxi8', 'oxi9', 'oxi10', 'oxi11']

# groups = [f'oxi{i}' for i in range(12, 31)]
group_numbers = range(31, 52)

script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, '../23rdJan2026')
output_dir = os.path.join(script_dir, '../processed')

# "O2(n12_Pres_0.444858_O2_1.243631_N2_23.847076_Temp_964.102600_time_142.722137_fps) X"

def parse_header(header_string):
    pattern = r"n(\d+)_Pres_([\d\.]+)_O2_([\d\.]+)_N2_([\d\.]+)_Temp_([\d\.]+)_time_([\d\.]+)"

    match = re.search(pattern, header_string)
    if match:
        return {
            'Step (n)': int(match.group(1)),
            'O2 Flow': float(match.group(3)),
            'N2 Flow': float(match.group(4)),
            'Temperature': float(match.group(5)),
            'Time': float(match.group(6))
        }
    return None

def process_group_data(group):
    print(f"Processing group: {group}")

    file_pattern = os.path.join(data_dir, f"*Oxi_{group}*.csv")
    files = glob.glob(file_pattern)

    if not files:
        print(f"No files found for group: {group}")
        return
    
    group_data = []

    for file in files:
        try:
            df_raw = pd.read_csv(file)
        
        except Exception as e:
            print(f"Error reading {file}: {e}")
            continue

        for i in range(0, len(df_raw.columns), 2):
            if i + 1 >= len(df_raw.columns):
                break

            col_x_header = df_raw.columns[i]
            col_y_header = df_raw.columns[i + 1]

            meta = parse_header(col_x_header)
            if not meta:
                continue

            step_df = df_raw[[col_x_header, col_y_header]].dropna()
            step_df.columns = ['X', 'Y']

            step_df['Step (n)'] = meta['Step (n)']
            step_df['O2 Flow'] = meta['O2 Flow']
            step_df['N2 Flow'] = meta['N2 Flow']
            step_df['Temperature'] = meta['Temperature']
            step_df['Time'] = meta['Time']

            group_data.append(step_df)

    if group_data:
        final_df = pd.concat(group_data, ignore_index=True)
        desired_order = ['Step (n)', 'O2 Flow', 'N2 Flow', 'Temperature', 'Time', 'X', 'Y']
        final_df = final_df[desired_order]

        final_df = final_df.sort_values(by = ['Step (n)', 'Time', 'X'])

        output_file = os.path.join(output_dir, f"Cleaned_oxi{group}.csv")
        final_df.to_csv(output_file, index=False)

        print(f"Success! Saved {output_file} with {len(final_df)} rows.")

    else:
        print(f"No valid data found for group: {group}")

if __name__ == "__main__":
    for group in group_numbers:
        process_group_data(group)                                                                                                                                                                 ## Vaiebhav Shreevarshan R, 2024AAPS1427G