import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
file = os.path.join(script_dir, '../processed/Cleaned_oxi3.csv')

df = pd.read_csv(file)
df['X'] = pd.to_numeric(df['X'], errors='coerce')
df['Y'] = pd.to_numeric(df['Y'], errors='coerce')
df = df.dropna(subset=['X', 'Y'])


print("\n1. DATAFRAME HEAD (First 5 rows):")
print(df.head())

print("\n2. DATAFRAME INFO:")
print(df.info())

print("\n3. STATS (Check min/max to see if physics make sense):")
print(df[['X', 'Y', 'Time']].describe())

unique_times = df['Time'].unique()
unique_times.sort()
print(f"\n--- Animation Setup ---")
print(f"Total Time Steps found: {len(unique_times)}")
print("Opening animation window... (Close window to stop)")

# Setup the plot
fig, ax = plt.subplots(figsize=(10, 6))
line, = ax.plot([], [], lw=2, color='blue', label='Oxidation Profile')
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

# Set static plot limits (so the axis doesn't jump around)
ax.set_xlim(df['X'].min(), df['X'].max())
ax.set_ylim(df['Y'].min(), df['Y'].max() * 1.1) # Add 10% headroom
ax.set_xlabel('X Coordinate (Depth)')
ax.set_ylabel('Y Value (Concentration/Oxide)')
ax.set_title(f'Oxidation Evolution: {file}')
ax.legend(loc='upper right')
ax.grid(True)

# Animation function
def update(frame_idx):
    current_time = unique_times[frame_idx]
    
    # Filter data for this specific time
    # (This is fast because we sorted the file earlier)
    step_data = df[df['Time'] == current_time]
    
    line.set_data(step_data['X'], step_data['Y'])
    time_text.set_text(f'Time: {current_time:.2f} s')
    return line, time_text

# Create Animation
# interval=50 means 50ms per frame (fast playback)
# frames=len(unique_times) loops through all steps
# repeat=True will loop the animation
ani = animation.FuncAnimation(fig, update, frames=len(unique_times), 
                              interval=20, blit=True, repeat=True)

plt.show()