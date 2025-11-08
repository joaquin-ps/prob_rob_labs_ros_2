'''
AI was used to assist with plotting functions. 
'''

import pickle
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Get the data directory:
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / 'saved_data' / 'data_distance'
data_path = DATA_DIR / 'error_characterization_log.pkl'

# Load the pickle data
with data_path.open('rb') as f:
    data = pickle.load(f)

# Convert to DataFrame
df = pd.DataFrame(data)

print(f'Loaded {len(df)} samples from {data_path}')
print(df.head())

# --- Plot: x = est_distance, y = distance_error ---
fig = plt.figure(figsize=(8, 5))
plt.scatter(df['est_distance'], df['distance_error'], s=20, alpha=0.7, edgecolors='k')

plt.title('Distance Error vs Measured Distance')
plt.xlabel('Measured Distance [m]')
plt.ylabel('Distance Error [m]')
plt.grid(True)
plt.tight_layout()
plt.show()
out_distance = SCRIPT_DIR / 'distance_error.png'
fig.savefig(out_distance, dpi=150)
plt.close(fig)
print(f"[saved] {out_distance}")



# Get the data directory:
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / 'saved_data' / 'data_bearing'
data_path = DATA_DIR / 'error_characterization_log.pkl'

# Load the pickle data
with data_path.open('rb') as f:
    data = pickle.load(f)

# Convert to DataFrame
df = pd.DataFrame(data)

print(df.head())
print(len(df['est_distance'].to_list()))


# --- 3-D scatter plot: z=bearing_error, x=est_distance, y=est_bearing ---
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

fig = plt.figure(figsize=(9, 6))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(
    df['est_distance'].to_numpy(),
    df['est_bearing'].to_numpy(),
    df['bearing_error'].to_numpy(),
    c=df['bearing_error'].to_numpy(),
    cmap='viridis',
    s=25,
    alpha=0.85,
    edgecolors='none',
)
ax.set_xlabel('Measured Distance [m]')
ax.set_ylabel('Measured Bearing [rad]')
ax.set_zlabel('Bearing Error [rad]')
ax.set_title('Bearing Error vs Measured Distance and Bearing')
fig.colorbar(sc, ax=ax, shrink=0.8, pad=0.1, label='Bearing Error [rad]')
plt.tight_layout()
plt.show()
out3d = SCRIPT_DIR / 'bearing_error_3d.png'
fig.savefig(out3d, dpi=150)
plt.close(fig)
print(f"[saved] {out3d}")