'''
AI was used to assist with plotting functions.
'''

import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Get the data directory:
SCRIPT_DIR = Path(__file__).resolve().parent

# ---------- Distance: y = distance_error vs x = est_distance (linear fit) ----------
dist_data_path = SCRIPT_DIR / 'saved_data' / 'data_distance' / 'error_characterization_log.pkl'
with dist_data_path.open('rb') as f:
    dist_data = pickle.load(f)
df = pd.DataFrame(dist_data)

# Convert to DataFrame
print(f'Loaded {len(df)} samples from {dist_data_path}')
print(df.head())

# Add velocity proxies:
df["est_distance_diff"] = df["est_distance"] - df["est_distance"].shift(1)
df["est_bearing_diff"] = df["est_bearing"] - df["est_bearing"].shift(1)

# Correlation table:
corr_matrix = df.corr(numeric_only=True)
print("Correlation Table:")
print(corr_matrix)

d_measured = df['est_distance'].to_numpy()
d_error =  df['distance_error'].to_numpy()
x = d_measured
y = d_error**2

a2, a1, a0 = np.polyfit(x, y, 2)
y_fit = a2 * x**2 + a1 * x + a0

ss_res = np.sum((y - y_fit) ** 2)
ss_tot = np.sum((y - np.mean(y)) ** 2)
r2 = 1 - ss_res / ss_tot
print(f'Fit: y={a2:.3e}x²+{a1:.3e}x+{a0:.2e},  $R^2$={r2:.4f}')

fig = plt.figure(figsize=(8, 5))
plt.scatter(x, y, s=20, alpha=0.7, edgecolors='k', label='Data')
x_line = np.linspace(x.min(), x.max(), 200)
y_line = a2 * x_line**2 + a1 * x_line + a0
plt.plot(x_line, y_line, 'r-', lw=2, 
    label=f'Fit: y={a2:.3e}x²+{a1:.3e}x+{a0:.2e},  $R^2$={r2:.4f}')
plt.title('Distance Error Squared vs Measured Distance')
plt.xlabel('Measured Distance [m]')
plt.ylabel('Distance Error Squared [m^2]')
plt.grid(True, ls=':')
plt.legend()
plt.tight_layout()
plt.show()

out_distance = SCRIPT_DIR / 'distance_error_quadratic_fit.png'
fig.savefig(out_distance, dpi=150)
print(f"[saved] {out_distance}")

# ---------- Bearing: bearing_error_squared vs |Δ est_bearing| (quadratic fit) ----------
bear_data_path = SCRIPT_DIR / 'saved_data' / 'data_bearing' / 'error_characterization_log.pkl'
with bear_data_path.open('rb') as f:
    bear_data = pickle.load(f)
df = pd.DataFrame(bear_data)

# Convert to DataFrame
print(f'Loaded {len(df)} samples from {bear_data_path}')
print(df.head())

# Add velocity proxies:
df["est_distance_diff"] = df["est_distance"] - df["est_distance"].shift(1)
df["est_bearing_diff"] = df["est_bearing"] - df["est_bearing"].shift(1)

# Correlation table:
corr_matrix = df.corr(numeric_only=True)
print("Correlation Table:")
print(corr_matrix)

d_est = df['est_bearing'].to_numpy()
d_err = df['bearing_error'].to_numpy()

# Δ between consecutive measured bearings (proxy for velocity);
d_prior    = d_est[:-2]
d_measured = d_est[1:-1]
x = d_measured - d_prior         
y = d_err[1:-1]**2                            

a2, a1, a0 = np.polyfit(x, y, 2)
y_fit = a2 * x**2 + a1 * x + a0

ss_res = np.sum((y - y_fit) ** 2)
ss_tot = np.sum((y - np.mean(y)) ** 2)
r2 = 1 - ss_res / ss_tot
print(f"[Bearing] Quadratic fit: y = {a2:.6f}x² + {a1:.6f}x + {a0:.6f} | R² = {r2:.6f}")

fig = plt.figure(figsize=(8, 5))
plt.scatter(x, y, s=20, alpha=0.7, edgecolors='k', label='Data')
x_line = np.linspace(x.min(), x.max(), 200)
y_line = a2 * x_line**2 + a1 * x_line + a0
plt.plot(x_line, y_line, 'r-', lw=2,
         label=f'Fit: y={a2:.3e}x²+{a1:.3e}x+{a0:.2e},  $R^2$={r2:.4f}')
plt.title('Bearing Error^2 vs (Measured Bearing - Prior Bearing) (Quadratic Fit)')
plt.xlabel('(Measured Bearing - Prior Bearing) [rad]')
plt.ylabel('Bearing Error Squared [rad^2]')
plt.grid(True, ls=':')
plt.legend()
plt.tight_layout()
plt.show()

out_bearing = SCRIPT_DIR / 'bearing_error_quadratic_fit.png'
fig.savefig(out_bearing, dpi=150)
print(f"[saved] {out_bearing}")
