import os
import numpy as np
import pandas as pd
from glob import glob
import skfmm

path = "data_phi_new/data_1.csv"
df = pd.read_csv(path)
H_DEFAULT = 0.00625
alpha_col = None
for col in df.columns:
    c = col.lower()
    if c in ["alpha", "alpha.water", "alpha_water"]:
        alpha_col = col
        break

# --- grid coords ---
x = np.sort(df["Points:0"].unique())
y = np.sort(df["Points:1"].unique())

# оценка шага; используем точный, если совпадает
dx = float(np.mean(np.diff(x))) if len(x) > 1 else H_DEFAULT
dy = float(np.mean(np.diff(y))) if len(y) > 1 else H_DEFAULT
print(dx,dy)
# если нужно принудительно использовать известный h:
# dx = dy = H_DEFAULT
alpha_grid = (
    df.pivot(index="Points:1", columns="Points:0", values=alpha_col)
    .sort_index(ascending=True)
    .sort_index(axis=1, ascending=True)
    .values
)

if np.isnan(alpha_grid).any():
    raise ValueError(
        f"NaNs in alpha_grid for {path}. "
        f"Likely missing points / non-rectangular sampling."
    )

# --- Fast Marching: signed distance to the 0-level set of phi0 ---
# phi0 = alpha - 0.5, so phi0=0 corresponds to VOF interface alpha=0.5
phi0 = alpha_grid - 0.5

# skfmm expects dx as scalar (uniform) or tuple; we pass tuple (dy, dx) for [y,x] axes
d_grid = skfmm.distance(phi0, dx=(dy, dx))  # meters; positive where phi0>0 (water)
h = min(dx, dy)
s = 1.0 / (2.0 * h)  # e.g. for h=0.00625 => s=80
print(s)
phi_grid = np.tanh(s * d_grid)
mask_interface = np.abs(d_grid) < 5*h  # узкая зона вокруг интерфейса

phi_vals = phi_grid[mask_interface]
d_vals = d_grid[mask_interface]

# находим где phi близко к +0.9 и -0.9
pos = d_vals[np.abs(phi_vals - 0.9) < 0.02]
neg = d_vals[np.abs(phi_vals + 0.9) < 0.02]

thickness = np.mean(pos) - np.mean(neg)
print("Толщина (м):", thickness)
print("Толщина в клетках:", thickness / H_DEFAULT)