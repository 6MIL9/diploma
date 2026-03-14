import os
from glob import glob

import numpy as np
import pandas as pd

import skfmm

# input_dir = "raw_data"
# output_dir = "data_phi_new"
input_dir = "eval_raw"
output_dir = "eval"
file_pattern = "*.csv"

os.makedirs(output_dir, exist_ok=True)
files = sorted(glob(os.path.join(input_dir, file_pattern)))
print(f"Found {len(files)} files")

# Если шаг сетки известен точно:
H_DEFAULT = 0.00625

for path in files:
    df = pd.read_csv(path)

    # --- column name (more robust) ---
    alpha_col = None
    for col in df.columns:
        c = col.lower()
        if c in ["alpha", "alpha.water", "alpha_water"]:
            alpha_col = col
            break
    if alpha_col is None:
        raise ValueError(f"No alpha column found in {path}. Columns: {list(df.columns)}")

    # --- grid coords ---
    x = np.sort(df["Points:0"].unique())
    y = np.sort(df["Points:1"].unique())

    # оценка шага; используем точный, если совпадает
    dx = float(np.mean(np.diff(x))) if len(x) > 1 else H_DEFAULT
    dy = float(np.mean(np.diff(y))) if len(y) > 1 else H_DEFAULT

    # если нужно принудительно использовать известный h:
    # dx = dy = H_DEFAULT

    # --- pivot -> 2D alpha(y, x) ---
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

    # --- optional: convert to smooth level-set-like field in [-1,1] ---
    # s controls interface thickness in meters (1/s ~ thickness scale).
    # Example: thickness ~ 2h -> s = 1/(2h)
    h = min(dx, dy)
    s = 1.0 / (2.0 * h)  # e.g. for h=0.00625 => s=80
    phi_grid = np.tanh(s * d_grid)

    # --- flatten consistent with sorting y then x ---
    df_sorted = df.sort_values(["Points:1", "Points:0"]).reset_index(drop=True)
    df_sorted["alpha"] = d_grid.ravel()
    df_sorted["phi"] = phi_grid.ravel()

    # --- save ---
    filename = os.path.basename(path)
    out_path = os.path.join(output_dir, filename)
    df_sorted.to_csv(out_path, index=False)

    print(f"Processed: {filename}")

print("Done.")