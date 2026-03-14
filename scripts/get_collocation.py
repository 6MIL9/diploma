import numpy as np
import pandas as pd
from pathlib import Path
from typing import Iterable, Tuple, Union, Optional

def radical_inverse_base2(i: int) -> float:
    """Van der Corput radical inverse in base 2 for integer i."""
    inv = 0.5
    x = 0.0
    while i > 0:
        x += (i & 1) * inv
        i >>= 1
        inv *= 0.5
    return x

def hammersley_2d_unit(n: int) -> np.ndarray:
    """
    Hammersley points in [0,1]^2, shape (n,2).
    Uses u=(i+0.5)/n to avoid 0/1; v=phi_2(i).
    """
    u = (np.arange(n, dtype=np.float64) + 0.5) / n
    v = np.fromiter((radical_inverse_base2(i) for i in range(n)),
                    dtype=np.float64, count=n)
    return np.stack([u, v], axis=1)

def cranley_patterson_shift(points_uv: np.ndarray, shift_xy: np.ndarray) -> np.ndarray:
    """
    Apply random shift mod 1 (Cranley-Patterson rotation).
    points_uv: (n,2) in [0,1]
    shift_xy: (2,) in [0,1)
    """
    return (points_uv + shift_xy[None, :]) % 1.0

def load_time_from_snapshot(csv_path: Union[str, Path], time_col: str = "Time") -> float:
    df0 = pd.read_csv(csv_path, nrows=1)
    if time_col not in df0.columns:
        raise ValueError(f"Column '{time_col}' not found in {csv_path}. Columns: {list(df0.columns)}")
    return float(df0.loc[0, time_col])

def build_collocation_points(
    snapshot_files: Iterable[Union[str, Path]],
    n_per_snapshot: int = 1000,
    x_bounds: Tuple[float, float] = (0.0, 1.0),
    y_bounds: Tuple[float, float] = (0.0, 2.0),
    time_col: str = "Time",
    base_seed: int = 12345,
) -> pd.DataFrame:
    """
    For each snapshot, generate n_per_snapshot collocation points (x,y,t).
    Spatial points are Hammersley + per-snapshot random shift (mod 1).
    Time is read from snapshot CSV column `time_col`.
    """
    x0, x1 = map(float, x_bounds)
    y0, y1 = map(float, y_bounds)

    snapshot_files = [Path(f) for f in snapshot_files]

    # Base Hammersley pattern (unit square)
    uv0 = hammersley_2d_unit(n_per_snapshot)

    chunks = []
    for k, f in enumerate(snapshot_files):
        t = load_time_from_snapshot(f, time_col=time_col)

        # Deterministic per-snapshot RNG (reproducible across runs)
        # Using snapshot index to keep reproducible even if times are float-ish.
        rng = np.random.default_rng(base_seed + k)

        # Two independent shifts in [0,1)
        shift = rng.random(2, dtype=np.float64)

        uv = cranley_patterson_shift(uv0, shift)

        x = x0 + (x1 - x0) * uv[:, 0]
        y = y0 + (y1 - y0) * uv[:, 1]

        data = {
            "x": x,
            "y": y,
            "t": np.full(n_per_snapshot, t, dtype=np.float64)
        }

        chunks.append(pd.DataFrame(data))

    return pd.concat(chunks, ignore_index=True)

if __name__ == "__main__":
    # --- You provided one example snapshot (t=0) here ---
    example_snapshot = Path('data_phi_tanh/data_0.csv')

    # Put ALL 301 snapshot CSVs in one directory, then set pattern below.
    # By default, we look in the same folder as the example file.
    snapshots_dir = example_snapshot.parent

    # Adjust to match your naming (e.g. "t_*.csv", "snapshot_*.csv", etc.)
    pattern = "*.csv"

    snapshot_files = sorted(snapshots_dir.glob(pattern))
    if len(snapshot_files) < 301:
        raise RuntimeError(
            f"Found only {len(snapshot_files)} CSV files in {snapshots_dir} with pattern '{pattern}'. "
            f"Place all 301 snapshots there (or change pattern)."
        )

    snapshot_files = snapshot_files[:301]  # take first 301 in sorted order

    df_colloc = build_collocation_points(
        snapshot_files=snapshot_files,
        n_per_snapshot=1000,
        x_bounds=(0.0, 1.0),
        y_bounds=(0.0, 2.0),
        time_col="Time",
        base_seed=20260223,
    )
    out_path = "datasets/collocation.csv"

    df_colloc.to_csv(out_path, index=False)

    print(f"Saved: {out_path}")
    print(df_colloc.head())
    print("Total points:", len(df_colloc))