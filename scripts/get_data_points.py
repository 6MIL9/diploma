import os
import re
from glob import glob
from pathlib import Path
from typing import Iterable, Optional, Tuple, Union

import numpy as np
import pandas as pd


def radical_inverse_base2(i: int) -> float:
    inv = 0.5
    x = 0.0
    while i > 0:
        x += (i & 1) * inv
        i >>= 1
        inv *= 0.5
    return x


def hammersley_2d_unit(n: int) -> np.ndarray:
    u = (np.arange(n, dtype=np.float64) + 0.5) / n
    v = np.fromiter((radical_inverse_base2(i) for i in range(n)),
                    dtype=np.float64, count=n)
    return np.stack([u, v], axis=1)


def cranley_patterson_shift(points_uv: np.ndarray, shift_uv: np.ndarray) -> np.ndarray:
    return (points_uv + shift_uv[None, :]) % 1.0


# -------------------------
# Helpers
# -------------------------
def load_time_from_snapshot(csv_path: Union[str, Path], time_col: str = "Time") -> float:
    df0 = pd.read_csv(csv_path, nrows=1)
    if time_col in df0.columns:
        return float(df0.loc[0, time_col])

    name = Path(csv_path).stem
    m = re.search(r"([-+]?\d*\.\d+|[-+]?\d+)", name)
    if m:
        return float(m.group(0))

    raise ValueError(
        f"Cannot read time: no '{time_col}' column and no number in filename: {csv_path}"
    )


def build_structured_index(df_sorted: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    x_unique = np.sort(df_sorted["Points:0"].unique())
    y_unique = np.sort(df_sorted["Points:1"].unique())
    return x_unique, y_unique


def nearest_grid_indices(
    xq: np.ndarray,
    yq: np.ndarray,
    x_unique: np.ndarray,
    y_unique: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    ix = np.searchsorted(x_unique, xq, side="left")
    ix = np.clip(ix, 0, len(x_unique) - 1)
    ix_left = np.clip(ix - 1, 0, len(x_unique) - 1)
    pick_left = np.abs(xq - x_unique[ix_left]) <= np.abs(xq - x_unique[ix])
    ix = np.where(pick_left, ix_left, ix)

    iy = np.searchsorted(y_unique, yq, side="left")
    iy = np.clip(iy, 0, len(y_unique) - 1)
    iy_left = np.clip(iy - 1, 0, len(y_unique) - 1)
    pick_left = np.abs(yq - y_unique[iy_left]) <= np.abs(yq - y_unique[iy])
    iy = np.where(pick_left, iy_left, iy)

    return ix.astype(np.int64), iy.astype(np.int64)


def flat_index_from_ij(ix: np.ndarray, iy: np.ndarray, nx: int) -> np.ndarray:
    return iy * nx + ix


def sample_without_replacement_weighted(
    rng: np.random.Generator,
    n_total: int,
    k: int,
    weights: np.ndarray,
    exclude: Optional[np.ndarray] = None,
) -> np.ndarray:
    if exclude is None:
        exclude = np.zeros(n_total, dtype=bool)
    else:
        exclude = exclude.astype(bool, copy=False)

    w = weights.astype(np.float64, copy=True)
    w[exclude] = 0.0
    w[w < 0] = 0.0

    s = w.sum()
    if s <= 0:
        candidates = np.flatnonzero(~exclude)
        if len(candidates) == 0:
            return np.array([], dtype=np.int64)
        k_eff = min(k, len(candidates))
        return rng.choice(candidates, size=k_eff, replace=False)

    p = w / s
    candidates = np.flatnonzero(p > 0)
    k_eff = min(k, len(candidates))
    return rng.choice(n_total, size=k_eff, replace=False, p=p)


def sample_boundary_indices(
    rng: np.random.Generator,
    df_sorted: pd.DataFrame,
    n_boundary: int,
    tol: float = 1e-12,
    exclude_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    x = df_sorted["Points:0"].to_numpy(dtype=np.float64)
    y = df_sorted["Points:1"].to_numpy(dtype=np.float64)

    xmin, xmax = float(x.min()), float(x.max())
    ymin, ymax = float(y.min()), float(y.max())

    on_b = (
        (np.abs(x - xmin) <= tol) |
        (np.abs(x - xmax) <= tol) |
        (np.abs(y - ymin) <= tol) |
        (np.abs(y - ymax) <= tol)
    )
    candidates = np.flatnonzero(on_b)

    if exclude_mask is not None:
        candidates = candidates[~exclude_mask[candidates]]

    if len(candidates) == 0:
        return np.array([], dtype=np.int64)

    k_eff = min(n_boundary, len(candidates))
    return rng.choice(candidates, size=k_eff, replace=False)


def sample_volume_hammersley_outside_heavy(
    rng: np.random.Generator,
    df_sorted: pd.DataFrame,
    x_bounds: Tuple[float, float],
    y_bounds: Tuple[float, float],
    n_volume: int,
    phi_col: str = "phi",
    oversample_factor: int = 6,
    base_seed_shift: Optional[np.ndarray] = None,
    exclude_mask: Optional[np.ndarray] = None,
    outside_frac: float = 0.70,
    inside_frac: float = 0.20,
    transition_frac: float = 0.10,
    phi_outside_thr: float = 0.8,
    phi_inside_thr: float = -0.8,
) -> np.ndarray:
    """
    Hammersley volume sampling with emphasis on OUTSIDE points.

    Categories:
      outside:    phi >= phi_outside_thr
      inside:     phi <= phi_inside_thr
      transition: phi_inside_thr < phi < phi_outside_thr

    The goal is to anchor the outer liquid phase more strongly and avoid
    extra spurious negative regions ("phantom bubbles").
    """
    x0, x1 = map(float, x_bounds)
    y0, y1 = map(float, y_bounds)

    if n_volume <= 0:
        return np.array([], dtype=np.int64)

    total_frac = outside_frac + inside_frac + transition_frac
    if not np.isclose(total_frac, 1.0):
        outside_frac = outside_frac / total_frac
        inside_frac = inside_frac / total_frac
        transition_frac = transition_frac / total_frac

    m = max(n_volume * oversample_factor, n_volume)

    uv0 = hammersley_2d_unit(m)
    if base_seed_shift is None:
        shift = rng.random(2, dtype=np.float64)
    else:
        shift = np.asarray(base_seed_shift, dtype=np.float64)
    uv = cranley_patterson_shift(uv0, shift)

    xq = x0 + (x1 - x0) * uv[:, 0]
    yq = y0 + (y1 - y0) * uv[:, 1]

    x_unique, y_unique = build_structured_index(df_sorted)
    nx = len(x_unique)
    ix, iy = nearest_grid_indices(xq, yq, x_unique, y_unique)
    idx = flat_index_from_ij(ix, iy, nx)

    _, first = np.unique(idx, return_index=True)
    idx_unique = idx[np.sort(first)]

    if exclude_mask is not None:
        idx_unique = idx_unique[~exclude_mask[idx_unique]]

    if len(idx_unique) == 0:
        return np.array([], dtype=np.int64)

    phi = df_sorted[phi_col].to_numpy(dtype=np.float64)

    outside = idx_unique[phi[idx_unique] >= phi_outside_thr]
    inside = idx_unique[phi[idx_unique] <= phi_inside_thr]
    transition = idx_unique[
        (phi[idx_unique] > phi_inside_thr) & (phi[idx_unique] < phi_outside_thr)
    ]

    n_out = int(round(n_volume * outside_frac))
    n_in = int(round(n_volume * inside_frac))
    n_tr = n_volume - n_out - n_in

    picked = []

    def take_from(pool: np.ndarray, k: int):
        if k <= 0 or len(pool) == 0:
            return np.array([], dtype=np.int64)
        kk = min(k, len(pool))
        return rng.choice(pool, size=kk, replace=False)

    picked_out = take_from(outside, n_out)
    picked_in = take_from(inside, n_in)
    picked_tr = take_from(transition, n_tr)

    if len(picked_out) > 0:
        picked.append(picked_out)
    if len(picked_in) > 0:
        picked.append(picked_in)
    if len(picked_tr) > 0:
        picked.append(picked_tr)

    picked = np.concatenate(picked) if picked else np.array([], dtype=np.int64)

    if len(picked) < n_volume:
        remaining = n_volume - len(picked)

        # fill priority: outside -> transition -> inside
        rest_out = np.setdiff1d(outside, picked, assume_unique=False)
        rest_tr = np.setdiff1d(transition, picked, assume_unique=False)
        rest_in = np.setdiff1d(inside, picked, assume_unique=False)

        fill_order = [rest_out, rest_tr, rest_in]
        for pool in fill_order:
            if remaining <= 0:
                break
            if len(pool) == 0:
                continue
            take = min(remaining, len(pool))
            add = rng.choice(pool, size=take, replace=False)
            picked = np.concatenate([picked, add])
            remaining = n_volume - len(picked)

    if len(picked) < n_volume:
        rest_candidates = np.setdiff1d(idx_unique, picked, assume_unique=False)
        if len(rest_candidates) > 0:
            take = min(n_volume - len(picked), len(rest_candidates))
            add = rng.choice(rest_candidates, size=take, replace=False)
            picked = np.concatenate([picked, add])

    return picked.astype(np.int64)


def build_data_points_dataset(
    snapshot_files: Iterable[Union[str, Path]],
    out_csv: Union[str, Path],
    n_interface: int = 700,
    n_volume: int = 500,
    n_boundary: int = 100,
    x_bounds: Tuple[float, float] = (0.0, 1.0),
    y_bounds: Tuple[float, float] = (0.0, 2.0),
    phi_col: str = "phi",
    time_col: str = "Time",
    base_seed: int = 20260224,
    delta_gamma: float = 2.0,
    boundary_tol: float = 1e-12,
    outside_frac: float = 0.70,
    inside_frac: float = 0.20,
    transition_frac: float = 0.10,
    phi_outside_thr: float = 0.8,
    phi_inside_thr: float = -0.8,
) -> pd.DataFrame:
    snapshot_files = [Path(f) for f in snapshot_files]
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    chunks = []

    for k, f in enumerate(snapshot_files):
        t = load_time_from_snapshot(f, time_col=time_col)
        rng = np.random.default_rng(base_seed + k)

        df = pd.read_csv(f)

        for c in ["Points:0", "Points:1", phi_col]:
            if c not in df.columns:
                raise ValueError(f"Missing column '{c}' in {f}. Columns: {list(df.columns)}")

        df_sorted = df.sort_values(["Points:1", "Points:0"]).reset_index(drop=True)

        n_total = len(df_sorted)
        if n_total == 0:
            continue

        phi = df_sorted[phi_col].to_numpy(dtype=np.float64)

        # --- 1) Interface points: weighted by delta ≈ 1 - phi^2
        delta = 1.0 - phi * phi
        weights = np.power(np.clip(delta, 0.0, None), float(delta_gamma))

        exclude = np.zeros(n_total, dtype=bool)

        idx_I = sample_without_replacement_weighted(
            rng=rng,
            n_total=n_total,
            k=n_interface,
            weights=weights,
            exclude=exclude
        )
        exclude[idx_I] = True

        # --- 2) Volume points: Hammersley with emphasis on OUTSIDE region
        shift = rng.random(2, dtype=np.float64)
        idx_V = sample_volume_hammersley_outside_heavy(
            rng=rng,
            df_sorted=df_sorted,
            x_bounds=x_bounds,
            y_bounds=y_bounds,
            n_volume=n_volume,
            phi_col=phi_col,
            oversample_factor=8,
            base_seed_shift=shift,
            exclude_mask=exclude,
            outside_frac=outside_frac,
            inside_frac=inside_frac,
            transition_frac=transition_frac,
            phi_outside_thr=phi_outside_thr,
            phi_inside_thr=phi_inside_thr,
        )
        exclude[idx_V] = True

        # --- 3) Boundary points
        idx_B = sample_boundary_indices(
            rng=rng,
            df_sorted=df_sorted,
            n_boundary=n_boundary,
            tol=boundary_tol,
            exclude_mask=exclude
        )
        exclude[idx_B] = True

        idx_all = np.concatenate([idx_I, idx_V, idx_B]).astype(np.int64)

        out = df_sorted.loc[idx_all, ["Points:0", "Points:1", phi_col]].copy()
        out.rename(columns={"Points:0": "x", "Points:1": "y", phi_col: "phi"}, inplace=True)
        out["t"] = t

        chunks.append(out)

        phi_sel = out["phi"].to_numpy(dtype=np.float64)
        n_out_sel = np.sum(phi_sel >= phi_outside_thr)
        n_in_sel = np.sum(phi_sel <= phi_inside_thr)
        n_tr_sel = np.sum((phi_sel > phi_inside_thr) & (phi_sel < phi_outside_thr))

        print(
            f"{f.name}: t={t:g} | "
            f"interface={len(idx_I)} volume={len(idx_V)} boundary={len(idx_B)} | "
            f"outside={n_out_sel} inside={n_in_sel} transition={n_tr_sel} | total={len(idx_all)}"
        )

    df_all = pd.concat(chunks, ignore_index=True)
    df_all.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv} | total points: {len(df_all)}")
    return df_all


if __name__ == "__main__":
    snapshots_dir = Path("data_phi_new")
    pattern = "*.csv"
    snapshot_files = sorted(snapshots_dir.glob(pattern))
    if len(snapshot_files) == 0:
        raise RuntimeError(f"No CSV files found in {snapshots_dir} with pattern '{pattern}'")

    df_data = build_data_points_dataset(
        snapshot_files=snapshot_files,
        out_csv="datasets/data_points_phi.csv",
        n_interface=400,
        n_volume=800,
        n_boundary=150,
        x_bounds=(0.0, 1.0),
        y_bounds=(0.0, 2.0),
        phi_col="phi",
        time_col="Time",
        base_seed=20260224,
        delta_gamma=2.0,
        boundary_tol=1e-12,

        # main change: more points outside bubble
        outside_frac=0.75,
        inside_frac=0.15,
        transition_frac=0.10,

        # thresholds for region split
        phi_outside_thr=0.8,
        phi_inside_thr=-0.8,
    )

    print(df_data.head())