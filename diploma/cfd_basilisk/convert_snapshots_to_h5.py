from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


DEFAULT_PRESSURE_OFFSET = 6116.7114937818405


def load_snapshot(path: Path) -> tuple[float, np.ndarray]:
    data = np.loadtxt(path, comments="#")
    if data.ndim != 2 or data.shape[1] != 7:
        raise ValueError(f"Unexpected snapshot format in {path}")
    return float(data[0, 0]), data


def chamfer_distance(mask: np.ndarray, dx: float, dy: float) -> np.ndarray:
    dist = np.where(mask, 0.0, np.inf)
    diagonal = float(np.hypot(dx, dy))
    nx, ny = dist.shape

    for i in range(nx):
        for j in range(ny):
            value = dist[i, j]
            if i > 0:
                value = min(value, dist[i - 1, j] + dx)
            if j > 0:
                value = min(value, dist[i, j - 1] + dy)
            if i > 0 and j > 0:
                value = min(value, dist[i - 1, j - 1] + diagonal)
            if i > 0 and j < ny - 1:
                value = min(value, dist[i - 1, j + 1] + diagonal)
            dist[i, j] = value

    for i in range(nx - 1, -1, -1):
        for j in range(ny - 1, -1, -1):
            value = dist[i, j]
            if i < nx - 1:
                value = min(value, dist[i + 1, j] + dx)
            if j < ny - 1:
                value = min(value, dist[i, j + 1] + dy)
            if i < nx - 1 and j < ny - 1:
                value = min(value, dist[i + 1, j + 1] + diagonal)
            if i < nx - 1 and j > 0:
                value = min(value, dist[i + 1, j - 1] + diagonal)
            dist[i, j] = value

    return dist


def signed_levelset_from_alpha(alpha: np.ndarray, dx: float, dy: float) -> np.ndarray:
    bubble = alpha > 0.5
    if not np.any(bubble) or np.all(bubble):
        raise ValueError("Snapshot does not contain both phases.")
    dist_to_bubble = chamfer_distance(bubble, dx, dy)
    dist_to_liquid = chamfer_distance(~bubble, dx, dy)
    half_cell = 0.5 * min(dx, dy)
    outside = np.maximum(dist_to_bubble - half_cell, 0.0)
    inside = -np.maximum(dist_to_liquid - half_cell, 0.0)
    return np.where(bubble, inside, outside)


def apply_pressure_gauge(
    pressure: np.ndarray,
    y: np.ndarray,
    times: np.ndarray,
    mode: str,
    pressure_offset: float,
    pressure_bottom: float,
    rho_liquid: float,
    gravity: float,
    reference_path: Path | None,
) -> np.ndarray:
    if mode == "raw":
        return pressure
    if mode == "absolute":
        adjusted = pressure + pressure_offset
        zero_time = np.isclose(times, 0.0)
        if np.any(zero_time):
            hydrostatic = pressure_bottom - rho_liquid * abs(gravity) * y
            adjusted[zero_time] = hydrostatic[None, None, :]
        return adjusted
    if mode == "reference-mean":
        if reference_path is None:
            raise ValueError("--reference-path is required for --pressure-mode reference-mean")
        with h5py.File(reference_path, "r") as f:
            reference_pressure = np.asarray(f["pressure"])
        if reference_pressure.shape != pressure.shape:
            raise ValueError(
                f"Reference pressure shape {reference_pressure.shape} does not match candidate shape {pressure.shape}"
            )
        offset = reference_pressure.mean(axis=(1, 2), keepdims=True) - pressure.mean(axis=(1, 2), keepdims=True)
        return pressure + offset
    raise ValueError(f"Unknown pressure mode: {mode}")


def snapshots_to_h5(
    input_dir: Path,
    output: Path,
    pressure_mode: str = "absolute",
    pressure_offset: float = DEFAULT_PRESSURE_OFFSET,
    pressure_bottom: float = 7000.0,
    rho_liquid: float = 1000.0,
    gravity: float = 0.98,
    reference_path: Path | None = None,
) -> None:
    paths = sorted(input_dir.glob("snapshot-*.tsv"))
    if not paths:
        raise FileNotFoundError(f"No snapshot-*.tsv files found in {input_dir}")

    loaded = [load_snapshot(path) for path in paths]
    times = np.asarray([item[0] for item in loaded], dtype=np.float64)
    first = loaded[0][1]
    vertical = np.unique(first[:, 1])
    horizontal_half = np.unique(first[:, 2])
    dx = float(np.diff(vertical).mean())
    dy = float(np.diff(horizontal_half).mean())

    x_full = np.concatenate([-horizontal_half[::-1], horizontal_half])
    y_full = vertical
    shape = (len(times), len(x_full), len(y_full))
    levelset = np.empty(shape, dtype=np.float64)
    density = np.empty(shape, dtype=np.float64)
    pressure = np.empty(shape, dtype=np.float64)
    velocity_x = np.empty(shape, dtype=np.float64)
    velocity_y = np.empty(shape, dtype=np.float64)

    for it, (_, data) in enumerate(loaded):
        ix = np.searchsorted(vertical, data[:, 1])
        iy = np.searchsorted(horizontal_half, data[:, 2])
        values = np.empty((len(vertical), len(horizontal_half), 7), dtype=np.float64)
        values[ix, iy] = data
        alpha_liquid_half = values[:, :, 3]
        u_vertical_half = values[:, :, 4]
        u_horizontal_half = values[:, :, 5]
        p_half = values[:, :, 6]

        alpha_bubble_half = 1.0 - alpha_liquid_half
        alpha_bubble_full = np.concatenate([alpha_bubble_half[:, ::-1], alpha_bubble_half], axis=1).T
        u_x_full = np.concatenate([-u_horizontal_half[:, ::-1], u_horizontal_half], axis=1).T
        u_y_full = np.concatenate([u_vertical_half[:, ::-1], u_vertical_half], axis=1).T
        p_full = np.concatenate([p_half[:, ::-1], p_half], axis=1).T

        levelset[it] = signed_levelset_from_alpha(alpha_bubble_full, dy, dx)
        density[it] = 1000.0 + (100.0 - 1000.0) * alpha_bubble_full
        pressure[it] = p_full
        velocity_x[it] = u_x_full
        velocity_y[it] = u_y_full

    pressure = apply_pressure_gauge(
        pressure=pressure,
        y=y_full,
        times=times,
        mode=pressure_mode,
        pressure_offset=pressure_offset,
        pressure_bottom=pressure_bottom,
        rho_liquid=rho_liquid,
        gravity=gravity,
        reference_path=reference_path,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output, "w") as f:
        f.create_dataset("X", data=x_full)
        f.create_dataset("Y", data=y_full)
        f.create_dataset("time", data=times)
        f.create_dataset("levelset", data=levelset)
        f.create_dataset("density", data=density)
        f.create_dataset("pressure", data=pressure)
        f.create_dataset("velocityX", data=velocity_x)
        f.create_dataset("velocityY", data=velocity_y)
        f.attrs["source"] = "Basilisk rising_bubble.c snapshots"
        f.attrs["pressure_mode"] = pressure_mode
        f.attrs["pressure_offset"] = pressure_offset
        f.attrs["pressure_bottom"] = pressure_bottom


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Basilisk rising-bubble snapshots to PINN HDF5 format.")
    parser.add_argument("--input-dir", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("../cfd_data/rising_bubble_basilisk.h5"))
    parser.add_argument(
        "--pressure-mode",
        choices=["absolute", "raw", "reference-mean"],
        default="absolute",
        help=(
            "absolute writes pressure in the same gauge as cfd_data/rising_bubble.h5; "
            "raw stores Basilisk pressure; reference-mean aligns each frame to --reference-path."
        ),
    )
    parser.add_argument("--pressure-offset", type=float, default=DEFAULT_PRESSURE_OFFSET)
    parser.add_argument("--pressure-bottom", type=float, default=7000.0)
    parser.add_argument("--rho-liquid", type=float, default=1000.0)
    parser.add_argument("--gravity", type=float, default=0.98)
    parser.add_argument("--reference-path", type=Path)
    args = parser.parse_args()
    snapshots_to_h5(
        input_dir=args.input_dir,
        output=args.output,
        pressure_mode=args.pressure_mode,
        pressure_offset=args.pressure_offset,
        pressure_bottom=args.pressure_bottom,
        rho_liquid=args.rho_liquid,
        gravity=args.gravity,
        reference_path=args.reference_path,
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
