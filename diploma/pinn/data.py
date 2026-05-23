from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import re
from collections.abc import Sequence

import h5py
import numpy as np
import torch

from .config import PointConfig


@dataclass
class TrainingData:
    alpha: torch.Tensor
    pde: torch.Tensor
    north: torch.Tensor
    east_west: torch.Tensor
    nsew: torch.Tensor

    def to(self, device: torch.device, dtype: torch.dtype) -> "TrainingData":
        return TrainingData(
            alpha=self.alpha.to(device=device, dtype=dtype),
            pde=self.pde.to(device=device, dtype=dtype),
            north=self.north.to(device=device, dtype=dtype),
            east_west=self.east_west.to(device=device, dtype=dtype),
            nsew=self.nsew.to(device=device, dtype=dtype),
        )

    @property
    def sizes(self) -> dict[str, int]:
        return {
            "alpha": len(self.alpha),
            "pde": len(self.pde),
            "north": len(self.north),
            "east_west": len(self.east_west),
            "nsew": len(self.nsew),
        }


def _choice(rng: np.random.Generator, values: np.ndarray, size: int, replace: bool = False) -> np.ndarray:
    if size <= len(values):
        return rng.choice(values, size, replace=replace)
    return rng.choice(values, size, replace=True)


def _points_for_interface(
    rng: np.random.Generator,
    n_points: int,
    t_value: float,
    interface: list[tuple[float, float]],
    normal: list[tuple[float, float]],
    tangent: list[tuple[float, float]],
    cell_size: float,
    refine_start: float,
    refine_end: float,
) -> np.ndarray:
    if not interface:
        raise ValueError("No interface cells were detected in the selected CFD snapshot.")

    per_cell = math.ceil(n_points / (2 * len(interface)))
    points_inward: list[np.ndarray] = []
    points_outward: list[np.ndarray] = []

    for x_p, n_p, t_p in zip(interface, normal, tangent):
        x_p = np.asarray(x_p, dtype=np.float64)
        n_p = np.asarray(n_p, dtype=np.float64)
        t_p = np.asarray(t_p, dtype=np.float64)
        inward_n = rng.uniform(refine_start, refine_end, per_cell)
        outward_n = rng.uniform(refine_start, refine_end, per_cell)
        inward_t1 = rng.uniform(refine_start, refine_end, per_cell)
        inward_t2 = rng.uniform(refine_start, refine_end, per_cell)
        outward_t1 = rng.uniform(refine_start, refine_end, per_cell)
        outward_t2 = rng.uniform(refine_start, refine_end, per_cell)

        for in_n, in_t1, in_t2, out_n, out_t1, out_t2 in zip(
            inward_n, inward_t1, inward_t2, outward_n, outward_t1, outward_t2
        ):
            points_inward.append(x_p + in_n * n_p)
            points_inward.append(x_p + cell_size / 3.0 * t_p + in_t1 * n_p)
            points_inward.append(x_p + 2.0 * cell_size / 3.0 * t_p + in_t2 * n_p)
            points_outward.append(x_p - out_n * n_p)
            points_outward.append(x_p + cell_size / 3.0 * t_p - out_t1 * n_p)
            points_outward.append(x_p + 2.0 * cell_size / 3.0 * t_p - out_t2 * n_p)

    inward = np.hstack(
        [
            np.asarray(points_inward),
            t_value * np.ones((len(points_inward), 1)),
            np.ones((len(points_inward), 1)),
        ]
    )
    outward = np.hstack(
        [
            np.asarray(points_outward),
            t_value * np.ones((len(points_outward), 1)),
            np.zeros((len(points_outward), 1)),
        ]
    )
    data = np.vstack([inward, outward])
    return data[_choice(rng, np.arange(len(data)), n_points), :]


def _points_for_domain(
    rng: np.random.Generator,
    n_points: int,
    t_value: float,
    x: np.ndarray,
    y: np.ndarray,
    levelset: np.ndarray,
    cell_size: float,
    max_levelset: float,
) -> np.ndarray:
    per_x = math.ceil(n_points / len(x))
    data = np.empty((0, 4), dtype=np.float64)
    levelset_threshold = max_levelset * cell_size

    for index_x, x_value in enumerate(x):
        t_domain = t_value * np.ones((per_x, 1))
        x_domain = x_value * np.ones((per_x, 1))
        y_chunks: list[np.ndarray] = []
        a_chunks: list[np.ndarray] = []
        remaining = per_x

        while remaining:
            indices_y = _choice(rng, np.arange(len(y)), remaining)
            y_temp = y[indices_y]
            levelset_temp = levelset[index_x, indices_y]
            a_temp = (levelset_temp > 0).astype(np.float64)
            keep = np.where(np.abs(levelset_temp) >= levelset_threshold)[0]
            if len(keep) == 0:
                continue
            y_chunks.append(y_temp[keep])
            a_chunks.append(a_temp[keep])
            remaining -= len(keep)

        y_domain = np.hstack(y_chunks).reshape(per_x, 1)
        a_domain = np.hstack(a_chunks).reshape(per_x, 1)
        data = np.vstack([data, np.hstack([x_domain, y_domain, t_domain, a_domain])])

    return data[_choice(rng, np.arange(len(data)), n_points), :]


def _compute_normals(
    x: np.ndarray, y: np.ndarray, levelset: np.ndarray, cell_size: float
) -> tuple[list, list, list]:
    interfaces: list[list[tuple[float, float]]] = []
    normals: list[list[tuple[float, float]]] = []
    tangents: list[list[tuple[float, float]]] = []

    for levelset_t in levelset:
        ix, iy = np.where(np.abs(levelset_t) <= 0.75 * cell_size)
        valid = (ix > 0) & (ix < len(x) - 1) & (iy > 0) & (iy < len(y) - 1)
        ix = ix[valid]
        iy = iy[valid]

        interfaces.append(list(zip(x[ix], y[iy])))
        ls_x = (levelset_t[ix + 1, iy] - levelset_t[ix - 1, iy]) / (2.0 * cell_size)
        ls_y = (levelset_t[ix, iy + 1] - levelset_t[ix, iy - 1]) / (2.0 * cell_size)
        grad_abs = np.sqrt(ls_x**2 + ls_y**2)
        grad_abs[grad_abs == 0.0] = np.finfo(float).eps
        normal_x = ls_x / grad_abs
        normal_y = ls_y / grad_abs
        normals.append(list(zip(normal_x, normal_y)))
        tangents.append(list(zip(-normal_y, normal_x)))

    return interfaces, normals, tangents


def _points_alpha(
    rng: np.random.Generator,
    cfg: tuple[int, int],
    times: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    cell_size: float,
    interfaces: list,
    normals: list,
    tangents: list,
    levelset: np.ndarray,
) -> np.ndarray:
    chunks = []
    for vals in zip(times, interfaces, normals, tangents, levelset):
        t_value, interface, normal, tangent, levelset_t = vals
        chunks.append(_points_for_interface(rng, cfg[0], t_value, interface, normal, tangent, cell_size, 0.004, 0.008))
        chunks.append(_points_for_domain(rng, cfg[1], t_value, x, y, levelset_t, cell_size, 4.0))
    return np.vstack(chunks)


def _points_pde(
    rng: np.random.Generator,
    cfg: tuple[int, int, int],
    times: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    cell_size: float,
    interfaces: list,
    normals: list,
    tangents: list,
    levelset: np.ndarray,
) -> np.ndarray:
    chunks = []
    for vals in zip(times, interfaces, normals, tangents, levelset):
        t_value, interface, normal, tangent, levelset_t = vals
        chunks.append(_points_for_interface(rng, cfg[0], t_value, interface, normal, tangent, cell_size, 0.0, 0.001))
        chunks.append(_points_for_interface(rng, cfg[1], t_value, interface, normal, tangent, cell_size, 0.001, 0.1))
        chunks.append(_points_for_domain(rng, cfg[2], t_value, x, y, levelset_t, cell_size, 4.0))
    data = np.vstack(chunks)
    data[:, 3] = 0.0
    return data


def _north(rng: np.random.Generator, cfg: tuple[int, int], x_bounds, y_bounds, t_bounds) -> np.ndarray:
    low_t = t_bounds[0] + np.finfo(float).eps
    times = np.hstack([0.0, rng.uniform(low_t, t_bounds[1], max(cfg[1] - 2, 0)), t_bounds[1]])
    data = np.empty((0, 6), dtype=np.float64)
    for t_value in times:
        xs = rng.uniform(x_bounds[0], x_bounds[1], (cfg[0], 1))
        ys = y_bounds[1] * np.ones((cfg[0], 1))
        data = np.vstack([data, np.hstack([xs, ys, t_value * np.ones((cfg[0], 1)), np.zeros((cfg[0], 2)), np.ones((cfg[0], 1))])])
    return data


def _wall(cfg: tuple[int, int], x_bounds, y_bounds, t_bounds, side: str) -> np.ndarray:
    times = np.linspace(t_bounds[0], t_bounds[1], cfg[1])
    data = np.empty((0, 5), dtype=np.float64)
    for t_value in times:
        if side == "south":
            xs = np.linspace(x_bounds[0], x_bounds[1], cfg[0]).reshape(-1, 1)
            ys = y_bounds[0] * np.ones((cfg[0], 1))
        elif side == "east":
            xs = x_bounds[1] * np.ones((cfg[0], 1))
            ys = np.linspace(y_bounds[0], y_bounds[1], cfg[0]).reshape(-1, 1)
        elif side == "west":
            xs = x_bounds[0] * np.ones((cfg[0], 1))
            ys = np.linspace(y_bounds[0], y_bounds[1], cfg[0]).reshape(-1, 1)
        else:
            raise ValueError(side)
        data = np.vstack([data, np.hstack([xs, ys, t_value * np.ones((cfg[0], 1)), np.zeros((cfg[0], 2))])])
    return data


def selected_time_indices(n_times: int) -> np.ndarray:
    indices = np.arange(n_times)
    return np.sort(np.concatenate([indices[0:30:15], indices[30:100:5], indices[100::5], indices[1:3]], axis=0))


def resolve_data_paths(data_path: Path | Sequence[Path]) -> list[Path]:
    if isinstance(data_path, (str, Path)):
        path = Path(data_path)
        if path.is_dir():
            paths = sorted(path.glob("*.h5"))
            if not paths:
                raise FileNotFoundError(f"No .h5 files found in directory: {path}")
            return paths
        return [path]
    return [Path(path) for path in data_path]


def radius_from_h5_path(data_path: Path, default: float = 0.25) -> float:
    with h5py.File(data_path, "r") as data:
        for key in ("radius", "R"):
            if key in data.attrs:
                return float(data.attrs[key])

    match = re.search(r"(?:^|[_-])R(\d+(?:\.\d+)?)", data_path.stem, flags=re.IGNORECASE)
    if match:
        value = match.group(1)
        radius = float(value)
        if radius >= 1.0 and "." not in value:
            radius /= 100.0
        return radius

    return default


def _append_radius_column(data: np.ndarray, radius: float, target_start: int = 3) -> np.ndarray:
    radius_col = radius * np.ones((len(data), 1), dtype=data.dtype)
    return np.hstack([data[:, :target_start], radius_col, data[:, target_start:]])


def _concat_training_data(chunks: list[TrainingData]) -> TrainingData:
    return TrainingData(
        alpha=torch.cat([chunk.alpha for chunk in chunks], dim=0),
        pde=torch.cat([chunk.pde for chunk in chunks], dim=0),
        north=torch.cat([chunk.north for chunk in chunks], dim=0),
        east_west=torch.cat([chunk.east_west for chunk in chunks], dim=0),
        nsew=torch.cat([chunk.nsew for chunk in chunks], dim=0),
    )


def make_training_data(
    data_path: Path | Sequence[Path],
    cfg: PointConfig,
    l_ref: float,
    seed: int = 1234,
    parameterized: bool = True,
) -> TrainingData:
    data_paths = resolve_data_paths(data_path)
    if not parameterized and len(data_paths) != 1:
        raise ValueError("Ordinary PINN training requires exactly one HDF5 data file.")
    chunks = [
        _make_training_data_single(path, cfg, l_ref, seed + index, parameterized)
        for index, path in enumerate(data_paths)
    ]
    return _concat_training_data(chunks)


def _make_training_data_single(
    data_path: Path,
    cfg: PointConfig,
    l_ref: float,
    seed: int,
    parameterized: bool,
) -> TrainingData:
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(
            f"CFD file not found: {data_path}. Download rising_bubble.h5 and pass --data-path if needed."
        )

    rng = np.random.default_rng(seed)
    radius = radius_from_h5_path(data_path) / l_ref
    with h5py.File(data_path, "r") as data:
        x = np.asarray(data["X"])
        y = np.asarray(data["Y"])
        times = np.asarray(data["time"])
        levelset = -np.asarray(data["levelset"])

    indices = selected_time_indices(len(times))
    times = times[indices]
    levelset = levelset[indices]
    x_bounds = (x[0], x[-1])
    y_bounds = (y[0], y[-1])
    t_bounds = (times[0], times[-1])
    cell_size = float(np.diff(x)[0])

    interfaces, normals, tangents = _compute_normals(x, y, levelset, cell_size)
    alpha = _points_alpha(rng, cfg.alpha, times, x, y, cell_size, interfaces, normals, tangents, levelset)
    pde = _points_pde(rng, cfg.pde, times, x, y, cell_size, interfaces, normals, tangents, levelset)
    north = _north(rng, cfg.north, x_bounds, y_bounds, t_bounds)
    south = _wall(cfg.south, x_bounds, y_bounds, t_bounds, "south")
    east = _wall(cfg.east, x_bounds, y_bounds, t_bounds, "east")
    west = _wall(cfg.west, x_bounds, y_bounds, t_bounds, "west")

    nsew_coords = np.vstack([north[:, 0:3], south[:, 0:3], east[:, 0:3], west[:, 0:3]])
    nsew_for_pde = np.hstack([nsew_coords, np.zeros((len(nsew_coords), 1))])

    for arr in (alpha, pde, north, south, east, west, nsew_coords, nsew_for_pde):
        arr[:, :3] /= l_ref

    nsew_raw = np.vstack([north[:, 0:5], south, east[: cfg.east[0]], west[: cfg.west[0]]])

    if parameterized:
        alpha = _append_radius_column(alpha, radius)
        pde = _append_radius_column(pde, radius)
        nsew_for_pde = _append_radius_column(nsew_for_pde, radius)
        east_west = np.hstack(
            [
                east[:, 0:3],
                radius * np.ones((len(east), 1), dtype=east.dtype),
                west[:, 0:3],
                radius * np.ones((len(west), 1), dtype=west.dtype),
            ]
        )
        nsew = _append_radius_column(nsew_raw, radius)
        north_out = _append_radius_column(north[:, [0, 1, 2, 5]], radius)
    else:
        east_west = np.hstack([east[:, 0:3], west[:, 0:3]])
        nsew = nsew_raw
        north_out = north[:, [0, 1, 2, 5]]

    pde = np.vstack([pde, nsew_for_pde])

    return TrainingData(
        alpha=torch.from_numpy(alpha.astype(np.float32)),
        pde=torch.from_numpy(pde.astype(np.float32)),
        north=torch.from_numpy(north_out.astype(np.float32)),
        east_west=torch.from_numpy(east_west.astype(np.float32)),
        nsew=torch.from_numpy(nsew.astype(np.float32)),
    )


def load_cfd(data_path: Path, start: int = 0, end: int = 151, temporal_step: int = 10, spatial_step: int = 2):
    with h5py.File(data_path, "r") as data:
        return {
            "x": np.asarray(data["X"])[::spatial_step],
            "y": np.asarray(data["Y"])[::spatial_step],
            "time": np.asarray(data["time"])[start:end:temporal_step],
            "levelset": np.asarray(data["levelset"])[start:end:temporal_step, ::spatial_step, ::spatial_step],
            "pressure": np.asarray(data["pressure"])[start:end:temporal_step, ::spatial_step, ::spatial_step],
            "velocity_x": np.asarray(data["velocityX"])[start:end:temporal_step, ::spatial_step, ::spatial_step],
            "velocity_y": np.asarray(data["velocityY"])[start:end:temporal_step, ::spatial_step, ::spatial_step],
        }
