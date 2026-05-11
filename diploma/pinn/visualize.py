from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .config import PhysicsConfig
from .data import load_cfd, radius_from_h5_path
from .model import TwoPhasePINN


def resolve_device(device: str = "auto") -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    requested = torch.device(device)
    if requested.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    return requested


def _mesh_inputs(x: np.ndarray, y: np.ndarray, t: np.ndarray, radius: float | None = None) -> torch.Tensor:
    tt, yy, xx = np.meshgrid(t, y, x, indexing="ij")
    columns = [xx.ravel(), yy.ravel(), tt.ravel()]
    if radius is not None:
        columns.append(np.full(xx.size, radius, dtype=np.float64))
    return torch.from_numpy(np.stack(columns, axis=1).astype(np.float32))


def _reshape(values: torch.Tensor, t: np.ndarray, y: np.ndarray, x: np.ndarray) -> np.ndarray:
    return values.detach().cpu().numpy().reshape(len(t), len(y), len(x))


def _to_yx(values: np.ndarray) -> np.ndarray:
    return np.transpose(values, (0, 2, 1))


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[TwoPhasePINN, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    cfg = checkpoint["config"]
    physics = PhysicsConfig(**cfg["physics"])
    input_dim = int(checkpoint["model_state"]["net.trunk.0.weight"].shape[1])
    model = TwoPhasePINN(
        tuple(cfg["hidden_layers"]),
        physics,
        cfg.get("activation", "tanh"),
        tuple(cfg["loss_weights_pde"]),
        input_dim=input_dim,
    )
    dtype = torch.float64 if cfg.get("dtype") == "float64" else torch.float32
    model.to(device=device, dtype=dtype)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, cfg


def predict_cfd_grid(
    checkpoint_path: Path,
    data_path: Path,
    device: str = "auto",
    start: int = 0,
    end: int = 151,
    temporal_step: int = 10,
    spatial_step: int = 4,
    batch_size: int = 262_144,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict]:
    device_obj = resolve_device(device)
    model, cfg = load_model(Path(checkpoint_path), device_obj)
    physics = PhysicsConfig(**cfg["physics"])
    dtype = torch.float64 if cfg.get("dtype") == "float64" else torch.float32

    cfd = load_cfd(Path(data_path), start=start, end=end, temporal_step=temporal_step, spatial_step=spatial_step)
    x = cfd["x"] / physics.l_ref
    y = cfd["y"] / physics.l_ref
    t = cfd["time"] / physics.l_ref
    radius = radius_from_h5_path(Path(data_path)) / physics.l_ref
    input_dim = int(model.net.trunk[0].weight.shape[1])
    xyt = _mesh_inputs(x, y, t, radius if input_dim == 4 else None).to(device=device_obj, dtype=dtype)

    with torch.no_grad():
        u, v, p, alpha = model.predict(xyt, batch_size=batch_size)

    pred = {
        "u": _reshape(u, t, y, x),
        "v": _reshape(v, t, y, x),
        "p": _reshape(p, t, y, x),
        "alpha": _reshape(alpha, t, y, x),
    }
    true = {
        "u": _to_yx(cfd["velocity_x"]),
        "v": _to_yx(cfd["velocity_y"]),
        "p": _to_yx(cfd["pressure"]) / (physics.p_ref_rho * physics.u_ref**2),
        "alpha": (_to_yx(cfd["levelset"]) < 0.0).astype(np.float32),
    }
    coords = {
        "x": x,
        "y": y,
        "t": t,
        "time_raw": cfd["time"],
        "radius": radius,
        "physics": physics,
        "config": cfg,
    }
    return pred, true, coords


def error_metrics(pred: dict[str, np.ndarray], true: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for name in ("u", "v", "p", "alpha"):
        diff = pred[name] - true[name]
        rmse = float(np.sqrt(np.mean(diff**2)))
        mae = float(np.mean(np.abs(diff)))
        denom = float(np.sqrt(np.mean(true[name] ** 2)))
        rows.append(
            {
                "field": name,
                "rmse": rmse,
                "mae": mae,
                "relative_rmse": rmse / denom if denom > 0 else np.nan,
                "pred_min": float(np.min(pred[name])),
                "pred_max": float(np.max(pred[name])),
                "true_min": float(np.min(true[name])),
                "true_max": float(np.max(true[name])),
            }
        )
    return pd.DataFrame(rows)


def align_pressure_gauge(pred: dict[str, np.ndarray], true: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    aligned = dict(pred)
    aligned["p"] = pred["p"] + (true["p"][:, -1:, :].mean() - pred["p"][:, -1:, :].mean())
    return aligned


def plot_field_comparison(
    pred: dict[str, np.ndarray],
    true: dict[str, np.ndarray],
    coords: dict,
    time_index: int = -1,
    fields: tuple[str, ...] = ("alpha", "u", "v", "p"),
    cmap: str = "viridis",
    error_cmap: str = "coolwarm",
    align_pressure: bool = True,
    save_path: Path | None = None,
):
    x = coords["x"]
    y = coords["y"]
    t_value = coords["time_raw"][time_index]
    fig, axes = plt.subplots(len(fields), 3, figsize=(13, 3.2 * len(fields)), constrained_layout=True)
    if len(fields) == 1:
        axes = axes.reshape(1, 3)

    for row, field in enumerate(fields):
        pred_field = pred[field][time_index]
        true_field = true[field][time_index]
        if field == "p" and align_pressure:
            pred_field = pred_field + (true_field[-1, :].mean() - pred_field[-1, :].mean())
        err_field = pred_field - true_field
        vmin = min(float(pred_field.min()), float(true_field.min()))
        vmax = max(float(pred_field.max()), float(true_field.max()))
        err_abs = float(np.max(np.abs(err_field)))

        images = [
            axes[row, 0].contourf(x, y, true_field, levels=60, cmap=cmap, vmin=vmin, vmax=vmax),
            axes[row, 1].contourf(x, y, pred_field, levels=60, cmap=cmap, vmin=vmin, vmax=vmax),
            axes[row, 2].contourf(x, y, err_field, levels=60, cmap=error_cmap, vmin=-err_abs, vmax=err_abs),
        ]
        axes[row, 0].set_ylabel(field)
        axes[row, 0].set_title(f"CFD, t={t_value:.4g}")
        axes[row, 1].set_title("PINN" if field != "p" or not align_pressure else "PINN, aligned")
        axes[row, 2].set_title("PINN - CFD")

        for ax, image in zip(axes[row], images):
            ax.set_aspect("equal")
            ax.set_xlabel("x / L")
            fig.colorbar(image, ax=ax)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=180)
    return fig


def plot_centerline(
    pred: dict[str, np.ndarray],
    true: dict[str, np.ndarray],
    coords: dict,
    field: str = "alpha",
    time_index: int = -1,
    x_value: float = 0.0,
):
    x = coords["x"]
    y = coords["y"]
    x_idx = int(np.argmin(np.abs(x - x_value)))
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    ax.plot(true[field][time_index, :, x_idx], y, label="CFD")
    ax.plot(pred[field][time_index, :, x_idx], y, label="PINN")
    ax.set_xlabel(field)
    ax.set_ylabel("y / L")
    ax.set_title(f"{field} centerline at x={x[x_idx]:.3g}")
    ax.grid(True)
    ax.legend()
    return fig
