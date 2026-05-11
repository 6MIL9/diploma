from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .config import PhysicsConfig
from .data import radius_from_h5_path
from .model import TwoPhasePINN
from .visualize import error_metrics, plot_field_comparison, predict_cfd_grid


def mesh_inputs(x: np.ndarray, y: np.ndarray, t: np.ndarray, radius: float | None = None) -> torch.Tensor:
    yy, tt, xx = np.meshgrid(y, t, x)
    columns = [xx.ravel(), yy.ravel(), tt.ravel()]
    if radius is not None:
        columns.append(np.full(xx.size, radius, dtype=np.float64))
    return torch.from_numpy(np.stack(columns, axis=1).astype(np.float32))


def reshape_prediction(values: torch.Tensor, x: np.ndarray, y: np.ndarray, t: np.ndarray) -> np.ndarray:
    return values.detach().cpu().numpy().reshape(len(t), len(y), len(x), order="C")


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained rising bubble PINN checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=151)
    parser.add_argument("--temporal-step", type=int, default=10)
    parser.add_argument("--spatial-step", type=int, default=4)
    parser.add_argument("--time-index", type=int, default=-1)
    parser.add_argument("--save-figure", type=Path)
    args = parser.parse_args()

    pred, true, coords = predict_cfd_grid(
        args.checkpoint,
        args.data_path,
        device=args.device,
        start=args.start,
        end=args.end,
        temporal_step=args.temporal_step,
        spatial_step=args.spatial_step,
    )
    idx = args.time_index
    print(f"Predicted arrays at resolution t={len(coords['t'])}, y={len(coords['y'])}, x={len(coords['x'])}")
    print(f"Radius: {radius_from_h5_path(args.data_path):.4g}")
    print(f"u range: {pred['u'].min():.4e} .. {pred['u'].max():.4e}")
    print(f"v range: {pred['v'].min():.4e} .. {pred['v'].max():.4e}")
    print(f"p range: {pred['p'].min():.4e} .. {pred['p'].max():.4e}")
    print(f"alpha range: {pred['alpha'].min():.4e} .. {pred['alpha'].max():.4e}")
    print(error_metrics(pred, true).to_string(index=False))

    fig = plot_field_comparison(pred, true, coords, time_index=idx, save_path=args.save_figure)
    if not args.save_figure:
        import matplotlib.pyplot as plt

        plt.show()


if __name__ == "__main__":
    main()
