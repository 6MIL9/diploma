from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


def load(path: Path) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as f:
        return {name: np.asarray(f[name]) for name in f.keys()}


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def rel_rmse(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.sqrt(np.mean(b**2)))
    return rmse(a, b) / denom if denom > 0.0 else float("nan")


def bubble_alpha(data: dict[str, np.ndarray]) -> np.ndarray:
    return (data["levelset"] < 0.0).astype(np.float64)


def centers(alpha: np.ndarray, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xx, yy = np.meshgrid(x, y, indexing="ij")
    area = alpha.sum(axis=(1, 2)) * (x[1] - x[0]) * (y[1] - y[0])
    mass = np.maximum(alpha.sum(axis=(1, 2)), np.finfo(float).eps)
    cx = (alpha * xx[None, :, :]).sum(axis=(1, 2)) / mass
    cy = (alpha * yy[None, :, :]).sum(axis=(1, 2)) / mass
    return area, cx, cy


def align_pressure_by_frame(pred: np.ndarray, ref: np.ndarray) -> np.ndarray:
    offset = ref.mean(axis=(1, 2), keepdims=True) - pred.mean(axis=(1, 2), keepdims=True)
    return pred + offset


def print_metric(name: str, candidate: np.ndarray, reference: np.ndarray) -> None:
    print(
        f"{name:10s} rmse={rmse(candidate, reference):.6g} "
        f"mae={mae(candidate, reference):.6g} rel_rmse={rel_rmse(candidate, reference):.6g}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two rising-bubble HDF5 simulations.")
    parser.add_argument("--reference", type=Path, default=Path("../cfd_data/rising_bubble.h5"))
    parser.add_argument("--candidate", type=Path, required=True)
    args = parser.parse_args()

    ref = load(args.reference)
    cand = load(args.candidate)

    print("reference:", args.reference)
    print("candidate:", args.candidate)
    print("shape reference:", ref["levelset"].shape)
    print("shape candidate :", cand["levelset"].shape)
    print("max |dX|:", float(np.max(np.abs(ref["X"] - cand["X"]))) if ref["X"].shape == cand["X"].shape else "shape mismatch")
    print("max |dY|:", float(np.max(np.abs(ref["Y"] - cand["Y"]))) if ref["Y"].shape == cand["Y"].shape else "shape mismatch")
    print(
        "max |dt|:",
        float(np.max(np.abs(ref["time"] - cand["time"]))) if ref["time"].shape == cand["time"].shape else "shape mismatch",
    )

    if ref["levelset"].shape != cand["levelset"].shape:
        raise SystemExit("Cannot compare fields directly: levelset shapes differ.")

    ref_alpha = bubble_alpha(ref)
    cand_alpha = bubble_alpha(cand)
    intersection = np.logical_and(ref_alpha > 0.5, cand_alpha > 0.5).sum()
    union = np.logical_or(ref_alpha > 0.5, cand_alpha > 0.5).sum()
    print(f"alpha_iou  {intersection / union if union else float('nan'):.6g}")

    print_metric("alpha", cand_alpha, ref_alpha)
    print_metric("levelset", cand["levelset"], ref["levelset"])
    print_metric("density", cand["density"], ref["density"])
    print_metric("velocityX", cand["velocityX"], ref["velocityX"])
    print_metric("velocityY", cand["velocityY"], ref["velocityY"])
    print_metric("pressure", cand["pressure"], ref["pressure"])
    print_metric("p_aligned", align_pressure_by_frame(cand["pressure"], ref["pressure"]), ref["pressure"])

    ref_area, ref_cx, ref_cy = centers(ref_alpha, ref["X"], ref["Y"])
    cand_area, cand_cx, cand_cy = centers(cand_alpha, cand["X"], cand["Y"])
    print("\nbubble trajectory")
    print(f"area rmse={rmse(cand_area, ref_area):.6g}, final ref={ref_area[-1]:.6g}, candidate={cand_area[-1]:.6g}")
    print(f"cx   rmse={rmse(cand_cx, ref_cx):.6g}, final ref={ref_cx[-1]:.6g}, candidate={cand_cx[-1]:.6g}")
    print(f"cy   rmse={rmse(cand_cy, ref_cy):.6g}, final ref={ref_cy[-1]:.6g}, candidate={cand_cy[-1]:.6g}")


if __name__ == "__main__":
    main()
