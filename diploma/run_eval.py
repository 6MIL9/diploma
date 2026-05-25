"""
Evaluate PINN checkpoints on CFD data and print L2 error tables.

Usage examples
--------------
# One checkpoint, multiple data files:
python run_eval.py \
    --checkpoint checkpoints/simple_pinn/best.pt \
    --data exp/train/rising_bubble_R025.h5 \
    --data exp/test/rising_bubble_R020.h5 \
    --data exp/test/rising_bubble_R030.h5

# Multiple checkpoints vs multiple data files (all combinations):
python run_eval.py \
    --checkpoint checkpoints_param/hard_2stage/best.pt \
    --checkpoint checkpoints_param/medium_2stage/best.pt \
    --checkpoint checkpoints_param/light_2stage/best.pt \
    --data exp/test/rising_bubble_R020.h5 \
    --data exp/test/rising_bubble_R030.h5

# Save comparison figures:
python run_eval.py \
    --checkpoint checkpoints_param/hard_2stage/best.pt \
    --data exp/test/rising_bubble_R030.h5 \
    --figures figures/

# Predefined experiment sets (--report):
python run_eval.py --report arch          # 3 arch comparison on R=0.30
python run_eval.py --report ablation      # sampling ablation on R=0.20, R=0.30
python run_eval.py --report simple        # simple PINN: train radius vs test radius
python run_eval.py --report all           # all of the above
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np

from pinn.visualize import predict_cfd_grid, error_metrics, plot_field_comparison, align_pressure_gauge
from pinn.data import radius_from_h5_path


# ---------------------------------------------------------------------------
# Report definitions
# ---------------------------------------------------------------------------

REPORTS: dict[str, dict] = {
    "arch": {
        "title": "Architecture comparison (parametric, R_test=0.30)",
        "checkpoints": [
            "checkpoints_param/hard_2stage/best.pt",
            "checkpoints_param/medium_2stage/best.pt",
            "checkpoints_param/light_2stage/best.pt",
        ],
        "data": [
            "exp/test/rising_bubble_R030.h5",
        ],
    },
    "ablation": {
        "title": "Sampling ablation (hard arch, R_test=0.20 and R_test=0.30)",
        "checkpoints": [
            "checkpoints_param/hard_2stage/best.pt",
            "checkpoints_param/hard_2stage_alpha_plus/best.pt",
            "checkpoints_param/hard_2stage_pde_plus/best.pt",
            "checkpoints_param/hard_2stage_alpha_pde_plus/best.pt",
        ],
        "data": [
            "exp/test/rising_bubble_R020.h5",
            "exp/test/rising_bubble_R030.h5",
        ],
    },
    "stages": {
        "title": "Training stages comparison (hard arch, R_test=0.20 and R_test=0.30)",
        "checkpoints": [
            "checkpoints_param/hard_2stage/best.pt",
            "checkpoints_param/hard_4stage/best.pt",
        ],
        "data": [
            "exp/test/rising_bubble_R020.h5",
            "exp/test/rising_bubble_R030.h5",
        ],
    },
    "simple": {
        "title": "Simple PINN: training radius vs unseen radii",
        "checkpoints": [
            "checkpoints/simple_pinn/best.pt",
        ],
        "data": [
            "exp/train/rising_bubble_R025.h5",
            "exp/test/rising_bubble_R020.h5",
            "exp/test/rising_bubble_R030.h5",
        ],
    },
    "all": {
        "title": "All experiments",
        "checkpoints": [
            "checkpoints/simple_pinn/best.pt",
            "checkpoints_param/hard_2stage/best.pt",
            "checkpoints_param/medium_2stage/best.pt",
            "checkpoints_param/light_2stage/best.pt",
            "checkpoints_param/hard_4stage/best.pt",
            "checkpoints_param/hard_2stage_alpha_plus/best.pt",
            "checkpoints_param/hard_2stage_pde_plus/best.pt",
            "checkpoints_param/hard_2stage_alpha_pde_plus/best.pt",
        ],
        "data": [
            "exp/train/rising_bubble_R025.h5",
            "exp/test/rising_bubble_R020.h5",
            "exp/test/rising_bubble_R030.h5",
        ],
    },
}


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def evaluate_one(
    checkpoint: Path,
    data: Path,
    device: str = "auto",
    temporal_step: int = 10,
    spatial_step: int = 4,
    figures_dir: Path | None = None,
    time_index: int = -1,
) -> dict:
    """Return dict with fields u/v/p/alpha relative L2 errors."""
    pred, true, coords = predict_cfd_grid(
        checkpoint,
        data,
        device=device,
        temporal_step=temporal_step,
        spatial_step=spatial_step,
    )
    pred_aligned = align_pressure_gauge(pred, true)
    df = error_metrics(pred_aligned, true)
    # coords["radius"] is dimensionless (R/l_ref); store physical radius for display
    phys_radius = radius_from_h5_path(data)
    result = {
        "checkpoint": checkpoint,
        "data": data,
        "radius": phys_radius,
        "metrics": df,
    }

    if figures_dir is not None:
        figures_dir = Path(figures_dir)
        figures_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{checkpoint.parent.name}_R{coords['radius']:.2f}_t{coords['time_raw'][time_index]:.3f}"
        fig_path = figures_dir / f"{stem}.png"
        plot_field_comparison(pred, true, coords, time_index=time_index, save_path=fig_path)
        print(f"  -> figure saved: {fig_path}")
        result["figure"] = fig_path

    return result


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

FIELDS = ["u", "v", "p", "alpha"]


def _fmt(val: float) -> str:
    return f"{val * 100:.3f}%"


def print_table(results: list[dict], metric: str = "relative_rmse") -> None:
    """Print a compact table: rows=checkpoints, cols=data files."""
    if not results:
        return

    # Group by data file then checkpoint
    data_files = list(dict.fromkeys(str(r["data"]) for r in results))
    checkpoints = list(dict.fromkeys(str(r["checkpoint"]) for r in results))

    index: dict[tuple[str, str], dict] = {}
    for r in results:
        index[(str(r["checkpoint"]), str(r["data"]))] = r

    col_w = 11

    for data_path in data_files:
        radius_label = f"R={results[0]['radius']:.2f}" if results else ""
        # find actual radius for this data file
        for r in results:
            if str(r["data"]) == data_path:
                radius_label = f"R={r['radius']:.3g}"
                break
        print(f"\n  Data: {Path(data_path).name}  ({radius_label})")
        header = f"  {'Checkpoint':<40}" + "".join(f"  {f:>{col_w}}" for f in FIELDS)
        print(header)
        print("  " + "-" * (40 + (col_w + 2) * len(FIELDS)))
        for ckpt_path in checkpoints:
            key = (ckpt_path, data_path)
            if key not in index:
                continue
            df = index[key]["metrics"]
            row_vals = {row["field"]: row[metric] for _, row in df.iterrows()}
            name = Path(ckpt_path).parent.name
            line = f"  {name:<40}" + "".join(f"  {_fmt(row_vals.get(f, float('nan'))):>{col_w}}" for f in FIELDS)
            print(line)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PINN checkpoints on CFD data.")
    parser.add_argument("--checkpoint", type=Path, action="append", dest="checkpoints", default=[],
                        help="Path to a checkpoint file (can repeat).")
    parser.add_argument("--data", type=Path, action="append", dest="data_files", default=[],
                        help="Path to an HDF5 CFD data file (can repeat).")
    parser.add_argument("--report", choices=list(REPORTS) + ["all"], default=None,
                        help="Run a predefined experiment set.")
    parser.add_argument("--figures", type=Path, default=None,
                        help="Directory to save comparison figures.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--temporal-step", type=int, default=10)
    parser.add_argument("--spatial-step", type=int, default=4)
    parser.add_argument("--time-index", type=int, default=-1,
                        help="Time snapshot index for figures (-1 = last).")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent

    if args.report:
        report_keys = list(REPORTS) if args.report == "all" else [args.report]
        for key in report_keys:
            report = REPORTS[key]
            print(f"\n{'='*70}")
            print(f"  REPORT: {report['title']}")
            print(f"{'='*70}")
            checkpoints = [base_dir / p for p in report["checkpoints"]]
            data_files = [base_dir / p for p in report["data"]]
            results = _run_all(checkpoints, data_files, args)
            print_table(results)
        return

    if not args.checkpoints:
        parser.error("Provide at least one --checkpoint or use --report.")
    if not args.data_files:
        parser.error("Provide at least one --data file or use --report.")

    results = _run_all(args.checkpoints, args.data_files, args)
    print_table(results)


def _run_all(checkpoints: list[Path], data_files: list[Path], args) -> list[dict]:
    results = []
    for ckpt in checkpoints:
        if not ckpt.exists():
            print(f"  [SKIP] checkpoint not found: {ckpt}")
            continue
        for data in data_files:
            if not data.exists():
                print(f"  [SKIP] data file not found: {data}")
                continue
            radius = radius_from_h5_path(data)
            print(f"\nEvaluating {ckpt.parent.name} on R={radius:.3g} ...")
            try:
                r = evaluate_one(
                    ckpt, data,
                    device=args.device,
                    temporal_step=args.temporal_step,
                    spatial_step=args.spatial_step,
                    figures_dir=args.figures,
                    time_index=args.time_index,
                )
                results.append(r)
                df = r["metrics"]
                for _, row in df.iterrows():
                    print(f"  {row['field']:>5}: rel_L2={row['relative_rmse']*100:.3f}%  mae={row['mae']:.4e}")
            except Exception as e:
                print(f"  [ERROR] {e}")
    return results


if __name__ == "__main__":
    main()
