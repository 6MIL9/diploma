from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import h5py
import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from pinn.data import radius_from_h5_path
from pinn.config import PhysicsConfig
from pinn.model import TwoPhasePINN


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    requested = torch.device(device)
    if requested.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    return requested


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


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def selected_grid(
    data_path: Path,
    k: int,
    start: int,
    temporal_step: int,
    spatial_step: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(data_path, "r") as data:
        x = np.asarray(data["X"])[::spatial_step]
        y = np.asarray(data["Y"])[::spatial_step]
        times_all = np.asarray(data["time"])

    indices = start + temporal_step * np.arange(k)
    if len(indices) == 0:
        raise ValueError("K must be positive.")
    if indices[-1] >= len(times_all):
        raise ValueError(
            f"Requested time index {indices[-1]}, but {data_path} contains only {len(times_all)} time snapshots."
        )
    return x, y, times_all[indices]


def mesh_inputs(x: np.ndarray, y: np.ndarray, t: np.ndarray, radius: float | None) -> torch.Tensor:
    tt, yy, xx = np.meshgrid(t, y, x, indexing="ij")
    columns = [xx.ravel(), yy.ravel(), tt.ravel()]
    if radius is not None:
        columns.append(np.full(xx.size, radius, dtype=np.float64))
    return torch.from_numpy(np.stack(columns, axis=1).astype(np.float32))


def benchmark_pinn(
    checkpoint: Path,
    data_path: Path,
    k: int,
    start: int,
    temporal_step: int,
    spatial_step: int,
    batch_size: int,
    warmup: int,
    repeats: int,
    device_name: str,
) -> dict:
    device = resolve_device(device_name)
    model, cfg = load_model(checkpoint, device)
    dtype = torch.float64 if cfg.get("dtype") == "float64" else torch.float32
    physics = cfg["physics"]

    x_raw, y_raw, t_raw = selected_grid(data_path, k, start, temporal_step, spatial_step)
    x = x_raw / physics["l_ref"]
    y = y_raw / physics["l_ref"]
    t = t_raw / physics["l_ref"]
    input_dim = int(model.net.trunk[0].weight.shape[1])
    radius = radius_from_h5_path(data_path) / physics["l_ref"] if input_dim == 4 else None

    prepare_start = time.perf_counter()
    xyt = mesh_inputs(x, y, t, radius).to(device=device, dtype=dtype)
    synchronize(device)
    prepare_seconds = time.perf_counter() - prepare_start

    with torch.inference_mode():
        for _ in range(warmup):
            model.predict(xyt, batch_size=batch_size)
        synchronize(device)

        measurements = []
        for _ in range(repeats):
            started = time.perf_counter()
            model.predict(xyt, batch_size=batch_size)
            synchronize(device)
            measurements.append(time.perf_counter() - started)

    seconds = float(np.median(measurements))
    return {
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
        "k": k,
        "grid_x": int(len(x)),
        "grid_y": int(len(y)),
        "points_total": int(xyt.shape[0]),
        "batch_size": int(batch_size),
        "warmup": int(warmup),
        "repeats": int(repeats),
        "prepare_seconds": float(prepare_seconds),
        "forward_seconds_median": seconds,
        "forward_seconds_all": [float(value) for value in measurements],
        "seconds_per_snapshot": seconds / k,
        "points_per_second": float(xyt.shape[0] / seconds) if seconds > 0 else float("inf"),
        "time_start": float(t_raw[0]),
        "time_end": float(t_raw[-1]),
        "snapshot_dt": float(t_raw[1] - t_raw[0]) if len(t_raw) > 1 else None,
    }


def run_basilisk(
    source: Path,
    run_dir: Path,
    qcc: str,
    level: int,
    radius: float,
    snapshot_dt: float,
    t_end: float,
    executable_name: str = "rising_bubble_benchmark",
) -> dict:
    run_dir.mkdir(parents=True, exist_ok=True)
    for stale in [*run_dir.glob("snapshot-*.tsv"), run_dir / "interface-final.dat", run_dir / "log.txt"]:
        if stale.exists():
            stale.unlink()

    local_source = run_dir / source.name
    if source.resolve() != local_source.resolve():
        shutil.copy2(source, local_source)

    exe = run_dir / executable_name
    compile_cmd = [
        qcc,
        "-O2",
        "-Wall",
        f"-DLEVEL={level}",
        f"-DRADIUS={radius:.17g}",
        f"-DSNAPSHOT_DT={snapshot_dt:.17g}",
        f"-DT_END={t_end:.17g}",
        local_source.name,
        "-o",
        exe.name,
        "-lm",
    ]

    compile_started = time.perf_counter()
    compile_result = subprocess.run(compile_cmd, cwd=run_dir, text=True, capture_output=True, check=False)
    compile_seconds = time.perf_counter() - compile_started
    if compile_result.returncode != 0:
        raise RuntimeError(
            "Basilisk compilation failed.\n"
            f"Command: {' '.join(compile_cmd)}\n"
            f"stdout:\n{compile_result.stdout}\n"
            f"stderr:\n{compile_result.stderr}"
        )

    log_path = run_dir / "log.txt"
    run_started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        run_result = subprocess.run([f"./{exe.name}"], cwd=run_dir, text=True, stdout=log, stderr=subprocess.PIPE, check=False)
    run_seconds = time.perf_counter() - run_started
    if run_result.returncode != 0:
        raise RuntimeError(f"Basilisk run failed with code {run_result.returncode}.\nstderr:\n{run_result.stderr}")

    snapshots = sorted(run_dir.glob("snapshot-*.tsv"))
    return {
        "level": int(level),
        "radius": float(radius),
        "snapshot_dt": float(snapshot_dt),
        "t_end": float(t_end),
        "run_dir": str(run_dir),
        "log_path": str(log_path),
        "compile_seconds": float(compile_seconds),
        "run_seconds": float(run_seconds),
        "total_seconds_with_compile": float(compile_seconds + run_seconds),
        "snapshots_written": int(len(snapshots)),
    }


def print_summary(pinn: dict, basilisk: dict | None) -> None:
    print("PINN inference")
    print(f"  K snapshots: {pinn['k']}")
    print(f"  grid: {pinn['grid_x']} x {pinn['grid_y']} ({pinn['points_total']} points)")
    print(f"  device/dtype: {pinn['device']} / {pinn['dtype']}")
    print(f"  input preparation: {pinn['prepare_seconds']:.6f} s")
    print(f"  forward median: {pinn['forward_seconds_median']:.6f} s")
    print(f"  per snapshot: {pinn['seconds_per_snapshot']:.6f} s")
    print(f"  throughput: {pinn['points_per_second']:.3e} points/s")

    if basilisk is None:
        print("Basilisk run: skipped")
        return

    speedup = basilisk["run_seconds"] / pinn["forward_seconds_median"]
    print("Basilisk")
    print(f"  level/radius: {basilisk['level']} / {basilisk['radius']}")
    print(f"  t_end, snapshot_dt: {basilisk['t_end']} / {basilisk['snapshot_dt']}")
    print(f"  snapshots written: {basilisk['snapshots_written']}")
    print(f"  compile time: {basilisk['compile_seconds']:.6f} s")
    print(f"  run time: {basilisk['run_seconds']:.6f} s")
    print(f"  PINN forward speedup vs Basilisk run: {speedup:.3f}x")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark trained PINN prediction time against Basilisk wall time.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to best.pt/last.pt.")
    parser.add_argument("--data-path", type=Path, required=True, help="HDF5 file used only for the target grid and times.")
    parser.add_argument("-K", "--snapshots", type=int, default=10, help="Number of time snapshots to predict.")
    parser.add_argument("--start", type=int, default=0, help="First HDF5 time index.")
    parser.add_argument("--temporal-step", type=int, default=1, help="Stride between HDF5 time indices.")
    parser.add_argument("--spatial-step", type=int, default=1, help="Spatial stride for the prediction grid.")
    parser.add_argument("--batch-size", type=int, default=262_144)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--json-output", type=Path, help="Optional path for benchmark results JSON.")

    parser.add_argument("--skip-basilisk", action="store_true", help="Measure only PINN inference.")
    parser.add_argument("--qcc", default=shutil.which("qcc") or "/Users/erklyavlin/repos/basilisk/src/qcc")
    parser.add_argument("--basilisk-source", type=Path, default=ROOT / "cfd_basilisk" / "rising_bubble.c")
    parser.add_argument("--basilisk-run-dir", type=Path, default=ROOT / "cfd_basilisk" / "runs" / "benchmark")
    parser.add_argument("--basilisk-level", type=int, default=8)
    parser.add_argument("--basilisk-radius", type=float, help="Defaults to the radius stored in --data-path.")
    args = parser.parse_args()

    pinn = benchmark_pinn(
        checkpoint=args.checkpoint,
        data_path=args.data_path,
        k=args.snapshots,
        start=args.start,
        temporal_step=args.temporal_step,
        spatial_step=args.spatial_step,
        batch_size=args.batch_size,
        warmup=args.warmup,
        repeats=args.repeats,
        device_name=args.device,
    )

    basilisk = None
    if not args.skip_basilisk:
        if pinn["snapshot_dt"] is None:
            raise ValueError("Basilisk comparison needs K >= 2 to infer SNAPSHOT_DT.")
        radius = args.basilisk_radius if args.basilisk_radius is not None else radius_from_h5_path(args.data_path)
        basilisk = run_basilisk(
            source=args.basilisk_source,
            run_dir=args.basilisk_run_dir,
            qcc=args.qcc,
            level=args.basilisk_level,
            radius=radius,
            snapshot_dt=pinn["snapshot_dt"],
            t_end=pinn["time_end"],
        )

    result = {"pinn": pinn, "basilisk": basilisk}
    print_summary(pinn, basilisk)
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"wrote {args.json_output}")


if __name__ == "__main__":
    main()
