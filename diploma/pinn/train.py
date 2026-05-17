from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from tqdm import trange

from .config import TrainingConfig, preset_config
from .data import TrainingData, make_training_data
from .model import TwoPhasePINN


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    requested = torch.device(device)
    if requested.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    return requested


def resolve_dtype(dtype: str) -> torch.dtype:
    if dtype == "float32":
        return torch.float32
    if dtype == "float64":
        return torch.float64
    raise ValueError("dtype must be float32 or float64")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_run_dir(output_dir: Path) -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def compute_total_batch_size(data: TrainingData, batch_size: int | None, num_batches: int | None) -> int:
    sizes = data.sizes
    total = sum(sizes.values())
    if batch_size is not None:
        return batch_size
    if num_batches is None:
        return total
    if num_batches <= 0:
        raise ValueError("num_batches must be a positive integer.")
    return int(np.ceil(total / num_batches))


def dataset_batch_sizes(data: TrainingData, total_batch_size: int) -> dict[str, int]:
    sizes = data.sizes
    total = sum(sizes.values())
    if total_batch_size < 0 or total_batch_size >= total:
        return sizes
    return {key: max(1, int(np.ceil(total_batch_size * value / total))) for key, value in sizes.items()}


def iter_batches(data: TrainingData, batch_sizes: dict[str, int]):
    sizes = data.sizes
    number_of_batches = max(int(np.ceil(sizes[key] / batch_sizes[key])) for key in sizes)
    permutations = {key: torch.randperm(sizes[key], device=getattr(data, key).device) for key in sizes}
    for batch_no in range(number_of_batches):
        batch = {}
        for key in sizes:
            tensor = getattr(data, key)
            start = batch_no * batch_sizes[key]
            end = min(start + batch_sizes[key], sizes[key])
            if start >= sizes[key]:
                idx = torch.randint(0, sizes[key], (batch_sizes[key],), device=tensor.device)
            else:
                idx = permutations[key][start:end]
            batch[key] = tensor[idx]
        yield batch


def save_checkpoint(
    path: Path,
    model: TwoPhasePINN,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    stage: int,
    loss: float,
    history: list[dict[str, float]],
    cfg: TrainingConfig,
) -> None:
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "stage": stage,
            "loss": loss,
            "history": history,
            "config": cfg.to_dict(),
        },
        path,
    )


class TrainingRun:
    def __init__(self, cfg: TrainingConfig, run_dir: Path | None = None):
        if len(cfg.epochs) != len(cfg.learning_rates):
            raise ValueError("epochs and learning_rates must have the same length.")

        self.cfg = cfg
        set_seed(cfg.seed)
        self.device = resolve_device(cfg.device)
        self.dtype = resolve_dtype(cfg.dtype)

        print(f"Loading and sampling training data from {cfg.data_path}")
        self.data = make_training_data(cfg.data_path, cfg.points, cfg.physics.l_ref, cfg.seed).to(self.device, self.dtype)
        print("Training point counts:", self.data.sizes)

        self.run_dir = make_run_dir(cfg.output_dir) if run_dir is None else Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        if run_dir is None:
            (self.run_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2), encoding="utf-8")

        self.model = TwoPhasePINN(cfg.hidden_layers, cfg.physics, cfg.activation, cfg.loss_weights_pde).to(
            device=self.device,
            dtype=self.dtype,
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.learning_rates[0])
        self.total_batch_size = compute_total_batch_size(self.data, cfg.batch_size, cfg.num_batches)
        self.batch_sizes = dataset_batch_sizes(self.data, self.total_batch_size)
        self.number_of_batches = max(
            int(np.ceil(self.data.sizes[key] / self.batch_sizes[key])) for key in self.data.sizes
        )
        print(
            f"Device: {self.device}, dtype: {self.dtype}, "
            f"total batch size: {self.total_batch_size}, batches/epoch: {self.number_of_batches}, "
            f"batch sizes: {self.batch_sizes}"
        )

        self.history: list[dict[str, float]] = []
        self.best_loss = float("inf")
        self.global_epoch = 0
        self.current_stage = 0

    def load_checkpoint(self, checkpoint_path: Path) -> None:
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.history = checkpoint.get("history", [])
        self.global_epoch = int(checkpoint.get("epoch", 0))
        self.current_stage = int(checkpoint.get("stage", 0))
        self.best_loss = min((row["total"] for row in self.history), default=float(checkpoint.get("loss", "inf")))
        print(
            f"Resumed from {checkpoint_path}: "
            f"epoch={self.global_epoch}, stage={self.current_stage}, best_loss={self.best_loss:.3e}"
        )

    def _write_history(self) -> None:
        (self.run_dir / "history.json").write_text(json.dumps(self.history, indent=2), encoding="utf-8")

    def _save_checkpoint(self, name: str, loss: float, stage: int) -> None:
        save_checkpoint(
            self.run_dir / name,
            self.model,
            self.optimizer,
            self.global_epoch,
            stage,
            loss,
            self.history,
            self.cfg,
        )

    def train_stage(
        self,
        epochs: int | None = None,
        lr: float | None = None,
        stage: int | None = None,
        progress: bool = True,
    ) -> list[dict[str, float]]:
        if stage is None:
            stage = self.current_stage + 1
        if epochs is None:
            epochs = self.cfg.epochs[stage - 1]
        if lr is None:
            lr = self.cfg.learning_rates[stage - 1]

        for group in self.optimizer.param_groups:
            group["lr"] = lr

        stage_history: list[dict[str, float]] = []
        epoch_iter = trange(epochs, desc=f"stage {stage} lr={lr:g}") if progress else range(epochs)
        for _ in epoch_iter:
            self.global_epoch += 1
            epoch_losses: dict[str, float] = {}
            batches = 0

            for batch in iter_batches(self.data, self.batch_sizes):
                self.optimizer.zero_grad(set_to_none=True)
                loss = self.model.loss(batch)
                loss.total.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=100.0)
                self.optimizer.step()

                detached = loss.detached()
                for key, value in detached.items():
                    epoch_losses[key] = epoch_losses.get(key, 0.0) + value
                batches += 1

            averaged = {key: value / batches for key, value in epoch_losses.items()}
            averaged["epoch"] = self.global_epoch
            averaged["stage"] = stage
            averaged["lr"] = lr
            self.history.append(averaged)
            stage_history.append(averaged)

            if progress:
                epoch_iter.set_postfix(
                    total=f"{averaged['total']:.3e}",
                    alpha=f"{averaged['alpha']:.2e}",
                    pde_u=f"{averaged['momentum_x']:.2e}",
                )

            if averaged["total"] < self.best_loss:
                self.best_loss = averaged["total"]
                self._save_checkpoint("best.pt", self.best_loss, stage)

            if self.cfg.checkpoint_interval and self.global_epoch % self.cfg.checkpoint_interval == 0:
                self._save_checkpoint(f"epoch_{self.global_epoch:06d}.pt", averaged["total"], stage)
                self._write_history()

        self.current_stage = max(self.current_stage, stage)
        self._save_checkpoint("last.pt", self.history[-1]["total"], stage)
        self._write_history()
        return stage_history

    def train_all(self) -> Path:
        start_stage = self.current_stage + 1
        for stage, (stage_epochs, lr) in enumerate(zip(self.cfg.epochs, self.cfg.learning_rates), start=start_stage):
            self.train_stage(stage_epochs, lr, stage=stage)
        print(f"Finished. Best checkpoint: {self.run_dir / 'best.pt'}")
        return self.run_dir


def train(cfg: TrainingConfig, resume: Path | None = None) -> Path:
    run = TrainingRun(cfg, run_dir=resume.parent if resume is not None else None)
    if resume is not None:
        run.load_checkpoint(resume)
    return run.train_all()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a PyTorch PINN for the rising bubble case.")
    parser.add_argument("--preset", choices=["smoke", "default", "paper_light", "paper"], default="default")
    parser.add_argument("--data-path", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default=None, help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--dtype", choices=["float32", "float64"], default=None)
    parser.add_argument("--epochs", type=int, default=None, help="Override with a single training stage.")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-batches", type=int, default=None, help="Compute total batch size as ceil(total_points / num_batches).")
    parser.add_argument("--hidden-width", type=int, default=None)
    parser.add_argument("--hidden-depth", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None, help="Override with a single learning rate stage.")
    parser.add_argument("--resume", type=Path, default=None, help="Resume from a checkpoint such as checkpoints/.../last.pt.")
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = preset_config(args.preset)
    if args.data_path is not None:
        cfg.data_path = args.data_path
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir
    if args.device is not None:
        cfg.device = args.device
    if args.dtype is not None:
        cfg.dtype = args.dtype
    if args.epochs is not None:
        cfg.epochs = (args.epochs,)
        cfg.learning_rates = (args.lr if args.lr is not None else cfg.learning_rates[0],)
    elif args.lr is not None:
        cfg.learning_rates = tuple(args.lr for _ in cfg.learning_rates)
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
        cfg.num_batches = None
    if args.num_batches is not None:
        cfg.num_batches = args.num_batches
        cfg.batch_size = None
    if args.hidden_width is not None or args.hidden_depth is not None:
        width = args.hidden_width if args.hidden_width is not None else cfg.hidden_layers[0]
        depth = args.hidden_depth if args.hidden_depth is not None else len(cfg.hidden_layers)
        cfg.hidden_layers = (width,) * depth
    if args.seed is not None:
        cfg.seed = args.seed
    train(cfg, resume=args.resume)


if __name__ == "__main__":
    main()
