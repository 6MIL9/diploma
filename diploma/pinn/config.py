from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class PointConfig:
    alpha: tuple[int, int] = (120, 100)
    pde: tuple[int, int, int] = (100, 300, 400)
    north: tuple[int, int] = (12, 12)
    south: tuple[int, int] = (12, 12)
    east: tuple[int, int] = (12, 12)
    west: tuple[int, int] = (12, 12)


@dataclass
class PhysicsConfig:
    mu: tuple[float, float] = (1.0, 10.0)
    rho: tuple[float, float] = (100.0, 1000.0)
    sigma: float = 24.5
    g: float = -0.98
    u_ref: float = 1.0
    l_ref: float = 0.25
    p_ref_rho: float = 1000.0


@dataclass
class TrainingConfig:
    data_path: Path = Path("cfd_data/rising_bubble.h5")
    output_dir: Path = Path("diploma/checkpoints")
    hidden_layers: tuple[int, ...] = (128, 128, 128, 128)
    activation: str = "tanh"
    points: PointConfig = field(default_factory=PointConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    loss_weights_pde: tuple[float, float, float, float] = (1.0, 10.0, 10.0, 1.0)
    epochs: tuple[int, ...] = (1000, 1000, 1000)
    learning_rates: tuple[float, ...] = (1e-4, 5e-5, 1e-5)
    batch_size: int | None = 4096
    num_batches: int | None = None
    checkpoint_interval: int = 100
    seed: int = 1234
    device: str = "auto"
    dtype: str = "float32"
    num_workers: int = 0

    def to_dict(self) -> dict:
        data = asdict(self)
        data["data_path"] = str(self.data_path)
        data["output_dir"] = str(self.output_dir)
        return data


def preset_config(name: str) -> TrainingConfig:
    if name == "smoke":
        return TrainingConfig(
            hidden_layers=(32, 32),
            points=PointConfig(
                alpha=(20, 20),
                pde=(20, 40, 40),
                north=(6, 4),
                south=(6, 4),
                east=(6, 4),
                west=(6, 4),
            ),
            epochs=(2,),
            learning_rates=(1e-3,),
            batch_size=512,
            checkpoint_interval=1,
        )
    if name == "default":
        return TrainingConfig()
    if name == "paper":
        return TrainingConfig(
            hidden_layers=(350,) * 8,
            points=PointConfig(
                alpha=(500, 400),
                pde=(400, 2000, 3000),
                north=(20, 20),
                south=(20, 20),
                east=(20, 20),
                west=(20, 20),
            ),
            epochs=(5000,) * 5,
            learning_rates=(1e-4, 5e-5, 1e-5, 5e-6, 1e-6),
            batch_size=None,
            num_batches=20,
            checkpoint_interval=100,
        )
    if name == "paper_light":
        return TrainingConfig(
            hidden_layers=(32,64,128,256,128,64,32),
            points=PointConfig(
                alpha=(500, 400),
                pde=(400, 2000, 3000),
                north=(20, 20),
                south=(20, 20),
                east=(20, 20),
                west=(20, 20),
            ),
            epochs=(5000,) * 5,
            learning_rates=(1e-4, 5e-5, 1e-5, 5e-6, 1e-6),
            batch_size=None,
            num_batches=20,
            checkpoint_interval=100,
        )
    raise ValueError(f"Unknown preset '{name}'. Use smoke, default, paper_light, or paper.")
