from __future__ import annotations

import glob
import json
import math
import os
from dataclasses import dataclass, field
from itertools import cycle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


@dataclass
class PipelineConfig:
    data_csv: str = "datasets/data_points_phi.csv"
    phys_csv: str = "datasets/collocation.csv"
    eval_glob: str = "eval/eval_*.csv"

    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    dtype: torch.dtype = torch.float32

    lr_pretrain: float = 1e-3
    lr_full: float = 1e-4
    pretrain_epochs: int = 200
    flow_stage_epochs: int = 200
    joint_finetune_epochs: int = 100
    full_train_epochs: int = 1000

    batch_data: int = 8192
    batch_phys: int = 2048
    batch_uvp: int = 4096
    num_workers: int = 0

    lambda_data_pretrain: float = 20.0
    lambda_data_full: float = 10.0
    lambda_uvp_full: float = 5.0
    lambda_pressure_gauge: float = 1.0
    lambda_u_supervised: float = 1.0
    lambda_v_supervised: float = 1.0
    lambda_p_supervised: float = 5.0

    pretrain_ckpt_path: str = "checkpoints_uvp/model_pretrain.pt"
    full_ckpt_path: str = "checkpoints_uvp/model_full.pt"
    full_save_every: int = 100
    full_weights_save_every: int = 50
    full_state_dir: str = "checkpoints_uvp/full_state"
    full_weights_dir: str = "checkpoints_uvp/full_weights"
    full_latest_state_path: str = "checkpoints_uvp/full_state/latest.pt"
    full_latest_history_path: str = "checkpoints_uvp/full_state/latest_history.json"
    flow_save_every: int = 50
    flow_state_dir: str = "checkpoints_uvp/flow_state"
    flow_weights_dir: str = "checkpoints_uvp/flow_weights"
    flow_latest_state_path: str = "checkpoints_uvp/flow_state/latest.pt"
    flow_latest_history_path: str = "checkpoints_uvp/flow_state/latest_history.json"
    flow_supervised_save_every: int = 50
    flow_supervised_state_dir: str = "checkpoints_uvp/flow_supervised_state"
    flow_supervised_weights_dir: str = "checkpoints_uvp/flow_supervised_weights"
    flow_supervised_latest_state_path: str = "checkpoints_uvp/flow_supervised_state/latest.pt"
    flow_supervised_latest_history_path: str = "checkpoints_uvp/flow_supervised_state/latest_history.json"

    s_pretrain: float = 5.0
    s_full_start: float = 10.0
    s_full_end: float = 40.0
    s_full_ramp_epochs: int = 400

    x_min: float = 0.0
    x_max: float = 1.0
    y_min: float = 0.0
    y_max: float = 2.0
    t_min: float = 0.0
    t_max: float = 3.0

    g_x: float = 0.0
    g_y: float = 9.80

    rho_l: float = 1000.0
    rho_g: float = 100.0
    mu_l: float = 10.0
    mu_g: float = 1.0
    sigma: float = 24.5
    eps_norm: float = 1e-6

    hidden: int = 64
    depth: int = 4
    out_dim: int = 4
    activation: str = "tanh"
    model_style: str = "two_head"

    ema_beta: float = 0.99
    ema_eps: float = 1e-8
    ema_warmup_steps: int = 200

    seed_torch: int = 0
    seed_numpy: int = 0

    def __post_init__(self) -> None:
        self.x_center = 0.5 * (self.x_min + self.x_max)
        self.y_center = 0.5 * (self.y_min + self.y_max)
        self.t_center = 0.5 * (self.t_min + self.t_max)

        self.x_half_range = 0.5 * (self.x_max - self.x_min)
        self.y_half_range = 0.5 * (self.y_max - self.y_min)
        self.t_half_range = 0.5 * (self.t_max - self.t_min)

        self.p_gauge_x = 0.5 * (self.x_min + self.x_max)
        self.p_gauge_y = 0.5 * (self.y_min + self.y_max)
        self.p_gauge_t = self.t_min

        self.l_ref = self.x_max - self.x_min
        self.rho_ref = self.rho_l
        self.mu_ref = self.mu_l
        self.u_ref = math.sqrt(abs(self.g_y) * self.l_ref) if abs(self.g_y) > 0.0 else 1.0
        self.t_ref = self.l_ref / self.u_ref
        self.p_ref = self.rho_ref * self.u_ref**2

        self.re_ref = self.rho_ref * self.u_ref * self.l_ref / self.mu_ref
        self.we_ref = self.rho_ref * self.u_ref**2 * self.l_ref / self.sigma
        self.g_x_ref = self.g_x * self.l_ref / (self.u_ref**2)
        self.g_y_ref = self.g_y * self.l_ref / (self.u_ref**2)

    def describe_reference_scales(self) -> str:
        return (
            f"Reference scales: L={self.l_ref:.4f}, U={self.u_ref:.4f}, T={self.t_ref:.4f}, "
            f"P={self.p_ref:.4f}, Re={self.re_ref:.4f}, We={self.we_ref:.4f}, "
            f"Gx={self.g_x_ref:.4f}, Gy={self.g_y_ref:.4f}"
        )


class DataPoints(Dataset):
    def __init__(self, csv_path: str):
        df = pd.read_csv(csv_path)
        required = ["x", "y", "t", "phi"]
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing column '{col}' in {csv_path}. Columns: {df.columns}")
        self.xyt = np.array(df[["x", "y", "t"]].to_numpy(), dtype=np.float32, copy=True)
        self.phi = np.array(df["phi"].to_numpy()[:, None], dtype=np.float32, copy=True)

    def __len__(self) -> int:
        return self.xyt.shape[0]

    def __getitem__(self, idx: int):
        return self.xyt[idx].copy(), self.phi[idx].copy()


class PhysicsPoints(Dataset):
    def __init__(self, csv_path: str):
        df = pd.read_csv(csv_path)
        required = ["x", "y", "t"]
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing column '{col}' in {csv_path}. Columns: {df.columns}")
        self.xyt = np.array(df[["x", "y", "t"]].to_numpy(), dtype=np.float32, copy=True)

    def __len__(self) -> int:
        return self.xyt.shape[0]

    def __getitem__(self, idx: int):
        return self.xyt[idx].copy()


class UVPPoints(Dataset):
    def __init__(self, eval_glob: str, gauge_x: float, gauge_y: float):
        paths = sorted(glob.glob(eval_glob))
        if not paths:
            raise ValueError(f"No eval files found for pattern: {eval_glob}")

        frames: list[pd.DataFrame] = []
        required = ["Time", "Points:0", "Points:1", "U:0", "U:1", "p"]
        for csv_path in paths:
            df = pd.read_csv(csv_path)
            for col in required:
                if col not in df.columns:
                    raise ValueError(f"Missing column '{col}' in {csv_path}. Columns: {df.columns}")

            xy = np.array(df[["Points:0", "Points:1"]].to_numpy(), dtype=np.float32, copy=True)
            gauge_idx = np.argmin((xy[:, 0] - gauge_x) ** 2 + (xy[:, 1] - gauge_y) ** 2)
            y_np = np.array(df["Points:1"].to_numpy(), dtype=np.float32, copy=True)
            p_np = np.array(df["p"].to_numpy(), dtype=np.float32, copy=True)
            p_bg = fit_linear_pressure_background(y_np, p_np)
            p_residual = p_np - p_bg
            p_gauge = float(p_residual[gauge_idx])

            frames.append(
                pd.DataFrame(
                    {
                        "x": np.array(df["Points:0"].to_numpy(), dtype=np.float32, copy=True),
                        "y": y_np,
                        "t": np.array(df["Time"].to_numpy(), dtype=np.float32, copy=True),
                        "u": np.array(df["U:0"].to_numpy(), dtype=np.float32, copy=True),
                        "v": np.array(df["U:1"].to_numpy(), dtype=np.float32, copy=True),
                        "p": p_residual - p_gauge,
                    }
                )
            )

        uvp_df = pd.concat(frames, ignore_index=True)
        self.xyt = np.array(uvp_df[["x", "y", "t"]].to_numpy(), dtype=np.float32, copy=True)
        self.uvp = np.array(uvp_df[["u", "v", "p"]].to_numpy(), dtype=np.float32, copy=True)

    def __len__(self) -> int:
        return self.xyt.shape[0]

    def __getitem__(self, idx: int):
        return self.xyt[idx].copy(), self.uvp[idx].copy()


class AdaptiveSwish(nn.Module):
    def __init__(self, init_beta: float = 1.0):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(float(init_beta)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(self.beta * x)


def fit_linear_pressure_background(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """
    Fit a simple affine background p_bg(y) = a + b*y and return it at each y.
    This lets the model focus on the pressure residual instead of spending
    capacity on the dominant hydrostatic-like vertical trend.
    """
    y64 = np.asarray(y, dtype=np.float64).reshape(-1)
    p64 = np.asarray(p, dtype=np.float64).reshape(-1)
    a, b = np.polyfit(y64, p64, deg=1)
    return (a * y64 + b).astype(np.float32)


def make_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "tanh":
        return nn.Tanh()
    if name == "silu":
        return nn.SiLU()
    if name == "gelu":
        return nn.GELU()
    if name == "relu":
        return nn.ReLU()
    if name == "adaptive_swish":
        return AdaptiveSwish(1.0)
    raise ValueError(f"Unsupported activation: {name}")


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int = 3,
        out_dim: int = 4,
        hidden: int = 64,
        depth: int = 4,
        activation: str = "tanh",
    ):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden), make_activation(activation)]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(hidden, hidden), make_activation(activation)])
        layers.append(nn.Linear(hidden, out_dim))
        self.net = nn.Sequential(*layers)
        self._init()

    def _init(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, xyt: torch.Tensor) -> torch.Tensor:
        return self.net(xyt)


class TwoHeadMLP(nn.Module):
    def __init__(
        self,
        in_dim: int = 3,
        hidden: int = 64,
        depth: int = 4,
        activation: str = "tanh",
    ):
        super().__init__()
        if depth < 2:
            raise ValueError("TwoHeadMLP expects depth >= 2")

        trunk_layers: list[nn.Module] = [nn.Linear(in_dim, hidden), make_activation(activation)]
        for _ in range(depth - 2):
            trunk_layers.extend([nn.Linear(hidden, hidden), make_activation(activation)])
        self.trunk = nn.Sequential(*trunk_layers)

        self.alpha_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            make_activation(activation),
            nn.Linear(hidden, 1),
        )
        self.flow_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            make_activation(activation),
            nn.Linear(hidden, 3),
        )
        self._init()

    def _init(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, xyt: torch.Tensor) -> torch.Tensor:
        features = self.trunk(xyt)
        flow = self.flow_head(features)
        alpha = self.alpha_head(features)
        return torch.cat([flow, alpha], dim=1)


class PinnWorkflow:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        torch.manual_seed(cfg.seed_torch)
        np.random.seed(cfg.seed_numpy)
        os.makedirs(os.path.dirname(cfg.pretrain_ckpt_path) or ".", exist_ok=True)
        self.data_ds = DataPoints(cfg.data_csv)
        self.phys_ds = PhysicsPoints(cfg.phys_csv)
        self.uvp_ds = UVPPoints(cfg.eval_glob, cfg.p_gauge_x, cfg.p_gauge_y)
        self.data_loader = self._make_loader(self.data_ds, cfg.batch_data)
        self.phys_loader = self._make_loader(self.phys_ds, cfg.batch_phys)
        self.uvp_loader = self._make_loader(self.uvp_ds, cfg.batch_uvp)

    def _make_loader(self, dataset: Dataset, batch_size: int) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            persistent_workers=False,
        )

    def build_model(self) -> nn.Module:
        if self.cfg.model_style == "two_head":
            model = TwoHeadMLP(
                in_dim=3,
                hidden=self.cfg.hidden,
                depth=self.cfg.depth,
                activation=self.cfg.activation,
            )
        elif self.cfg.model_style == "single_head":
            model = MLP(
                in_dim=3,
                out_dim=self.cfg.out_dim,
                hidden=self.cfg.hidden,
                depth=self.cfg.depth,
                activation=self.cfg.activation,
            )
        else:
            raise ValueError(f"Unsupported model_style: {self.cfg.model_style}")
        return model.to(self.cfg.device).to(self.cfg.dtype)

    def build_optimizer(self, model: nn.Module, lr: float) -> torch.optim.Optimizer:
        params = [p for p in model.parameters() if p.requires_grad]
        return torch.optim.Adam(params, lr=lr)

    @staticmethod
    def freeze_alpha_head(model: nn.Module) -> None:
        if not hasattr(model, "alpha_head"):
            raise ValueError("freeze_alpha_head() requires a model with alpha_head")
        for param in model.alpha_head.parameters():
            param.requires_grad = False

    @staticmethod
    def unfreeze_all(model: nn.Module) -> None:
        for param in model.parameters():
            param.requires_grad = True

    def load_checkpoint_weights(self, model: nn.Module, ckpt_path: str) -> nn.Module:
        state = torch.load(ckpt_path, map_location=self.cfg.device)
        if isinstance(state, dict) and "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"])
        elif isinstance(state, dict):
            model.load_state_dict(state)
        else:
            raise ValueError(f"Unexpected checkpoint format: {ckpt_path}")
        return model

    def normalize_xyt(self, xyt: torch.Tensor) -> torch.Tensor:
        center = xyt.new_tensor([self.cfg.x_center, self.cfg.y_center, self.cfg.t_center])
        half_range = xyt.new_tensor([self.cfg.x_half_range, self.cfg.y_half_range, self.cfg.t_half_range])
        return (xyt - center) / half_range

    def mixture_rho(self, phi: torch.Tensor) -> torch.Tensor:
        return 0.5 * ((1.0 + phi) * self.cfg.rho_l + (1.0 - phi) * self.cfg.rho_g)

    def mixture_mu(self, phi: torch.Tensor) -> torch.Tensor:
        return 0.5 * ((1.0 + phi) * self.cfg.mu_l + (1.0 - phi) * self.cfg.mu_g)

    def mixture_rho_nd(self, phi: torch.Tensor) -> torch.Tensor:
        return self.mixture_rho(phi) / self.cfg.rho_ref

    def mixture_mu_nd(self, phi: torch.Tensor) -> torch.Tensor:
        return self.mixture_mu(phi) / self.cfg.mu_ref

    @staticmethod
    def grad(outputs: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
        return torch.autograd.grad(
            outputs,
            inputs,
            grad_outputs=torch.ones_like(outputs),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

    def model_fields(self, model: nn.Module, xyt: torch.Tensor):
        xyt_norm = self.normalize_xyt(xyt)
        out = model(xyt_norm)
        u_raw = out[:, 0:1]
        v_raw = out[:, 1:2]
        p = out[:, 2:3]
        alpha = out[:, 3:4]

        x = xyt[:, 0:1]
        y = xyt[:, 1:2]
        l_u = torch.sin(math.pi * x) * torch.sin(math.pi * y / 2.0)
        l_v = torch.sin(math.pi * y / 2.0)

        u = l_u * u_raw
        v = l_v * v_raw
        return u, v, p, alpha

    def uvp_supervision_loss(self, model: nn.Module, xyt: torch.Tensor, uvp_true: torch.Tensor):
        u_pred, v_pred, p_pred, _ = self.model_fields(model, xyt)
        u_true = uvp_true[:, 0:1]
        v_true = uvp_true[:, 1:2]
        p_true = uvp_true[:, 2:3]

        u_pred_nd = u_pred / self.cfg.u_ref
        v_pred_nd = v_pred / self.cfg.u_ref
        p_pred_nd = p_pred / self.cfg.p_ref

        u_true_nd = u_true / self.cfg.u_ref
        v_true_nd = v_true / self.cfg.u_ref
        p_true_nd = p_true / self.cfg.p_ref

        l_u = torch.mean((u_pred_nd - u_true_nd) ** 2)
        l_v = torch.mean((v_pred_nd - v_true_nd) ** 2)
        l_p = torch.mean((p_pred_nd - p_true_nd) ** 2)
        return l_u, l_v, l_p

    def weighted_uvp_supervision_loss(self, model: nn.Module, xyt: torch.Tensor, uvp_true: torch.Tensor):
        l_u, l_v, l_p = self.uvp_supervision_loss(model, xyt, uvp_true)
        loss_uvp = (
            self.cfg.lambda_u_supervised * l_u
            + self.cfg.lambda_v_supervised * l_v
            + self.cfg.lambda_p_supervised * l_p
        )
        return loss_uvp, l_u, l_v, l_p

    def pressure_gauge_loss(self, model: nn.Module) -> torch.Tensor:
        gauge_point = torch.tensor(
            [[self.cfg.p_gauge_x, self.cfg.p_gauge_y, self.cfg.p_gauge_t]],
            device=self.cfg.device,
            dtype=self.cfg.dtype,
        )
        _, _, p_anchor, _ = self.model_fields(model, gauge_point)
        return torch.mean((p_anchor / self.cfg.p_ref) ** 2)

    def get_phys_weights(self, epoch: int):
        if epoch <= 50:
            return 1.0, 0.5, 0.1, 0.0, 0.02
        if epoch <= 150:
            return 1.0, 0.7, 0.3, 0.0, 0.05
        if epoch <= 300:
            return 1.0, 1.0, 0.7, 0.0, 0.10
        if epoch <= 500:
            return 1.0, 1.0, 1.0, 0.02, 0.15
        return 1.0, 1.0, 1.2, 0.05, 0.20

    def get_flow_stage_weights(self, epoch: int):
        if epoch <= 50:
            return 1.0, 0.3, 1.0, 0.03
        if epoch <= 100:
            return 1.0, 0.3, 1.5, 0.05
        return 1.0, 0.2, 2.0, 0.08

    def get_joint_finetune_weights(self, epoch: int):
        if epoch <= 25:
            return 1.0, 0.2, 0.2, 0.0, 0.01
        if epoch <= 50:
            return 1.0, 0.3, 0.3, 0.0, 0.02
        if epoch <= 75:
            return 1.0, 0.4, 0.5, 0.0, 0.03
        return 1.0, 0.5, 0.7, 0.01, 0.05

    def get_s_full(self, epoch: int) -> float:
        if self.cfg.s_full_ramp_epochs <= 0:
            return self.cfg.s_full_end
        ramp = min(max((epoch - 1) / self.cfg.s_full_ramp_epochs, 0.0), 1.0)
        return self.cfg.s_full_start + (self.cfg.s_full_end - self.cfg.s_full_start) * ramp

    def residuals(self, model: nn.Module, xyt: torch.Tensor, s_val: float):
        xyt = xyt.requires_grad_(True)
        u, v, p, alpha = self.model_fields(model, xyt)

        phi = torch.tanh(float(s_val) * alpha)
        delta = 1.0 - phi**2
        rho_nd = self.mixture_rho_nd(phi)
        mu_nd = self.mixture_mu_nd(phi)

        u_grad = self.grad(u, xyt)
        v_grad = self.grad(v, xyt)
        p_grad = self.grad(p, xyt)
        a_grad = self.grad(alpha, xyt)

        u_x, u_y, u_t = u_grad[:, 0:1], u_grad[:, 1:2], u_grad[:, 2:3]
        v_x, v_y, v_t = v_grad[:, 0:1], v_grad[:, 1:2], v_grad[:, 2:3]
        p_x, p_y = p_grad[:, 0:1], p_grad[:, 1:2]
        a_x, a_y, a_t = a_grad[:, 0:1], a_grad[:, 1:2], a_grad[:, 2:3]

        r_div = u_x + v_y
        r_adv = a_t + u * a_x + v * a_y
        grad_a_norm = torch.sqrt(a_x**2 + a_y**2 + 1e-12)
        r_eik = grad_a_norm - 1.0

        n_x = a_x / (grad_a_norm + self.cfg.eps_norm)
        n_y = a_y / (grad_a_norm + self.cfg.eps_norm)
        n_x_grad = self.grad(n_x, xyt)
        n_y_grad = self.grad(n_y, xyt)

        dnxx_dx_nd = self.cfg.l_ref * n_x_grad[:, 0:1]
        dnyy_dy_nd = self.cfg.l_ref * n_y_grad[:, 1:2]
        kappa_nd = dnxx_dx_nd + dnyy_dy_nd

        f_st_x_nd = (1.0 / self.cfg.we_ref) * kappa_nd * n_x * delta
        f_st_y_nd = (1.0 / self.cfg.we_ref) * kappa_nd * n_y * delta

        u_nd = u / self.cfg.u_ref
        v_nd = v / self.cfg.u_ref

        u_x_nd = (self.cfg.l_ref / self.cfg.u_ref) * u_x
        u_y_nd = (self.cfg.l_ref / self.cfg.u_ref) * u_y
        u_t_nd = (self.cfg.t_ref / self.cfg.u_ref) * u_t
        v_x_nd = (self.cfg.l_ref / self.cfg.u_ref) * v_x
        v_y_nd = (self.cfg.l_ref / self.cfg.u_ref) * v_y
        v_t_nd = (self.cfg.t_ref / self.cfg.u_ref) * v_t
        p_x_nd = (self.cfg.l_ref / self.cfg.p_ref) * p_x
        p_y_nd = (self.cfg.l_ref / self.cfg.p_ref) * p_y

        dxx_nd = u_x_nd
        dyy_nd = v_y_nd
        dxy_nd = 0.5 * (u_y_nd + v_x_nd)

        term_xx_nd = 2.0 * mu_nd * dxx_nd
        term_xy_nd = 2.0 * mu_nd * dxy_nd
        term_yy_nd = 2.0 * mu_nd * dyy_nd

        term_xx_nd_grad = self.grad(term_xx_nd, xyt)
        term_xy_nd_grad = self.grad(term_xy_nd, xyt)
        term_yy_nd_grad = self.grad(term_yy_nd, xyt)

        visc_x_nd = self.cfg.l_ref * term_xx_nd_grad[:, 0:1] + self.cfg.l_ref * term_xy_nd_grad[:, 1:2]
        visc_y_nd = self.cfg.l_ref * term_xy_nd_grad[:, 0:1] + self.cfg.l_ref * term_yy_nd_grad[:, 1:2]

        adv_u_nd = u_t_nd + u_nd * u_x_nd + v_nd * u_y_nd
        adv_v_nd = v_t_nd + u_nd * v_x_nd + v_nd * v_y_nd

        r_mom_u = (
            rho_nd * adv_u_nd
            + p_x_nd
            - (1.0 / self.cfg.re_ref) * visc_x_nd
            - rho_nd * self.cfg.g_x_ref
            - f_st_x_nd
        )
        r_mom_v = (
            rho_nd * adv_v_nd
            + p_y_nd
            - (1.0 / self.cfg.re_ref) * visc_y_nd
            - rho_nd * self.cfg.g_y_ref
            - f_st_y_nd
        )

        return {
            "r_div": r_div,
            "r_adv": r_adv,
            "r_eik": r_eik,
            "r_mom_u": r_mom_u,
            "r_mom_v": r_mom_v,
            "phi": phi,
            "alpha": alpha,
            "u": u,
            "v": v,
            "p": p,
            "rho_nd": rho_nd,
            "mu_nd": mu_nd,
        }

    @staticmethod
    def make_pretrain_history() -> dict:
        return {"epoch": [], "loss": [], "loss_data": [], "s": []}

    @staticmethod
    def make_full_history() -> dict:
        return {
            "epoch": [],
            "loss": [],
            "loss_data": [],
            "loss_phys": [],
            "loss_uvp": [],
            "loss_gauge": [],
            "s": [],
            "L_u": [],
            "L_v": [],
            "L_p": [],
            "L_div": [],
            "L_adv": [],
            "L_mom": [],
            "L_eik": [],
            "ema_div": [],
            "ema_adv": [],
            "ema_mom": [],
            "ema_eik": [],
        }

    @staticmethod
    def make_flow_history() -> dict:
        return {
            "epoch": [],
            "loss": [],
            "loss_uvp": [],
            "loss_phys": [],
            "loss_gauge": [],
            "L_u": [],
            "L_v": [],
            "L_p": [],
            "L_div": [],
            "L_adv": [],
            "L_mom": [],
            "L_eik": [],
            "ema_div": [],
            "ema_adv": [],
            "ema_mom": [],
            "ema_eik": [],
            "s": [],
        }

    @staticmethod
    def make_flow_supervised_history() -> dict:
        return {
            "epoch": [],
            "loss": [],
            "loss_uvp": [],
            "L_u": [],
            "L_v": [],
            "L_p": [],
        }

    @staticmethod
    def make_joint_history() -> dict:
        return {
            "epoch": [],
            "loss": [],
            "loss_data": [],
            "loss_uvp": [],
            "loss_phys": [],
            "loss_gauge": [],
            "s": [],
            "L_u": [],
            "L_v": [],
            "L_p": [],
            "L_div": [],
            "L_adv": [],
            "L_mom": [],
            "L_eik": [],
            "ema_div": [],
            "ema_adv": [],
            "ema_mom": [],
            "ema_eik": [],
        }

    @staticmethod
    def _ema_to_cpu(ema: dict) -> dict:
        out = {}
        for key, value in ema.items():
            out[key] = None if value is None else value.detach().cpu()
        return out

    def save_checkpoint(self, model: nn.Module, ckpt_path: str, epoch: int, stage: str) -> None:
        torch.save({"epoch": epoch, "stage": stage, "model_state_dict": model.state_dict()}, ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")

    def save_full_weights(self, model: nn.Module, epoch: int) -> None:
        os.makedirs(self.cfg.full_weights_dir, exist_ok=True)
        weights_path = f"{self.cfg.full_weights_dir}/model_full_epoch_{epoch:04d}.pt"
        torch.save(
            {
                "epoch": epoch,
                "stage": "full",
                "model_state_dict": model.state_dict(),
            },
            weights_path,
        )
        print(f"Saved full weights: {weights_path}")

    def save_flow_stage_state(
        self,
        model: nn.Module,
        opt: torch.optim.Optimizer,
        history: dict,
        ema: dict,
        ema_steps: int,
        epoch: int,
        save_periodic: bool = False,
    ) -> None:
        os.makedirs(self.cfg.flow_state_dir, exist_ok=True)
        state = {
            "epoch": epoch,
            "stage": "flow",
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "history": history,
            "ema": self._ema_to_cpu(ema),
            "ema_steps": ema_steps,
        }

        torch.save(state, self.cfg.flow_latest_state_path)
        with open(self.cfg.flow_latest_history_path, "w", encoding="utf-8") as fh:
            json.dump(history, fh, ensure_ascii=False, indent=2)

        if save_periodic:
            state_path = f"{self.cfg.flow_state_dir}/flow_epoch_{epoch:04d}.pt"
            history_path = f"{self.cfg.flow_state_dir}/flow_epoch_{epoch:04d}_history.json"
            torch.save(state, state_path)
            with open(history_path, "w", encoding="utf-8") as fh:
                json.dump(history, fh, ensure_ascii=False, indent=2)
            print(f"Saved flow-stage state: {state_path}")
            print(f"Saved flow-stage history: {history_path}")

    def save_flow_weights(self, model: nn.Module, epoch: int) -> None:
        os.makedirs(self.cfg.flow_weights_dir, exist_ok=True)
        weights_path = f"{self.cfg.flow_weights_dir}/model_flow_epoch_{epoch:04d}.pt"
        torch.save(
            {
                "epoch": epoch,
                "stage": "flow",
                "model_state_dict": model.state_dict(),
            },
            weights_path,
        )
        print(f"Saved flow weights: {weights_path}")

    def save_flow_supervised_state(
        self,
        model: nn.Module,
        opt: torch.optim.Optimizer,
        history: dict,
        epoch: int,
        save_periodic: bool = False,
    ) -> None:
        os.makedirs(self.cfg.flow_supervised_state_dir, exist_ok=True)
        state = {
            "epoch": epoch,
            "stage": "flow_supervised",
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "history": history,
        }

        torch.save(state, self.cfg.flow_supervised_latest_state_path)
        with open(self.cfg.flow_supervised_latest_history_path, "w", encoding="utf-8") as fh:
            json.dump(history, fh, ensure_ascii=False, indent=2)

        if save_periodic:
            state_path = f"{self.cfg.flow_supervised_state_dir}/flow_supervised_epoch_{epoch:04d}.pt"
            history_path = f"{self.cfg.flow_supervised_state_dir}/flow_supervised_epoch_{epoch:04d}_history.json"
            torch.save(state, state_path)
            with open(history_path, "w", encoding="utf-8") as fh:
                json.dump(history, fh, ensure_ascii=False, indent=2)
            print(f"Saved flow-supervised state: {state_path}")
            print(f"Saved flow-supervised history: {history_path}")

    def save_flow_supervised_weights(self, model: nn.Module, epoch: int) -> None:
        os.makedirs(self.cfg.flow_supervised_weights_dir, exist_ok=True)
        weights_path = f"{self.cfg.flow_supervised_weights_dir}/model_flow_supervised_epoch_{epoch:04d}.pt"
        torch.save(
            {
                "epoch": epoch,
                "stage": "flow_supervised",
                "model_state_dict": model.state_dict(),
            },
            weights_path,
        )
        print(f"Saved flow-supervised weights: {weights_path}")

    def save_full_training_state(
        self,
        model: nn.Module,
        opt: torch.optim.Optimizer,
        history: dict,
        ema: dict,
        ema_steps: int,
        epoch: int,
        save_periodic: bool = False,
    ) -> None:
        os.makedirs(self.cfg.full_state_dir, exist_ok=True)
        state = {
            "epoch": epoch,
            "stage": "full",
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "history": history,
            "ema": self._ema_to_cpu(ema),
            "ema_steps": ema_steps,
        }

        torch.save(state, self.cfg.full_latest_state_path)
        with open(self.cfg.full_latest_history_path, "w", encoding="utf-8") as fh:
            json.dump(history, fh, ensure_ascii=False, indent=2)

        if save_periodic:
            state_path = f"{self.cfg.full_state_dir}/full_epoch_{epoch:04d}.pt"
            history_path = f"{self.cfg.full_state_dir}/full_epoch_{epoch:04d}_history.json"
            torch.save(state, state_path)
            with open(history_path, "w", encoding="utf-8") as fh:
                json.dump(history, fh, ensure_ascii=False, indent=2)
            print(f"Saved full-training state: {state_path}")
            print(f"Saved full-training history: {history_path}")

    def load_full_training_state(self, model: nn.Module, opt: torch.optim.Optimizer, ckpt_path: str):
        state = torch.load(ckpt_path, map_location=self.cfg.device)
        model.load_state_dict(state["model_state_dict"])
        opt.load_state_dict(state["optimizer_state_dict"])
        history = state.get("history", self.make_full_history())
        ema = state.get("ema", {"div": None, "adv": None, "mom": None, "eik": None})
        ema_steps = state.get("ema_steps", 0)
        start_epoch = int(state.get("epoch", 0)) + 1
        return model, opt, history, ema, ema_steps, start_epoch

    def train_pretrain(self, model: nn.Module, opt: torch.optim.Optimizer, epochs: int):
        history = self.make_pretrain_history()
        s_pretrain_tensor = torch.tensor(self.cfg.s_pretrain, device=self.cfg.device, dtype=self.cfg.dtype)

        for epoch in range(1, epochs + 1):
            model.train()
            running_loss = 0.0
            running_loss_data = 0.0

            with tqdm(
                self.data_loader,
                desc=f"Pretrain {epoch}/{epochs}",
                total=len(self.data_loader),
                dynamic_ncols=True,
                leave=False,
            ) as pbar:
                for xyt_d_np, phi_np in pbar:
                    xyt_d = xyt_d_np.to(self.cfg.device, dtype=self.cfg.dtype)
                    phi_true = phi_np.to(self.cfg.device, dtype=self.cfg.dtype)

                    _, _, _, alpha_d = self.model_fields(model, xyt_d)
                    phi_pred = torch.tanh(s_pretrain_tensor * alpha_d)
                    loss_data = torch.mean((phi_pred - phi_true) ** 2)
                    loss = self.cfg.lambda_data_pretrain * loss_data

                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()

                    running_loss += float(loss.item())
                    running_loss_data += float(loss_data.item())
                    pbar.set_postfix(loss=f"{loss.item():.4e}", loss_data=f"{loss_data.item():.4e}")

            n_batches = len(self.data_loader)
            history["epoch"].append(epoch)
            history["loss"].append(running_loss / n_batches)
            history["loss_data"].append(running_loss_data / n_batches)
            history["s"].append(self.cfg.s_pretrain)

            if epoch % 25 == 0 or epoch == 1:
                print(
                    f"pretrain_epoch={epoch:5d} "
                    f"loss={history['loss'][-1]:.4e} "
                    f"loss_data={history['loss_data'][-1]:.4e}"
                )

        self.save_checkpoint(model, self.cfg.pretrain_ckpt_path, epochs, stage="pretrain")
        return history

    def train_full(
        self,
        model: nn.Module,
        opt: torch.optim.Optimizer,
        epochs: int,
        start_epoch: int = 1,
        history: dict | None = None,
        ema: dict | None = None,
        ema_steps: int = 0,
    ):
        history = self.make_full_history() if history is None else history
        ema = {"div": None, "adv": None, "mom": None, "eik": None} if ema is None else ema

        def ema_update(key: str, value: torch.Tensor) -> None:
            nonlocal ema_steps
            detached = value.detach()
            if ema[key] is None:
                ema[key] = detached
            else:
                ema[key] = self.cfg.ema_beta * ema[key] + (1.0 - self.cfg.ema_beta) * detached

        def ema_scale(key: str) -> torch.Tensor:
            if ema[key] is None:
                raise RuntimeError(f"EMA for {key} is not initialized")
            if ema_steps < self.cfg.ema_warmup_steps:
                return (ema[key] + self.cfg.ema_eps).detach() + 1.0
            return (ema[key] + self.cfg.ema_eps).detach()

        n_steps = max(len(self.data_loader), len(self.phys_loader), len(self.uvp_loader))
        last_completed_epoch = history["epoch"][-1] if history["epoch"] else (start_epoch - 1)

        try:
            for epoch in range(start_epoch, epochs + 1):
                model.train()
                s_full = self.get_s_full(epoch)
                s_full_tensor = torch.tensor(s_full, device=self.cfg.device, dtype=self.cfg.dtype)
                running = {k: 0.0 for k in [
                    "loss", "loss_data", "loss_phys", "loss_uvp", "loss_gauge",
                    "L_u", "L_v", "L_p", "L_div", "L_adv", "L_mom", "L_eik"
                ]}

                data_iter = cycle(self.data_loader)
                phys_iter = cycle(self.phys_loader)
                uvp_iter = cycle(self.uvp_loader)

                with tqdm(
                    range(n_steps),
                    desc=f"Full train {epoch}/{epochs}",
                    total=n_steps,
                    dynamic_ncols=True,
                    leave=False,
                ) as pbar:
                    for _ in pbar:
                        xyt_d_np, phi_np = next(data_iter)
                        xyt_p_np = next(phys_iter)
                        xyt_uvp_np, uvp_np = next(uvp_iter)

                        xyt_d = xyt_d_np.to(self.cfg.device, dtype=self.cfg.dtype)
                        phi_true = phi_np.to(self.cfg.device, dtype=self.cfg.dtype)
                        xyt_p = xyt_p_np.to(self.cfg.device, dtype=self.cfg.dtype)
                        xyt_uvp = xyt_uvp_np.to(self.cfg.device, dtype=self.cfg.dtype)
                        uvp_true = uvp_np.to(self.cfg.device, dtype=self.cfg.dtype)

                        _, _, _, alpha_d = self.model_fields(model, xyt_d)
                        phi_pred = torch.tanh(s_full_tensor * alpha_d)
                        loss_data = torch.mean((phi_pred - phi_true) ** 2)

                        loss_uvp, l_u, l_v, l_p = self.weighted_uvp_supervision_loss(model, xyt_uvp, uvp_true)

                        res = self.residuals(model, xyt_p, s_val=s_full)
                        l_div = torch.mean(res["r_div"] ** 2)
                        l_adv = torch.mean(res["r_adv"] ** 2)
                        l_eik = torch.mean(res["r_eik"] ** 2)
                        l_mom = torch.mean(res["r_mom_u"] ** 2) + torch.mean(res["r_mom_v"] ** 2)
                        loss_gauge = self.pressure_gauge_loss(model)

                        ema_update("div", l_div)
                        ema_update("adv", l_adv)
                        ema_update("mom", l_mom)
                        ema_update("eik", l_eik)
                        ema_steps += 1

                        l_div_n = l_div / ema_scale("div")
                        l_adv_n = l_adv / ema_scale("adv")
                        l_mom_n = l_mom / ema_scale("mom")
                        l_eik_n = l_eik / ema_scale("eik")

                        w_div, w_adv, w_mom, w_eik, lam_phys = self.get_phys_weights(epoch)
                        loss_phys = w_div * l_div_n + w_adv * l_adv_n + w_mom * l_mom_n + w_eik * l_eik_n
                        loss = (
                            self.cfg.lambda_data_full * loss_data
                            + self.cfg.lambda_uvp_full * loss_uvp
                            + lam_phys * loss_phys
                            + self.cfg.lambda_pressure_gauge * loss_gauge
                        )

                        opt.zero_grad(set_to_none=True)
                        loss.backward()
                        opt.step()

                        running["loss"] += float(loss.item())
                        running["loss_data"] += float(loss_data.item())
                        running["loss_phys"] += float(loss_phys.item())
                        running["loss_uvp"] += float(loss_uvp.item())
                        running["loss_gauge"] += float(loss_gauge.item())
                        running["L_u"] += float(l_u.item())
                        running["L_v"] += float(l_v.item())
                        running["L_p"] += float(l_p.item())
                        running["L_div"] += float(l_div.item())
                        running["L_adv"] += float(l_adv.item())
                        running["L_mom"] += float(l_mom.item())
                        running["L_eik"] += float(l_eik.item())

                        pbar.set_postfix(
                            loss=f"{loss.item():.4e}",
                            loss_data=f"{loss_data.item():.4e}",
                            loss_uvp=f"{loss_uvp.item():.2e}",
                            loss_gauge=f"{loss_gauge.item():.2e}",
                            s=f"{s_full:.1f}",
                            lambda_phys=f"{lam_phys:.2e}",
                            mom=f"{l_mom.item():.2e}",
                            u=f"{l_u.item():.2e}",
                            v=f"{l_v.item():.2e}",
                            p=f"{l_p.item():.2e}",
                        )

                history["epoch"].append(epoch)
                history["loss"].append(running["loss"] / n_steps)
                history["loss_data"].append(running["loss_data"] / n_steps)
                history["loss_phys"].append(running["loss_phys"] / n_steps)
                history["loss_uvp"].append(running["loss_uvp"] / n_steps)
                history["loss_gauge"].append(running["loss_gauge"] / n_steps)
                history["s"].append(s_full)
                history["L_u"].append(running["L_u"] / n_steps)
                history["L_v"].append(running["L_v"] / n_steps)
                history["L_p"].append(running["L_p"] / n_steps)
                history["L_div"].append(running["L_div"] / n_steps)
                history["L_adv"].append(running["L_adv"] / n_steps)
                history["L_mom"].append(running["L_mom"] / n_steps)
                history["L_eik"].append(running["L_eik"] / n_steps)
                history["ema_div"].append(float(ema["div"].item()))
                history["ema_adv"].append(float(ema["adv"].item()))
                history["ema_mom"].append(float(ema["mom"].item()))
                history["ema_eik"].append(float(ema["eik"].item()))
                last_completed_epoch = epoch

                self.save_full_training_state(
                    model,
                    opt,
                    history,
                    ema,
                    ema_steps,
                    epoch,
                    save_periodic=(epoch % self.cfg.full_save_every == 0),
                )
                if epoch % self.cfg.full_weights_save_every == 0:
                    self.save_full_weights(model, epoch)

                if epoch % 50 == 0 or epoch == start_epoch:
                    print(
                        f"full_epoch={epoch:5d} "
                        f"s={history['s'][-1]:.2f} "
                        f"loss={history['loss'][-1]:.4e} "
                        f"loss_data={history['loss_data'][-1]:.4e} "
                        f"loss_uvp={history['loss_uvp'][-1]:.4e} "
                        f"loss_phys={history['loss_phys'][-1]:.4e} "
                        f"loss_gauge={history['loss_gauge'][-1]:.4e} "
                        f"lambda_phys={self.get_phys_weights(epoch)[4]:.2e} "
                        f"L_u={history['L_u'][-1]:.4e} "
                        f"L_v={history['L_v'][-1]:.4e} "
                        f"L_p={history['L_p'][-1]:.4e} "
                        f"L_div={history['L_div'][-1]:.4e} "
                        f"L_adv={history['L_adv'][-1]:.4e} "
                        f"L_mom={history['L_mom'][-1]:.4e} "
                        f"L_eik={history['L_eik'][-1]:.4e}"
                    )
        finally:
            if last_completed_epoch >= start_epoch:
                self.save_full_training_state(
                    model,
                    opt,
                    history,
                    ema,
                    ema_steps,
                    last_completed_epoch,
                    save_periodic=False,
                )

        self.save_checkpoint(model, self.cfg.full_ckpt_path, last_completed_epoch, stage="full")
        return history

    def train_flow_stage(
        self,
        model: nn.Module,
        opt: torch.optim.Optimizer,
        epochs: int,
        start_epoch: int = 1,
        history: dict | None = None,
        ema: dict | None = None,
        ema_steps: int = 0,
    ):
        history = self.make_flow_history() if history is None else history
        ema = {"div": None, "adv": None, "mom": None, "eik": None} if ema is None else ema

        def ema_update(key: str, value: torch.Tensor) -> None:
            nonlocal ema_steps
            detached = value.detach()
            if ema[key] is None:
                ema[key] = detached
            else:
                ema[key] = self.cfg.ema_beta * ema[key] + (1.0 - self.cfg.ema_beta) * detached

        def ema_scale(key: str) -> torch.Tensor:
            if ema[key] is None:
                raise RuntimeError(f"EMA for {key} is not initialized")
            if ema_steps < self.cfg.ema_warmup_steps:
                return (ema[key] + self.cfg.ema_eps).detach() + 1.0
            return (ema[key] + self.cfg.ema_eps).detach()

        self.freeze_alpha_head(model)
        n_steps = max(len(self.phys_loader), len(self.uvp_loader))
        last_completed_epoch = history["epoch"][-1] if history["epoch"] else (start_epoch - 1)

        try:
            for epoch in range(start_epoch, epochs + 1):
                model.train()
                s_full = self.get_s_full(epoch)
                running = {k: 0.0 for k in [
                    "loss", "loss_uvp", "loss_phys", "loss_gauge",
                    "L_u", "L_v", "L_p", "L_div", "L_adv", "L_mom", "L_eik"
                ]}

                phys_iter = cycle(self.phys_loader)
                uvp_iter = cycle(self.uvp_loader)

                with tqdm(
                    range(n_steps),
                    desc=f"Flow stage {epoch}/{epochs}",
                    total=n_steps,
                    dynamic_ncols=True,
                    leave=False,
                ) as pbar:
                    for _ in pbar:
                        xyt_p_np = next(phys_iter)
                        xyt_uvp_np, uvp_np = next(uvp_iter)

                        xyt_p = xyt_p_np.to(self.cfg.device, dtype=self.cfg.dtype)
                        xyt_uvp = xyt_uvp_np.to(self.cfg.device, dtype=self.cfg.dtype)
                        uvp_true = uvp_np.to(self.cfg.device, dtype=self.cfg.dtype)

                        loss_uvp, l_u, l_v, l_p = self.weighted_uvp_supervision_loss(model, xyt_uvp, uvp_true)

                        res = self.residuals(model, xyt_p, s_val=s_full)
                        l_div = torch.mean(res["r_div"] ** 2)
                        l_adv = torch.mean(res["r_adv"] ** 2)
                        l_eik = torch.mean(res["r_eik"] ** 2)
                        l_mom = torch.mean(res["r_mom_u"] ** 2) + torch.mean(res["r_mom_v"] ** 2)
                        loss_gauge = self.pressure_gauge_loss(model)

                        ema_update("div", l_div)
                        ema_update("adv", l_adv)
                        ema_update("mom", l_mom)
                        ema_update("eik", l_eik)
                        ema_steps += 1

                        l_div_n = l_div / ema_scale("div")
                        l_adv_n = l_adv / ema_scale("adv")
                        l_mom_n = l_mom / ema_scale("mom")
                        l_eik_n = l_eik / ema_scale("eik")

                        w_div, w_adv, w_mom, lam_phys = self.get_flow_stage_weights(epoch)
                        loss_phys = w_div * l_div_n + w_adv * l_adv_n + w_mom * l_mom_n
                        loss = self.cfg.lambda_uvp_full * loss_uvp + lam_phys * loss_phys + self.cfg.lambda_pressure_gauge * loss_gauge

                        opt.zero_grad(set_to_none=True)
                        loss.backward()
                        opt.step()

                        running["loss"] += float(loss.item())
                        running["loss_uvp"] += float(loss_uvp.item())
                        running["loss_phys"] += float(loss_phys.item())
                        running["loss_gauge"] += float(loss_gauge.item())
                        running["L_u"] += float(l_u.item())
                        running["L_v"] += float(l_v.item())
                        running["L_p"] += float(l_p.item())
                        running["L_div"] += float(l_div.item())
                        running["L_adv"] += float(l_adv.item())
                        running["L_mom"] += float(l_mom.item())
                        running["L_eik"] += float(l_eik.item())

                        pbar.set_postfix(
                            loss=f"{loss.item():.4e}",
                            loss_uvp=f"{loss_uvp.item():.2e}",
                            loss_gauge=f"{loss_gauge.item():.2e}",
                            s=f"{s_full:.1f}",
                            lambda_phys=f"{lam_phys:.2e}",
                            mom=f"{l_mom.item():.2e}",
                            u=f"{l_u.item():.2e}",
                            v=f"{l_v.item():.2e}",
                            p=f"{l_p.item():.2e}",
                        )

                history["epoch"].append(epoch)
                history["loss"].append(running["loss"] / n_steps)
                history["loss_uvp"].append(running["loss_uvp"] / n_steps)
                history["loss_phys"].append(running["loss_phys"] / n_steps)
                history["loss_gauge"].append(running["loss_gauge"] / n_steps)
                history["L_u"].append(running["L_u"] / n_steps)
                history["L_v"].append(running["L_v"] / n_steps)
                history["L_p"].append(running["L_p"] / n_steps)
                history["L_div"].append(running["L_div"] / n_steps)
                history["L_adv"].append(running["L_adv"] / n_steps)
                history["L_mom"].append(running["L_mom"] / n_steps)
                history["L_eik"].append(running["L_eik"] / n_steps)
                history["ema_div"].append(float(ema["div"].item()))
                history["ema_adv"].append(float(ema["adv"].item()))
                history["ema_mom"].append(float(ema["mom"].item()))
                history["ema_eik"].append(float(ema["eik"].item()))
                history["s"].append(s_full)
                last_completed_epoch = epoch

                self.save_flow_stage_state(
                    model,
                    opt,
                    history,
                    ema,
                    ema_steps,
                    epoch,
                    save_periodic=(epoch % self.cfg.flow_save_every == 0),
                )
                if epoch % self.cfg.flow_save_every == 0:
                    self.save_flow_weights(model, epoch)

                if epoch % 25 == 0 or epoch == start_epoch:
                    print(
                        f"flow_epoch={epoch:5d} "
                        f"s={history['s'][-1]:.2f} "
                        f"loss={history['loss'][-1]:.4e} "
                        f"loss_uvp={history['loss_uvp'][-1]:.4e} "
                        f"loss_phys={history['loss_phys'][-1]:.4e} "
                        f"loss_gauge={history['loss_gauge'][-1]:.4e} "
                        f"lambda_phys={self.get_flow_stage_weights(epoch)[3]:.2e} "
                        f"L_u={history['L_u'][-1]:.4e} "
                        f"L_v={history['L_v'][-1]:.4e} "
                        f"L_p={history['L_p'][-1]:.4e} "
                        f"L_div={history['L_div'][-1]:.4e} "
                        f"L_adv={history['L_adv'][-1]:.4e} "
                        f"L_mom={history['L_mom'][-1]:.4e}"
                    )
        finally:
            if last_completed_epoch >= start_epoch:
                self.save_flow_stage_state(
                    model,
                    opt,
                    history,
                    ema,
                    ema_steps,
                    last_completed_epoch,
                    save_periodic=False,
                )

        return history

    def train_flow_supervised_stage(
        self,
        model: nn.Module,
        opt: torch.optim.Optimizer,
        epochs: int,
        start_epoch: int = 1,
        history: dict | None = None,
    ):
        history = self.make_flow_supervised_history() if history is None else history
        self.freeze_alpha_head(model)

        n_steps = len(self.uvp_loader)
        last_completed_epoch = history["epoch"][-1] if history["epoch"] else (start_epoch - 1)

        try:
            for epoch in range(start_epoch, epochs + 1):
                model.train()
                running = {"loss": 0.0, "loss_uvp": 0.0, "L_u": 0.0, "L_v": 0.0, "L_p": 0.0}

                with tqdm(
                    self.uvp_loader,
                    desc=f"Flow supervised {epoch}/{epochs}",
                    total=n_steps,
                    dynamic_ncols=True,
                    leave=False,
                ) as pbar:
                    for xyt_uvp_np, uvp_np in pbar:
                        xyt_uvp = xyt_uvp_np.to(self.cfg.device, dtype=self.cfg.dtype)
                        uvp_true = uvp_np.to(self.cfg.device, dtype=self.cfg.dtype)

                        loss_uvp, l_u, l_v, l_p = self.weighted_uvp_supervision_loss(model, xyt_uvp, uvp_true)
                        loss = self.cfg.lambda_uvp_full * loss_uvp

                        opt.zero_grad(set_to_none=True)
                        loss.backward()
                        opt.step()

                        running["loss"] += float(loss.item())
                        running["loss_uvp"] += float(loss_uvp.item())
                        running["L_u"] += float(l_u.item())
                        running["L_v"] += float(l_v.item())
                        running["L_p"] += float(l_p.item())

                        pbar.set_postfix(
                            loss=f"{loss.item():.4e}",
                            loss_uvp=f"{loss_uvp.item():.2e}",
                            u=f"{l_u.item():.2e}",
                            v=f"{l_v.item():.2e}",
                            p=f"{l_p.item():.2e}",
                        )

                history["epoch"].append(epoch)
                history["loss"].append(running["loss"] / n_steps)
                history["loss_uvp"].append(running["loss_uvp"] / n_steps)
                history["L_u"].append(running["L_u"] / n_steps)
                history["L_v"].append(running["L_v"] / n_steps)
                history["L_p"].append(running["L_p"] / n_steps)
                last_completed_epoch = epoch

                self.save_flow_supervised_state(
                    model,
                    opt,
                    history,
                    epoch,
                    save_periodic=(epoch % self.cfg.flow_supervised_save_every == 0),
                )
                if epoch % self.cfg.flow_supervised_save_every == 0:
                    self.save_flow_supervised_weights(model, epoch)

                if epoch % 25 == 0 or epoch == start_epoch:
                    print(
                        f"flow_sup_epoch={epoch:5d} "
                        f"loss={history['loss'][-1]:.4e} "
                        f"loss_uvp={history['loss_uvp'][-1]:.4e} "
                        f"L_u={history['L_u'][-1]:.4e} "
                        f"L_v={history['L_v'][-1]:.4e} "
                        f"L_p={history['L_p'][-1]:.4e}"
                    )
        finally:
            if last_completed_epoch >= start_epoch:
                self.save_flow_supervised_state(
                    model,
                    opt,
                    history,
                    last_completed_epoch,
                    save_periodic=False,
                )

        return history

    def train_joint_finetune_stage(
        self,
        model: nn.Module,
        opt: torch.optim.Optimizer,
        epochs: int,
        start_epoch: int = 1,
        history: dict | None = None,
        ema: dict | None = None,
        ema_steps: int = 0,
    ):
        history = self.make_joint_history() if history is None else history
        ema = {"div": None, "adv": None, "mom": None, "eik": None} if ema is None else ema
        self.unfreeze_all(model)

        def ema_update(key: str, value: torch.Tensor) -> None:
            nonlocal ema_steps
            detached = value.detach()
            if ema[key] is None:
                ema[key] = detached
            else:
                ema[key] = self.cfg.ema_beta * ema[key] + (1.0 - self.cfg.ema_beta) * detached

        def ema_scale(key: str) -> torch.Tensor:
            if ema[key] is None:
                raise RuntimeError(f"EMA for {key} is not initialized")
            if ema_steps < self.cfg.ema_warmup_steps:
                return (ema[key] + self.cfg.ema_eps).detach() + 1.0
            return (ema[key] + self.cfg.ema_eps).detach()

        n_steps = max(len(self.data_loader), len(self.phys_loader), len(self.uvp_loader))

        for epoch in range(start_epoch, epochs + 1):
            model.train()
            s_full = self.get_s_full(epoch)
            s_full_tensor = torch.tensor(s_full, device=self.cfg.device, dtype=self.cfg.dtype)
            running = {k: 0.0 for k in [
                "loss", "loss_data", "loss_uvp", "loss_phys", "loss_gauge",
                "L_u", "L_v", "L_p", "L_div", "L_adv", "L_mom", "L_eik"
            ]}

            data_iter = cycle(self.data_loader)
            phys_iter = cycle(self.phys_loader)
            uvp_iter = cycle(self.uvp_loader)

            with tqdm(
                range(n_steps),
                desc=f"Joint finetune {epoch}/{epochs}",
                total=n_steps,
                dynamic_ncols=True,
                leave=False,
            ) as pbar:
                for _ in pbar:
                    xyt_d_np, phi_np = next(data_iter)
                    xyt_p_np = next(phys_iter)
                    xyt_uvp_np, uvp_np = next(uvp_iter)

                    xyt_d = xyt_d_np.to(self.cfg.device, dtype=self.cfg.dtype)
                    phi_true = phi_np.to(self.cfg.device, dtype=self.cfg.dtype)
                    xyt_p = xyt_p_np.to(self.cfg.device, dtype=self.cfg.dtype)
                    xyt_uvp = xyt_uvp_np.to(self.cfg.device, dtype=self.cfg.dtype)
                    uvp_true = uvp_np.to(self.cfg.device, dtype=self.cfg.dtype)

                    _, _, _, alpha_d = self.model_fields(model, xyt_d)
                    phi_pred = torch.tanh(s_full_tensor * alpha_d)
                    loss_data = torch.mean((phi_pred - phi_true) ** 2)

                    loss_uvp, l_u, l_v, l_p = self.weighted_uvp_supervision_loss(model, xyt_uvp, uvp_true)

                    res = self.residuals(model, xyt_p, s_val=s_full)
                    l_div = torch.mean(res["r_div"] ** 2)
                    l_adv = torch.mean(res["r_adv"] ** 2)
                    l_eik = torch.mean(res["r_eik"] ** 2)
                    l_mom = torch.mean(res["r_mom_u"] ** 2) + torch.mean(res["r_mom_v"] ** 2)
                    loss_gauge = self.pressure_gauge_loss(model)

                    ema_update("div", l_div)
                    ema_update("adv", l_adv)
                    ema_update("mom", l_mom)
                    ema_update("eik", l_eik)
                    ema_steps += 1

                    l_div_n = l_div / ema_scale("div")
                    l_adv_n = l_adv / ema_scale("adv")
                    l_mom_n = l_mom / ema_scale("mom")
                    l_eik_n = l_eik / ema_scale("eik")

                    w_div, w_adv, w_mom, w_eik, lam_phys = self.get_joint_finetune_weights(epoch)
                    loss_phys = w_div * l_div_n + w_adv * l_adv_n + w_mom * l_mom_n + w_eik * l_eik_n

                    loss = (
                        self.cfg.lambda_data_full * loss_data
                        + self.cfg.lambda_uvp_full * loss_uvp
                        + lam_phys * loss_phys
                        + self.cfg.lambda_pressure_gauge * loss_gauge
                    )

                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()

                    running["loss"] += float(loss.item())
                    running["loss_data"] += float(loss_data.item())
                    running["loss_uvp"] += float(loss_uvp.item())
                    running["loss_phys"] += float(loss_phys.item())
                    running["loss_gauge"] += float(loss_gauge.item())
                    running["L_u"] += float(l_u.item())
                    running["L_v"] += float(l_v.item())
                    running["L_p"] += float(l_p.item())
                    running["L_div"] += float(l_div.item())
                    running["L_adv"] += float(l_adv.item())
                    running["L_mom"] += float(l_mom.item())
                    running["L_eik"] += float(l_eik.item())

                    pbar.set_postfix(
                        loss=f"{loss.item():.4e}",
                        loss_data=f"{loss_data.item():.2e}",
                        loss_uvp=f"{loss_uvp.item():.2e}",
                        lambda_phys=f"{lam_phys:.2e}",
                        mom=f"{l_mom.item():.2e}",
                    )

            history["epoch"].append(epoch)
            history["loss"].append(running["loss"] / n_steps)
            history["loss_data"].append(running["loss_data"] / n_steps)
            history["loss_uvp"].append(running["loss_uvp"] / n_steps)
            history["loss_phys"].append(running["loss_phys"] / n_steps)
            history["loss_gauge"].append(running["loss_gauge"] / n_steps)
            history["L_u"].append(running["L_u"] / n_steps)
            history["L_v"].append(running["L_v"] / n_steps)
            history["L_p"].append(running["L_p"] / n_steps)
            history["L_div"].append(running["L_div"] / n_steps)
            history["L_adv"].append(running["L_adv"] / n_steps)
            history["L_mom"].append(running["L_mom"] / n_steps)
            history["L_eik"].append(running["L_eik"] / n_steps)
            history["ema_div"].append(float(ema["div"].item()))
            history["ema_adv"].append(float(ema["adv"].item()))
            history["ema_mom"].append(float(ema["mom"].item()))
            history["ema_eik"].append(float(ema["eik"].item()))
            history["s"].append(s_full)

            if epoch % 25 == 0 or epoch == start_epoch:
                print(
                    f"joint_epoch={epoch:5d} "
                    f"s={history['s'][-1]:.2f} "
                    f"loss={history['loss'][-1]:.4e} "
                    f"loss_data={history['loss_data'][-1]:.4e} "
                    f"loss_uvp={history['loss_uvp'][-1]:.4e} "
                    f"loss_phys={history['loss_phys'][-1]:.4e} "
                    f"lambda_phys={self.get_joint_finetune_weights(epoch)[4]:.2e} "
                    f"L_u={history['L_u'][-1]:.4e} "
                    f"L_v={history['L_v'][-1]:.4e} "
                    f"L_p={history['L_p'][-1]:.4e} "
                    f"L_mom={history['L_mom'][-1]:.4e}"
                )

        return history
