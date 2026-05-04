from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from .config import PhysicsConfig


def _activation(name: str) -> nn.Module:
    if name == "tanh":
        return nn.Tanh()
    if name == "silu":
        return nn.SiLU()
    if name == "gelu":
        return nn.GELU()
    if name == "sin":
        return Sine()
    raise ValueError(f"Unsupported activation '{name}'.")


class Sine(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x)


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_layers: tuple[int, ...], activation: str = "tanh"):
        super().__init__()
        layers: list[nn.Module] = []
        previous = input_dim
        for width in hidden_layers:
            linear = nn.Linear(previous, width)
            nn.init.xavier_normal_(linear.weight)
            nn.init.zeros_(linear.bias)
            layers.extend([linear, _activation(activation)])
            previous = width

        self.trunk = nn.Sequential(*layers)
        self.u_head = nn.Linear(previous, 1)
        self.v_head = nn.Linear(previous, 1)
        self.p_head = nn.Linear(previous, 1)
        self.alpha_head = nn.Linear(previous, 1)
        for head in (self.u_head, self.v_head, self.p_head, self.alpha_head):
            nn.init.xavier_normal_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, xyt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.trunk(xyt)
        u = self.u_head(z)
        v = self.v_head(z)
        p = torch.exp(torch.clamp(self.p_head(z), min=-20.0, max=20.0))
        alpha = torch.sigmoid(self.alpha_head(z))
        return u, v, p, alpha


@dataclass
class LossBreakdown:
    total: torch.Tensor
    alpha: torch.Tensor
    boundary: torch.Tensor
    mass: torch.Tensor
    momentum_x: torch.Tensor
    momentum_y: torch.Tensor
    alpha_pde: torch.Tensor

    def detached(self) -> dict[str, float]:
        return {
            "total": float(self.total.detach().cpu()),
            "alpha": float(self.alpha.detach().cpu()),
            "boundary": float(self.boundary.detach().cpu()),
            "mass": float(self.mass.detach().cpu()),
            "momentum_x": float(self.momentum_x.detach().cpu()),
            "momentum_y": float(self.momentum_y.detach().cpu()),
            "alpha_pde": float(self.alpha_pde.detach().cpu()),
        }


class TwoPhasePINN(nn.Module):
    def __init__(
        self,
        hidden_layers: tuple[int, ...],
        physics: PhysicsConfig,
        activation: str = "tanh",
        loss_weights_pde: tuple[float, float, float, float] = (1.0, 10.0, 10.0, 1.0),
    ):
        super().__init__()
        self.net = MLP(3, hidden_layers, activation)
        self.physics = physics
        self.register_buffer("loss_weights_pde", torch.tensor(loss_weights_pde, dtype=torch.float32))

    def forward(self, xyt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.net(xyt)

    @staticmethod
    def _grad(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return torch.autograd.grad(
            y,
            x,
            grad_outputs=torch.ones_like(y),
            retain_graph=True,
            create_graph=True,
            only_inputs=True,
        )[0]

    def pde_residuals(self, xyt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        xyt = xyt.detach().clone().requires_grad_(True)
        u, v, p, alpha = self.forward(xyt)

        grad_u = self._grad(u, xyt)
        grad_v = self._grad(v, xyt)
        grad_p = self._grad(p, xyt)
        grad_alpha = self._grad(alpha, xyt)

        u_x, u_y, u_t = grad_u[:, 0:1], grad_u[:, 1:2], grad_u[:, 2:3]
        v_x, v_y, v_t = grad_v[:, 0:1], grad_v[:, 1:2], grad_v[:, 2:3]
        p_x, p_y = grad_p[:, 0:1], grad_p[:, 1:2]
        a_x, a_y, a_t = grad_alpha[:, 0:1], grad_alpha[:, 1:2], grad_alpha[:, 2:3]

        u_xx = self._grad(u_x, xyt)[:, 0:1]
        u_yy = self._grad(u_y, xyt)[:, 1:2]
        v_xx = self._grad(v_x, xyt)[:, 0:1]
        v_yy = self._grad(v_y, xyt)[:, 1:2]
        a_xx = self._grad(a_x, xyt)[:, 0:1]
        a_yy = self._grad(a_y, xyt)[:, 1:2]
        a_xy = self._grad(a_x, xyt)[:, 1:2]

        phys = self.physics
        mu1, mu2 = phys.mu
        rho1, rho2 = phys.rho
        mu = mu2 + (mu1 - mu2) * alpha
        mu_x = (mu1 - mu2) * a_x
        mu_y = (mu1 - mu2) * a_y
        rho = rho2 + (rho1 - rho2) * alpha

        eps = torch.finfo(xyt.dtype).eps
        abs_interface_grad = torch.sqrt(a_x.square() + a_y.square() + eps)
        curvature = -(
            (a_xx + a_yy) / abs_interface_grad
            - (a_x.square() * a_xx + a_y.square() * a_yy + 2.0 * a_x * a_y * a_xy)
            / abs_interface_grad.pow(3)
        )

        rho_ref = rho2
        one_re = mu / (rho_ref * phys.u_ref * phys.l_ref)
        one_re_x = mu_x / (rho_ref * phys.u_ref * phys.l_ref)
        one_re_y = mu_y / (rho_ref * phys.u_ref * phys.l_ref)
        one_we = phys.sigma / (rho_ref * phys.u_ref**2 * phys.l_ref)
        one_fr = phys.g * phys.l_ref / phys.u_ref**2

        mass = u_x + v_y
        alpha_transport = a_t + u * a_x + v * a_y
        momentum_x = (
            (u_t + u * u_x + v * u_y) * rho / rho_ref
            + p_x
            - one_we * curvature * a_x
            - one_re * (u_xx + u_yy)
            - 2.0 * one_re_x * u_x
            - one_re_y * (u_y + v_x)
        )
        momentum_y = (
            (v_t + u * v_x + v * v_y) * rho / rho_ref
            + p_y
            - one_we * curvature * a_y
            - one_re * (v_xx + v_yy)
            - rho / rho_ref * one_fr
            - 2.0 * one_re_y * v_y
            - one_re_x * (u_y + v_x)
        )
        return mass, momentum_x, momentum_y, alpha_transport

    def loss(self, batches: dict[str, torch.Tensor]) -> LossBreakdown:
        alpha_batch = batches["alpha"]
        _, _, _, pred_alpha = self.forward(alpha_batch[:, 0:3])
        loss_alpha = F.mse_loss(pred_alpha, alpha_batch[:, 3:4])

        north = batches["north"]
        _, _, p_n, _ = self.forward(north[:, 0:3])
        loss_north_p = F.mse_loss(p_n, north[:, 3:4])

        east_west = batches["east_west"]
        u_e, v_e, p_e, _ = self.forward(torch.stack([east_west[:, 0], east_west[:, 1], east_west[:, 4]], dim=1))
        u_w, v_w, p_w, _ = self.forward(east_west[:, 2:5])
        loss_ew = F.mse_loss(u_e, u_w) + F.mse_loss(v_e, v_w) + F.mse_loss(p_e, p_w)

        nsew = batches["nsew"]
        u_b, v_b, _, _ = self.forward(nsew[:, 0:3])
        loss_velocity_bc = F.mse_loss(u_b, nsew[:, 3:4]) + F.mse_loss(v_b, nsew[:, 4:5])
        loss_boundary = loss_north_p + loss_ew + loss_velocity_bc

        pde_batch = batches["pde"]
        target = pde_batch[:, 3:4]
        mass, momentum_x, momentum_y, alpha_transport = self.pde_residuals(pde_batch[:, 0:3])
        loss_mass = F.mse_loss(mass, target)
        loss_momentum_x = F.mse_loss(momentum_x, target)
        loss_momentum_y = F.mse_loss(momentum_y, target)
        loss_alpha_pde = F.mse_loss(alpha_transport, target)

        weights = self.loss_weights_pde.to(loss_mass.device, dtype=loss_mass.dtype)
        loss_pde = (
            weights[0] * loss_mass
            + weights[1] * loss_momentum_x
            + weights[2] * loss_momentum_y
            + weights[3] * loss_alpha_pde
        )
        total = loss_alpha + loss_boundary + loss_pde
        return LossBreakdown(
            total=total,
            alpha=loss_alpha,
            boundary=loss_boundary,
            mass=loss_mass,
            momentum_x=loss_momentum_x,
            momentum_y=loss_momentum_y,
            alpha_pde=loss_alpha_pde,
        )

    @torch.no_grad()
    def predict(self, xyt: torch.Tensor, batch_size: int = 1_000_000) -> tuple[torch.Tensor, ...]:
        outputs = []
        for chunk in xyt.split(batch_size):
            outputs.append(self.forward(chunk))
        return tuple(torch.cat([item[i] for item in outputs], dim=0) for i in range(4))
