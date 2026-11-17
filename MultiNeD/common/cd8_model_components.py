
from pathlib import Path as _PathForBootstrap
import sys as _sysForBootstrap

_CODE_DIR = _PathForBootstrap(__file__).resolve().parents[1]
if str(_CODE_DIR) not in _sysForBootstrap.path:
    _sysForBootstrap.path.insert(0, str(_CODE_DIR))

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ReactionEffectConfig:
    """Parameters of the paper reaction effects F(B) and G(P,B)."""
    f_b_coeff: float = 1.0
    g_b_coeff: float = 1.0
    g_pd1_coeff: float = 1.0
    pd1_k: float = 1.0
    b41bb_k: float = 1.0
    eps: float = 1e-6
    learnable: bool = False


@dataclass
class PKConfig:
    """Configuration for optional first-order antibody PK dynamics."""
    enabled: bool = False
    delta_pd1: float = 0.2
    delta_b41bb: float = 0.2
    pd1_dose_days: tuple = (1.0, 4.0)
    b41bb_dose_days: tuple = (2.0,)
    pd1_dose_amount: float = 1.0
    b41bb_dose_amount: float = 1.0
    normalize_to_unit_peak: bool = True


@dataclass
class DriftModelConfig:
    """Configuration for the neural drift field v_theta(z, P, B, t)."""
    z_dim: int = 5
    treatment_dim: int = 2
    time_dim: int = 1
    hidden_dim: int = 128
    n_layers: int = 3
    max_drift: Optional[float] = 2.0
    pk: PKConfig = field(default_factory=PKConfig)

    @classmethod
    def from_saved_config(cls, values):
        """Restore a drift configuration with its nested PK settings."""
        values = dict(values)
        values["pk"] = PKConfig(**values["pk"])
        return cls(**values)


def antibody_concentration_at_day(day, dose_days, dose_amount=1.0, delta=0.2):
    """
    First-order decay concentration from bolus doses.

    Concentration is sum_k dose_amount * exp(-delta * (day - dose_day_k)) for
    doses already administered at the queried day.
    """
    value = 0.0
    for dose_day in dose_days:
        if float(day) >= float(dose_day):
            value += float(dose_amount) * np.exp(-float(delta) * (float(day) - float(dose_day)))
    return float(value)


def treatment_values_at_day(P, B, day, config=None):
    """
    Convert treatment-arm indicators P/B into time-dependent treatment values.

    If PK is disabled, returns binary/raw P and B. If PK is enabled, P/B gate
    antibody concentration curves defined by dose days, dose amounts, and decay
    rates in `PKConfig`.
    """
    config = config or PKConfig()
    if not config.enabled:
        return float(P), float(B)

    P_value = float(P) * antibody_concentration_at_day(
        day,
        config.pd1_dose_days,
        dose_amount=config.pd1_dose_amount,
        delta=config.delta_pd1,
    )
    B_value = float(B) * antibody_concentration_at_day(
        day,
        config.b41bb_dose_days,
        dose_amount=config.b41bb_dose_amount,
        delta=config.delta_b41bb,
    )
    if config.normalize_to_unit_peak:
        P_norm = max(float(config.pd1_dose_amount), 1e-8)
        B_norm = max(float(config.b41bb_dose_amount), 1e-8)
        P_value /= P_norm
        B_value /= B_norm
    return P_value, B_value


class ReactionEffectTransform(nn.Module):
    """Mechanistic reaction-term effects returning [G(P,B), F(B)]."""

    def __init__(self, config=None):
        super().__init__()
        self.config = config or ReactionEffectConfig()

        if self.config.learnable:
            self.f_b_coeff = nn.Parameter(torch.tensor(float(self.config.f_b_coeff)))
            self.g_b_coeff = nn.Parameter(torch.tensor(float(self.config.g_b_coeff)))
            self.g_pd1_coeff = nn.Parameter(torch.tensor(float(self.config.g_pd1_coeff)))
            self.pd1_k_raw = nn.Parameter(torch.tensor(float(self.config.pd1_k)))
            self.b41bb_k_raw = nn.Parameter(torch.tensor(float(self.config.b41bb_k)))
        else:
            self.register_buffer("f_b_coeff", torch.tensor(float(self.config.f_b_coeff)))
            self.register_buffer("g_b_coeff", torch.tensor(float(self.config.g_b_coeff)))
            self.register_buffer("g_pd1_coeff", torch.tensor(float(self.config.g_pd1_coeff)))
            self.register_buffer("pd1_k_raw", torch.tensor(float(self.config.pd1_k)))
            self.register_buffer("b41bb_k_raw", torch.tensor(float(self.config.b41bb_k)))

    def forward(self, treatment):
        if treatment.ndim != 2 or treatment.shape[1] != 2:
            raise ValueError("treatment must have shape [n, 2] with columns [P, B].")
        P = treatment[:, 0:1]
        B = treatment[:, 1:2]
        return torch.cat([self.g_effect(P, B), self.f_effect(B)], dim=1)

    def _eps(self, dtype, device):
        return torch.tensor(float(self.config.eps), device=device, dtype=dtype)

    def _positive_k(self, raw, dtype, device):
        return F.softplus(raw).to(dtype=dtype, device=device) + self._eps(dtype, device)

    def f_effect(self, B):
        eps = self._eps(B.dtype, B.device)
        K_B = self._positive_k(self.b41bb_k_raw, B.dtype, B.device)
        coeff_b = self.f_b_coeff.to(dtype=B.dtype, device=B.device)
        return 1.0 + coeff_b * B / (K_B + B + eps)

    def g_effect(self, P, B):
        eps_p = self._eps(P.dtype, P.device)
        eps_b = self._eps(B.dtype, B.device)
        K_P = self._positive_k(self.pd1_k_raw, P.dtype, P.device)
        K_B = self._positive_k(self.b41bb_k_raw, B.dtype, B.device)
        coeff_p = self.g_pd1_coeff.to(dtype=P.dtype, device=P.device)
        coeff_b = self.g_b_coeff.to(dtype=B.dtype, device=B.device)
        p_part = coeff_p * P / (K_P + P + eps_p)
        b_part = coeff_b * B / (K_B + B + eps_b)
        return 1.0 + p_part + b_part


class CD8DriftNet(nn.Module):
    """
    Neural drift field for CD8 program-score dynamics.

    Forward signature:
        drift = model(z, treatment, time)

    where `z` has shape [n, z_dim], `treatment` has columns [P, B],
    and `time` is normalized time with shape [n, 1].
    """

    def __init__(self, config=None):
        super().__init__()
        self.config = config or DriftModelConfig()

        in_dim = self.config.z_dim + self.config.treatment_dim + self.config.time_dim
        layers = []
        dim = in_dim
        for _ in range(self.config.n_layers):
            layers.append(nn.Linear(dim, self.config.hidden_dim))
            layers.append(nn.SiLU())
            dim = self.config.hidden_dim
        layers.append(nn.Linear(dim, self.config.z_dim))
        self.net = nn.Sequential(*layers)

        final_layer = self.net[-1]
        nn.init.zeros_(final_layer.weight)
        nn.init.zeros_(final_layer.bias)

    def forward(self, z, treatment, time):
        x = torch.cat([z, treatment, time], dim=1)
        drift = self.net(x)
        if self.config.max_drift is not None:
            drift = self.config.max_drift * torch.tanh(drift / self.config.max_drift)
        return drift


class DriftIntegrator(nn.Module):
    """Explicit Euler integrator for dz/dt = v_theta(z, P, B, t)."""

    def __init__(self, model, n_steps_per_day=10, time_scale=15.0, pk_config=None):
        super().__init__()
        self.model = model
        self.n_steps_per_day = int(n_steps_per_day)
        self.time_scale = float(time_scale)
        self.pk_config = pk_config or PKConfig()

    def forward(self, z0, P, B, day0, dt):
        if float(dt) <= 0:
            raise ValueError(f"dt must be positive, got {dt}")

        n_steps = max(1, int(round(float(dt) * self.n_steps_per_day)))
        h = float(dt) / n_steps
        z = z0

        for step in range(n_steps):
            t_value = float(day0) + step * h
            z = z + h * self._velocity(z, P, B, t_value)

        return z

    def _velocity(self, z, P, B, day):
        P, B = treatment_values_at_day(P, B, day, self.pk_config)
        treatment = make_treatment_tensor(z, P, B)
        time = torch.full(
            (z.shape[0], 1),
            float(day) / self.time_scale,
            device=z.device,
            dtype=z.dtype,
        )
        return self.model(z, treatment, time)


def make_treatment_tensor(z, P, B):
    """Create an [n, 2] treatment tensor matching z device/dtype."""
    P_col = torch.full((z.shape[0], 1), float(P), device=z.device, dtype=z.dtype)
    B_col = torch.full((z.shape[0], 1), float(B), device=z.device, dtype=z.dtype)
    return torch.cat([P_col, B_col], dim=1)
