"""Shared direct-dose policy utilities and fixed treatment controls."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


from .config import TOTAL_BUDGET


class ConvHistoryBlock(nn.Module):
    """Residual convolutional block for a short state-history window."""

    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
        )
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.net(x))


class FixedSchedulePolicy(nn.Module):
    """Fixed anti-PD1/anti-4-1BB schedule used as a pharmacology control."""

    def __init__(self, actions: np.ndarray | torch.Tensor):
        super().__init__()
        actions = torch.as_tensor(actions, dtype=torch.float32)
        if actions.ndim != 2 or actions.shape[1] != 2:
            raise ValueError("actions must have shape [n_decisions, 2].")
        self.register_buffer("actions", actions)

    def act(self, history: torch.Tensor, step_index: int, remaining_budget: torch.Tensor) -> torch.Tensor:
        index = min(max(int(step_index), 0), int(self.actions.shape[0]) - 1)
        raw = self.actions[index].to(device=history.device, dtype=history.dtype)
        remaining = torch.clamp(remaining_budget.reshape(()), min=0.0)
        scale = torch.clamp(remaining / (raw.sum() + 1e-8), max=1.0)
        return raw * scale


def make_initial_history(feature: torch.Tensor, history_length: int) -> torch.Tensor:
    if feature.ndim == 1:
        feature = feature.unsqueeze(0)
    if feature.ndim != 2:
        raise ValueError("feature must be [feature] or [batch, feature].")
    return feature.unsqueeze(1).repeat(1, int(history_length), 1)


def append_history(history: torch.Tensor, feature: torch.Tensor) -> torch.Tensor:
    if feature.ndim == 1:
        feature = feature.unsqueeze(0)
    if history.ndim != 3 or feature.ndim != 2:
        raise ValueError("history must be [batch, length, feature] and feature must be [batch, feature].")
    return torch.cat([history[:, 1:, :], feature.unsqueeze(1)], dim=1)


def normalize_actions_to_budget(actions: np.ndarray, total_budget: float) -> np.ndarray:
    actions = np.clip(np.asarray(actions, dtype=np.float32), 0.0, 1.0)
    total = float(actions.sum())
    if total > float(total_budget):
        actions = actions * (float(total_budget) / max(total, 1e-8))
    return actions


def make_fixed_schedule(
    decision_days: np.ndarray,
    pd1_days: tuple[float, ...] = (),
    b41bb_days: tuple[float, ...] = (),
    pd1_dose: float = 1.0,
    b41bb_dose: float = 1.0,
    total_budget: float | None = None,
) -> np.ndarray:
    """Convert dosing days to an action table and optionally project to budget."""

    decision_days = np.asarray(decision_days, dtype=float)
    actions = np.zeros((len(decision_days), 2), dtype=np.float32)
    for day in pd1_days:
        index = np.flatnonzero(np.isclose(decision_days, float(day)))
        if len(index):
            actions[index[0], 0] = float(pd1_dose)
    for day in b41bb_days:
        index = np.flatnonzero(np.isclose(decision_days, float(day)))
        if len(index):
            actions[index[0], 1] = float(b41bb_dose)
    if total_budget is not None:
        actions = normalize_actions_to_budget(actions, float(total_budget))
    return actions


def fixed_regimen_policies(decision_days, labels: dict[str, str], device) -> dict[str, FixedSchedulePolicy]:
    """Build the requested paper controls in evaluation order."""
    pd1_days = (1.0, 4.0)
    b41bb_days = (2.0,)
    regimens = {
        "no_treatment": {},
        "anti_pd1": {"pd1_days": pd1_days},
        "anti_b41bb": {"b41bb_days": b41bb_days},
        "combine": {"pd1_days": pd1_days, "b41bb_days": b41bb_days},
    }
    return {
        label: FixedSchedulePolicy(
            make_fixed_schedule(decision_days, total_budget=TOTAL_BUDGET, **regimens[regimen])
        ).to(device)
        for regimen, label in labels.items()
    }


__all__ = [
    "ConvHistoryBlock",
    "FixedSchedulePolicy",
    "append_history",
    "fixed_regimen_policies",
    "make_fixed_schedule",
    "make_initial_history",
    "normalize_actions_to_budget",
]
