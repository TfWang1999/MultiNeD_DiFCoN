from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn


from core.config import DAILY_TOTAL_DOSE_CAP, TOTAL_BUDGET
from core.rollout import DirectRolloutConfig, apply_daily_total_dose_cap


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    config = json.loads(path.read_text(encoding="utf-8"))
    config["config_source"] = str(path)
    return config


def n_decisions(config: DirectRolloutConfig) -> int:
    return int(math.ceil((float(config.end_day) - float(config.start_day)) / float(config.decision_dt)))


def project_schedule_numpy(raw_actions: np.ndarray) -> np.ndarray:
    actions = np.asarray(raw_actions, dtype=float).reshape(-1, 2).copy()
    np.clip(actions, 0.0, 1.0, out=actions)
    totals = actions.sum(axis=1, keepdims=True)
    actions *= np.minimum(1.0, DAILY_TOTAL_DOSE_CAP / np.maximum(totals, 1e-12))
    total = float(actions.sum())
    if total > TOTAL_BUDGET:
        actions *= TOTAL_BUDGET / max(total, 1e-12)
    return actions.astype(np.float32)


class DifferentiableSchedulePolicy(nn.Module):
    """Open-loop schedule with differentiable hard budget and daily-cap projection."""

    def __init__(
        self,
        n_steps: int,
        initial_dose: float = 0.0625,
        prefix: np.ndarray | None = None,
        active_start: int = 0,
        active_stop: int | None = None,
    ):
        super().__init__()
        probability = np.clip(float(initial_dose), 1e-5, 1.0 - 1e-5)
        initial = math.log(probability / (1.0 - probability))
        self.raw_actions = nn.Parameter(torch.full((int(n_steps), 2), float(initial)))
        prefix_array = np.zeros((int(n_steps), 2), dtype=np.float32)
        if prefix is not None:
            prefix = np.asarray(prefix, dtype=np.float32).reshape(-1, 2)
            prefix_array[: prefix.shape[0]] = prefix
        self.register_buffer("fixed_prefix", torch.as_tensor(prefix_array))
        self.active_start = int(active_start)
        self.active_stop = int(active_stop if active_stop is not None else n_steps)

    def raw_action(self, index: int) -> torch.Tensor:
        if index < self.active_start:
            return self.fixed_prefix[index]
        if index >= self.active_stop:
            return self.raw_actions.new_zeros(2)
        return torch.sigmoid(self.raw_actions[index])

    def act(self, history: torch.Tensor, step_index: int, remaining_budget: torch.Tensor) -> torch.Tensor:
        index = min(max(int(step_index), 0), self.raw_actions.shape[0] - 1)
        action = self.raw_action(index).to(device=history.device, dtype=history.dtype)
        action = apply_daily_total_dose_cap(action)
        scale = torch.clamp(remaining_budget.reshape(()) / (action.sum() + 1e-8), max=1.0)
        return action * scale

    @torch.no_grad()
    def schedule(self, device: torch.device | None = None) -> np.ndarray:
        device = device or self.raw_actions.device
        remaining = torch.tensor(TOTAL_BUDGET, device=device)
        values = []
        dummy = torch.zeros((1, 1, 1), device=device)
        for index in range(self.raw_actions.shape[0]):
            action = self.act(dummy, index, remaining)
            values.append(action)
            remaining = torch.clamp(remaining - action.sum(), min=0.0)
        return torch.stack(values).detach().cpu().numpy().astype(np.float32)


class ArraySchedulePolicy(nn.Module):
    def __init__(self, schedule: np.ndarray):
        super().__init__()
        self.register_buffer("actions", torch.as_tensor(schedule, dtype=torch.float32))

    def act(self, history: torch.Tensor, step_index: int, remaining_budget: torch.Tensor) -> torch.Tensor:
        index = min(max(int(step_index), 0), self.actions.shape[0] - 1)
        action = self.actions[index].to(history.device, history.dtype)
        scale = torch.clamp(remaining_budget.reshape(()) / (action.sum() + 1e-8), max=1.0)
        return action * scale


__all__ = [
    "ArraySchedulePolicy",
    "DifferentiableSchedulePolicy",
    "load_config",
    "n_decisions",
    "project_schedule_numpy",
]
