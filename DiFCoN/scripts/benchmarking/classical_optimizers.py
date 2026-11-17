from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch

from core.losses import compute_direct_loss
from core.rollout import reset_rollout_rng, rollout_patient

from .policy_benchmark_common import (
    ArraySchedulePolicy,
    DifferentiableSchedulePolicy,
    n_decisions,
    project_schedule_numpy,
)


@dataclass
class OptimizerResult:
    policy: torch.nn.Module
    schedule: np.ndarray
    history: pd.DataFrame
    accounting: dict[str, Any]


def gradient_open_loop(
    reaction_model,
    spec: dict[str, Any],
    rollout_config,
    loss_config,
    settings: dict[str, Any],
    seed: int,
) -> OptimizerResult:
    started = time.perf_counter()
    policy = DifferentiableSchedulePolicy(
        n_decisions(rollout_config),
        initial_dose=float(settings.get("initial_dose", 0.0625)),
    ).to(next(reaction_model.parameters()).device)
    optimizer = torch.optim.AdamW(
        policy.parameters(), lr=float(settings["learning_rate"]),
        weight_decay=float(settings.get("weight_decay", 0.0)),
    )
    history = []
    for iteration in range(1, int(settings["iterations"]) + 1):
        reset_rollout_rng(int(seed))
        trajectory = rollout_patient(reaction_model, policy, spec, rollout_config)
        loss, components = compute_direct_loss(trajectory, loss_config)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), float(settings.get("grad_clip", 5.0)))
        optimizer.step()
        history.append({
            "iteration": iteration,
            "loss": float(loss.detach().cpu()),
            "final_tumor": float(components["final_tumor_loss"].detach().cpu()),
            "tumor_auc": float(components["tumor_auc_loss"].detach().cpu()),
            "total_dose": float(components["total_dose"].detach().cpu()),
        })
    schedule = policy.schedule()
    fixed = ArraySchedulePolicy(schedule).to(next(reaction_model.parameters()).device)
    return OptimizerResult(
        policy=fixed,
        schedule=schedule,
        history=pd.DataFrame(history),
        accounting={
            "method": "gradient_open_loop",
            "wall_seconds": float(time.perf_counter() - started),
            "forward_evaluations": int(settings["iterations"]),
            "gradient_evaluations": int(settings["iterations"]),
        },
    )


def rolling_horizon_mpc(
    reaction_model,
    spec: dict[str, Any],
    rollout_config,
    loss_config,
    settings: dict[str, Any],
    seed: int,
) -> OptimizerResult:
    """Re-solve a differentiable finite-horizon schedule after every decision.

    The already delivered prefix is fixed.  The deterministic simulator is
    rerun from baseline, which is equivalent to restarting from the current
    simulated state while retaining a simple, auditable implementation.
    """
    started = time.perf_counter()
    steps = n_decisions(rollout_config)
    horizon = int(settings["horizon_steps"])
    prefix = np.zeros((0, 2), dtype=np.float32)
    history_rows = []
    total_forward = 0
    for decision in range(steps):
        active_stop = min(steps, decision + horizon)
        candidate = DifferentiableSchedulePolicy(
            steps,
            initial_dose=float(settings.get("initial_dose", 0.0625)),
            prefix=prefix,
            active_start=decision,
            active_stop=active_stop,
        ).to(next(reaction_model.parameters()).device)
        optimizer = torch.optim.Adam(
            [candidate.raw_actions], lr=float(settings["learning_rate"])
        )
        for iteration in range(1, int(settings["iterations_per_decision"]) + 1):
            reset_rollout_rng(int(seed))
            trajectory = rollout_patient(reaction_model, candidate, spec, rollout_config)
            loss, _ = compute_direct_loss(trajectory, loss_config)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([candidate.raw_actions], float(settings.get("grad_clip", 5.0)))
            optimizer.step()
            total_forward += 1
            history_rows.append({
                "decision_index": decision,
                "iteration": iteration,
                "loss": float(loss.detach().cpu()),
                "active_stop": active_stop,
            })
        selected = candidate.schedule()[decision]
        prefix = np.vstack([prefix, selected])
    schedule = project_schedule_numpy(prefix)
    fixed = ArraySchedulePolicy(schedule).to(next(reaction_model.parameters()).device)
    return OptimizerResult(
        policy=fixed,
        schedule=schedule,
        history=pd.DataFrame(history_rows),
        accounting={
            "method": "rolling_horizon_mpc",
            "wall_seconds": float(time.perf_counter() - started),
            "forward_evaluations": total_forward,
            "gradient_evaluations": total_forward,
            "horizon_steps": horizon,
        },
    )


class DiagonalCMAES:
    """Dependency-free separable CMA-ES for bounded schedule logits."""

    def __init__(self, dimension: int, population: int, sigma: float, seed: int):
        self.dimension = int(dimension)
        self.population = int(population)
        self.mu = max(1, self.population // 2)
        raw_weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights = raw_weights / raw_weights.sum()
        self.mu_eff = 1.0 / np.sum(self.weights ** 2)
        self.mean = np.full(self.dimension, math.log(0.0625 / (1.0 - 0.0625)), dtype=float)
        self.sigma = float(sigma)
        self.diag_c = np.ones(self.dimension, dtype=float)
        self.path_sigma = np.zeros(self.dimension, dtype=float)
        self.path_c = np.zeros(self.dimension, dtype=float)
        n = self.dimension
        self.c_sigma = (self.mu_eff + 2.0) / (n + self.mu_eff + 5.0)
        self.d_sigma = 1.0 + 2.0 * max(0.0, math.sqrt((self.mu_eff - 1.0) / (n + 1.0)) - 1.0) + self.c_sigma
        self.c_c = (4.0 + self.mu_eff / n) / (n + 4.0 + 2.0 * self.mu_eff / n)
        self.c1 = 2.0 / ((n + 1.3) ** 2 + self.mu_eff)
        self.c_mu = min(1.0 - self.c1, 2.0 * (self.mu_eff - 2.0 + 1.0 / self.mu_eff) / ((n + 2.0) ** 2 + self.mu_eff))
        self.expected_norm = math.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n * n))
        self.rng = np.random.default_rng(int(seed))

    def ask(self):
        z = self.rng.normal(size=(self.population, self.dimension))
        x = self.mean + self.sigma * z * np.sqrt(self.diag_c)
        return x, z

    def tell(self, candidates: np.ndarray, z: np.ndarray, values: np.ndarray):
        order = np.argsort(values)
        selected = candidates[order[: self.mu]]
        selected_z = z[order[: self.mu]]
        old_mean = self.mean.copy()
        self.mean = np.sum(selected * self.weights[:, None], axis=0)
        weighted_z = np.sum(selected_z * self.weights[:, None], axis=0)
        self.path_sigma = (1.0 - self.c_sigma) * self.path_sigma + math.sqrt(
            self.c_sigma * (2.0 - self.c_sigma) * self.mu_eff
        ) * weighted_z
        norm = np.linalg.norm(self.path_sigma)
        self.sigma *= math.exp((self.c_sigma / self.d_sigma) * (norm / self.expected_norm - 1.0))
        normalized_step = (self.mean - old_mean) / max(self.sigma, 1e-12)
        self.path_c = (1.0 - self.c_c) * self.path_c + math.sqrt(
            self.c_c * (2.0 - self.c_c) * self.mu_eff
        ) * normalized_step
        rank_mu = np.sum(self.weights[:, None] * (selected_z ** 2), axis=0)
        self.diag_c = (
            (1.0 - self.c1 - self.c_mu) * self.diag_c
            + self.c1 * (self.path_c ** 2)
            + self.c_mu * rank_mu
        )
        self.diag_c = np.clip(self.diag_c, 1e-8, 1e8)


def diagonal_cmaes(
    reaction_model,
    spec: dict[str, Any],
    rollout_config,
    loss_config,
    settings: dict[str, Any],
    seed: int,
) -> OptimizerResult:
    started = time.perf_counter()
    steps = n_decisions(rollout_config)
    optimizer = DiagonalCMAES(
        dimension=steps * 2,
        population=int(settings["population"]),
        sigma=float(settings["sigma"]),
        seed=seed,
    )
    best_value = float("inf")
    best_schedule = None
    history = []
    forward = 0
    for generation in range(1, int(settings["generations"]) + 1):
        candidates, noise = optimizer.ask()
        values = []
        for candidate_index, candidate in enumerate(candidates):
            action = 1.0 / (1.0 + np.exp(-np.clip(candidate, -30.0, 30.0)))
            schedule = project_schedule_numpy(action.reshape(steps, 2))
            policy = ArraySchedulePolicy(schedule).to(next(reaction_model.parameters()).device)
            reset_rollout_rng(int(seed))
            with torch.no_grad():
                trajectory = rollout_patient(reaction_model, policy, spec, rollout_config)
                loss, _ = compute_direct_loss(trajectory, loss_config)
            value = float(loss.detach().cpu())
            values.append(value)
            forward += 1
            if value < best_value:
                best_value = value
                best_schedule = schedule.copy()
        optimizer.tell(candidates, noise, np.asarray(values))
        history.append({
            "generation": generation,
            "best_loss": float(np.min(values)),
            "mean_loss": float(np.mean(values)),
            "global_best_loss": best_value,
            "sigma": optimizer.sigma,
        })
    if best_schedule is None:
        raise RuntimeError("Diagonal CMA-ES produced no schedule")
    fixed = ArraySchedulePolicy(best_schedule).to(next(reaction_model.parameters()).device)
    return OptimizerResult(
        policy=fixed,
        schedule=best_schedule,
        history=pd.DataFrame(history),
        accounting={
            "method": "diagonal_cmaes",
            "wall_seconds": float(time.perf_counter() - started),
            "forward_evaluations": forward,
            "gradient_evaluations": 0,
            "population": int(settings["population"]),
            "generations": int(settings["generations"]),
        },
    )


OPTIMIZERS: dict[str, Callable[..., OptimizerResult]] = {
    "gradient_open_loop": gradient_open_loop,
    "rolling_horizon_mpc": rolling_horizon_mpc,
    "diagonal_cmaes": diagonal_cmaes,
}


__all__ = [
    "DiagonalCMAES",
    "OPTIMIZERS",
    "OptimizerResult",
    "diagonal_cmaes",
    "gradient_open_loop",
    "rolling_horizon_mpc",
]
