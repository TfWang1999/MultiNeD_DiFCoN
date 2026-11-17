from __future__ import annotations

from collections.abc import Iterator
from dataclasses import asdict, dataclass
import copy
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch

from .config import BUDGET_ID, TOTAL_BUDGET
from .losses import components_to_float, compute_direct_loss, normalized_auc
from .rollout import DirectRolloutConfig, reset_rollout_rng, rollout_patient


SCORE_NAMES = ["precursor", "activation", "eff_early", "eff_cytox", "dysfunction"]


@dataclass
class DirectTrainConfig:
    """Settings for end-to-end training on a large cohort."""

    n_epochs: int = 100
    batch_size: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    eval_every: int = 5
    eval_patients: int = 8
    lr_reduction_factor: float = 0.5
    lr_patience: int = 4
    lr_threshold: float = 1e-4
    min_learning_rate: float = 1e-6
    random_state: int = 2026

    def to_dict(self) -> dict:
        return asdict(self)


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _mean_component_dict(component_rows: list[dict[str, torch.Tensor]]) -> dict[str, float]:
    if not component_rows:
        return {}
    keys = component_rows[0]
    return {
        key: float(torch.stack([row[key].detach() for row in component_rows]).mean().cpu())
        for key in keys
    }


@torch.no_grad()
def mean_policy_loss(model, policy, specs, rollout_config, loss_config, max_patients: int | None = None) -> float:
    selected = list(specs) if max_patients is None else list(specs)[: int(max_patients)]
    losses = []
    policy.eval()
    for spec in selected:
        reset_rollout_rng(int(spec["rollout_seed"]))
        trajectory = rollout_patient(model, policy, spec, rollout_config)
        loss, _ = compute_direct_loss(trajectory, loss_config)
        losses.append(loss)
    return float(torch.stack(losses).mean().cpu()) if losses else float("nan")


def train_direct_policy(
    reaction_model,
    policy,
    train_specs: list[dict],
    validation_specs: list[dict],
    rollout_config: DirectRolloutConfig,
    loss_config,
    train_config: DirectTrainConfig,
) -> pd.DataFrame:
    """Backpropagate through full treatment trajectories for random minibatches of virtual patients."""

    if not train_specs:
        raise ValueError("train_specs is empty.")
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=float(train_config.learning_rate),
        weight_decay=float(train_config.weight_decay),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(train_config.lr_reduction_factor),
        patience=int(train_config.lr_patience),
        threshold=float(train_config.lr_threshold),
        min_lr=float(train_config.min_learning_rate),
    )
    rng = np.random.default_rng(int(train_config.random_state))
    history_rows: list[dict] = []
    best_validation_loss = float("inf")
    best_epoch = 0
    best_policy_state = None

    for epoch in range(1, int(train_config.n_epochs) + 1):
        policy.train()
        order = rng.permutation(len(train_specs))
        batch_objectives = []
        epoch_components: list[dict[str, torch.Tensor]] = []
        grad_norms = []

        for start in range(0, len(order), int(train_config.batch_size)):
            batch_indices = order[start : start + int(train_config.batch_size)]
            patient_losses = []
            batch_components = []
            optimizer.zero_grad(set_to_none=True)
            for patient_index in batch_indices:
                spec = train_specs[int(patient_index)]
                reset_rollout_rng(int(spec["rollout_seed"]) + epoch * 100_003)
                trajectory = rollout_patient(reaction_model, policy, spec, rollout_config)
                loss, components = compute_direct_loss(trajectory, loss_config)
                patient_losses.append(loss)
                batch_components.append(components)

            patient_loss_tensor = torch.stack(patient_losses)
            objective = patient_loss_tensor.mean()
            if not torch.isfinite(objective):
                raise FloatingPointError(f"Non-finite training objective at epoch {epoch}.")
            objective.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), float(train_config.grad_clip))
            optimizer.step()
            batch_objectives.append(float(objective.detach().cpu()))
            epoch_components.extend(batch_components)
            grad_norms.append(float(grad_norm.detach().cpu()))

        should_eval = epoch == 1 or epoch == int(train_config.n_epochs) or epoch % int(train_config.eval_every) == 0
        validation_loss = (
            mean_policy_loss(
                reaction_model,
                policy,
                validation_specs,
                rollout_config,
                loss_config,
                max_patients=train_config.eval_patients,
            )
            if should_eval and validation_specs
            else np.nan
        )
        is_best_validation = False
        if should_eval and np.isfinite(validation_loss):
            if validation_loss < best_validation_loss:
                best_validation_loss = float(validation_loss)
                best_epoch = int(epoch)
                best_policy_state = copy.deepcopy(policy.state_dict())
                is_best_validation = True
            scheduler.step(float(validation_loss))
        current_lr = float(optimizer.param_groups[0]["lr"])
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(batch_objectives)),
            "validation_loss": validation_loss,
            "grad_norm": float(np.mean(grad_norms)),
            "learning_rate": current_lr,
            "is_best_validation": is_best_validation,
            "best_validation_loss_so_far": (
                best_validation_loss if np.isfinite(best_validation_loss) else np.nan
            ),
            "best_epoch_so_far": best_epoch if best_epoch > 0 else np.nan,
            **{f"train_{key}": value for key, value in _mean_component_dict(epoch_components).items()},
        }
        history_rows.append(row)
        if should_eval:
            print(
                f"epoch={epoch:04d} train={row['train_loss']:.5f} "
                f"validation={row['validation_loss']:.5f} grad={row['grad_norm']:.4f} "
                f"lr={row['learning_rate']:.2e}"
            )
    if best_policy_state is not None:
        policy.load_state_dict(best_policy_state)
        print(f"restored best policy: epoch={best_epoch}, validation={best_validation_loss:.5f}")
    return pd.DataFrame(history_rows)


def _first_action_day(actions: np.ndarray, days: np.ndarray, column: int, threshold: float = 0.05) -> float:
    indices = np.flatnonzero(actions[:, column] >= float(threshold))
    return float(days[indices[0]]) if len(indices) else np.nan


def _order_class(first_pd1: float, first_b41bb: float) -> str:
    if np.isnan(first_pd1) and np.isnan(first_b41bb):
        return "none"
    if np.isnan(first_b41bb):
        return "PD1-only"
    if np.isnan(first_pd1):
        return "4-1BB-only"
    if first_pd1 < first_b41bb:
        return "PD1-first"
    if first_b41bb < first_pd1:
        return "4-1BB-first"
    return "same-day"


def summarize_trajectory(
    spec: dict,
    strategy: str,
    trajectory: dict,
    total_loss,
    components,
) -> dict:
    time = trajectory["time"].detach().cpu().numpy().astype(float)
    C = trajectory["volume"].detach().cpu().numpy().astype(float)
    N = trajectory["N"].detach().cpu().numpy().astype(float)
    actions = trajectory["actions"].detach().cpu().numpy().astype(float)
    scores = trajectory["scores"].detach().cpu().numpy().astype(float)
    days = time[:-1]
    C_rel = C / max(C[0], 1e-8)
    N_rel = N / max(N[0], 1e-8)
    first_pd1 = _first_action_day(actions, days, 0)
    first_b41bb = _first_action_day(actions, days, 1)
    row = {
        "patient_id": spec["patient_id"],
        "cohort": spec["cohort"],
        "strategy": strategy,
        "budget_id": BUDGET_ID,
        "budget_value": TOTAL_BUDGET,
        "loss": float(total_loss.detach().cpu()),
        "final_C_rel": float(C_rel[-1]),
        "tumor_regression": float(1.0 - C_rel[-1]),
        "tumor_auc_rel": float(normalized_auc(torch.as_tensor(C_rel), torch.as_tensor(time))),
        "final_N_rel": float(N_rel[-1]),
        "mean_activity": float(trajectory["activity"].mean().detach().cpu()),
        "mean_A_CD8": float(trajectory["A_CD8"].mean().detach().cpu()),
        "total_PD1_dose": float(actions[:, 0].sum()),
        "total_B41BB_dose": float(actions[:, 1].sum()),
        "total_dose": float(actions.sum()),
        "first_PD1_day": first_pd1,
        "first_B41BB_day": first_b41bb,
        "order_class": _order_class(first_pd1, first_b41bb),
    }
    for idx, score_name in enumerate(SCORE_NAMES):
        row[f"final_score_{score_name}"] = float(scores[-1, idx])
    row.update({f"loss_{key}": value for key, value in components_to_float(components).items()})
    return row


def trajectory_frame(spec: dict, strategy: str, trajectory: dict) -> pd.DataFrame:
    time = trajectory["time"].detach().cpu().numpy().astype(float)
    C = trajectory["volume"].detach().cpu().numpy().astype(float)
    N = trajectory["N"].detach().cpu().numpy().astype(float)
    scores = trajectory["scores"].detach().cpu().numpy().astype(float)
    frame = pd.DataFrame(
        {
            "patient_id": spec["patient_id"],
            "cohort": spec["cohort"],
            "strategy": strategy,
            "budget_id": BUDGET_ID,
            "budget_value": TOTAL_BUDGET,
            "day": time,
            "C": C,
            "C_rel": C / max(C[0], 1e-8),
            "N": N,
            "N_rel": N / max(N[0], 1e-8),
            "activity": trajectory["activity"].detach().cpu().numpy(),
            "A_CD8": trajectory["A_CD8"].detach().cpu().numpy(),
            "P_conc": trajectory["P_conc"].detach().cpu().numpy(),
            "B_conc": trajectory["B_conc"].detach().cpu().numpy(),
            "remaining_budget": trajectory["remaining_budget"].detach().cpu().numpy(),
        }
    )
    for idx, score_name in enumerate(SCORE_NAMES):
        frame[f"score_{score_name}"] = scores[:, idx]
    return frame


def action_frame(spec: dict, strategy: str, trajectory: dict) -> pd.DataFrame:
    actions = trajectory["actions"].detach().cpu().numpy().astype(float)
    days = trajectory["time"].detach().cpu().numpy().astype(float)[:-1]
    daily_total = actions.sum(axis=1)
    pd1_fraction = np.divide(
        actions[:, 0],
        daily_total,
        out=np.full_like(daily_total, np.nan),
        where=daily_total > 1e-8,
    )
    return pd.DataFrame(
        {
            "patient_id": spec["patient_id"],
            "cohort": spec["cohort"],
            "strategy": strategy,
            "budget_id": BUDGET_ID,
            "budget_value": TOTAL_BUDGET,
            "day": days,
            "action_PD1": actions[:, 0],
            "action_B41BB": actions[:, 1],
            "daily_total_dose": daily_total,
            "pd1_fraction": pd1_fraction,
        }
    )


@torch.no_grad()
def evaluation_rollout(reaction_model, policy, spec, rollout_config) -> dict[str, torch.Tensor]:
    policy.eval()
    reset_rollout_rng(int(spec["rollout_seed"]))
    return rollout_patient(reaction_model, policy, spec, rollout_config)


@torch.no_grad()
def evaluate_patient(reaction_model, policy, spec, rollout_config, loss_config, strategy: str) -> tuple[dict, dict]:
    trajectory = evaluation_rollout(reaction_model, policy, spec, rollout_config)
    total_loss, components = compute_direct_loss(trajectory, loss_config)
    return summarize_trajectory(spec, strategy, trajectory, total_loss, components), trajectory


def evaluate_strategies(
    reaction_model,
    strategies: dict[str, torch.nn.Module],
    specs: list[dict],
    rollout_config,
    loss_config,
) -> Iterator[tuple[dict, dict]]:
    """Yield metrics and trajectories in strategy and patient order for callers to assemble tables."""

    for strategy_name, policy in strategies.items():
        for spec in specs:
            yield evaluate_patient(reaction_model, policy, spec, rollout_config, loss_config, strategy_name)


def save_policy_checkpoint(
    path: str | Path,
    policy,
    policy_config,
    rollout_config,
    loss_config,
    train_config,
    model_identity: dict,
    history_df: pd.DataFrame,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "policy_type": policy.policy_type,
            "policy_state_dict": policy.state_dict(),
            "policy_config": policy_config.to_dict(),
            "rollout_config": asdict(rollout_config),
            "loss_config": loss_config.to_dict(),
            "train_config": train_config.to_dict(),
            "model_identity": dict(model_identity),
            "training_history": history_df.to_dict(orient="records"),
        },
        path,
    )
    return path


__all__ = [
    "DirectTrainConfig",
    "SCORE_NAMES",
    "evaluate_strategies",
    "mean_policy_loss",
    "save_policy_checkpoint",
    "seed_everything",
    "train_direct_policy",
]
