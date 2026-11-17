from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import torch
import pandas as pd

from common.cd8_zdist_data import mouse_balanced_sample, sample_transition
from training.drift_training import (
    build_rollout_arms,
    build_drift_model,
    build_lr_scheduler,
    get_current_lr,
    load_drift_checkpoint,
    save_drift_checkpoint,
)
from experiments.volume_loto.model_runner import _apply_overrides

from .dynamics_benchmark_common import PROJECT_ROOT, project_path, set_seed


def symmetric_nearest_neighbor_mse(predicted, target):
    """Symmetric Chamfer/nearest-neighbour MSE for unpaired cell snapshots."""
    distances = (predicted[:, None, :] - target[None, :, :]).pow(2).sum(dim=2)
    return 0.5 * (distances.min(dim=1).values.mean() + distances.min(dim=0).values.mean())


def _nearest_neighbor_rollout_loss(
    integrator,
    group_data: dict,
    arm: dict[str, Any],
    train_config,
    device,
    rng: np.random.Generator,
):
    groups = sorted(arm["groups"], key=lambda value: (float(value["day"]), value["group_id"]))
    anchor_group = "day1__Control"
    anchor_day = float(group_data[anchor_group]["day"])
    targets = [value for value in groups if float(value["day"]) > anchor_day]
    if not targets:
        raise ValueError(f"No rollout target for arm {arm['arm_id']}")
    n_max = int(train_config.rollout_n_max or train_config.n_max)
    source, _ = mouse_balanced_sample(
        group_data,
        anchor_group,
        n_max=n_max,
        random_state=int(rng.integers(0, 2**31 - 1)),
    )
    source_t = torch.as_tensor(source, dtype=torch.float32, device=device)
    losses, drift_penalties = [], []
    for target in targets:
        observed, _ = mouse_balanced_sample(
            group_data,
            target["group_id"],
            n_max=n_max,
            random_state=int(rng.integers(0, 2**31 - 1)),
        )
        predicted = integrator(
            source_t,
            P=float(arm["P"]),
            B=float(arm["B"]),
            day0=anchor_day,
            dt=float(target["day"]) - anchor_day,
        )
        losses.append(
            symmetric_nearest_neighbor_mse(
                predicted,
                torch.as_tensor(observed, dtype=torch.float32, device=device),
            )
        )
        drift_penalties.append((predicted - source_t).pow(2).mean())
    supervision = sum(losses) / len(losses)
    drift_l2 = sum(drift_penalties) / len(drift_penalties)
    total = (
        float(train_config.rollout_loss_weight) * supervision
        + float(train_config.drift_l2_weight) * drift_l2
    )
    return total, supervision


def train_nearest_neighbor_drift(
    base_config: dict[str, Any],
    group_data: dict,
    transitions: pd.DataFrame,
    output_dir: str | Path,
    seed: int,
) -> dict[str, Any]:
    """Train the existing drift architecture with NN-MSE instead of OT."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source = project_path(base_config["inputs"]["drift_checkpoint"], PROJECT_ROOT)
    _, _, model_config, train_config, _ = load_drift_checkpoint(source, device="cpu")
    values = copy.deepcopy(base_config.get("drift", {}))
    values["loss_mode"] = "local_rollout"
    values["seed"] = int(seed)
    train_config = _apply_overrides(train_config, values, "drift")
    rollout_arms = build_rollout_arms(group_data)
    if not rollout_arms:
        raise ValueError("NN-MSE rollout supervision requires a multi-day treatment arm.")
    set_seed(seed)
    model, integrator, device = build_drift_model(
        model_config=model_config, train_config=train_config, device=train_config.device
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(train_config.lr), weight_decay=float(train_config.weight_decay)
    )
    scheduler = build_lr_scheduler(optimizer, train_config)
    rng = np.random.default_rng(int(seed))
    history = []
    model.train()
    for epoch in range(1, int(train_config.n_epochs) + 1):
        local_step_losses, local_supervision_values = [], []
        rollout_step_losses, rollout_supervision_values = [], []
        order = rng.permutation(len(transitions))
        for row_index in order:
            row = transitions.iloc[int(row_index)]
            batch = sample_transition(
                group_data, row, n_max=int(train_config.n_max),
                random_state=int(rng.integers(0, 2**31 - 1)),
            )
            source_t = torch.as_tensor(batch["Z0"], dtype=torch.float32, device=device)
            target_t = torch.as_tensor(batch["Z1"], dtype=torch.float32, device=device)
            predicted = integrator(
                source_t, P=batch["P"], B=batch["B"],
                day0=batch["day0"], dt=batch["dt"],
            )
            supervision = symmetric_nearest_neighbor_mse(predicted, target_t)
            drift_l2 = (predicted - source_t).pow(2).mean()
            loss = supervision + float(train_config.drift_l2_weight) * drift_l2
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if train_config.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_config.grad_clip))
            optimizer.step()
            local_step_losses.append(float(loss.detach().cpu()))
            local_supervision_values.append(float(supervision.detach().cpu()))
        for arm in rollout_arms:
            loss, supervision = _nearest_neighbor_rollout_loss(
                integrator, group_data, arm, train_config, device, rng
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if train_config.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_config.grad_clip))
            optimizer.step()
            rollout_step_losses.append(float(loss.detach().cpu()))
            rollout_supervision_values.append(float(supervision.detach().cpu()))
        local_step_mean = float(np.mean(local_step_losses)) if local_step_losses else 0.0
        rollout_step_mean = float(np.mean(rollout_step_losses)) if rollout_step_losses else 0.0
        local_supervision_mean = (
            float(np.mean(local_supervision_values)) if local_supervision_values else 0.0
        )
        rollout_supervision_mean = (
            float(np.mean(rollout_supervision_values)) if rollout_supervision_values else 0.0
        )
        mean_loss = local_step_mean + rollout_step_mean
        scheduler.step(mean_loss)
        history.append({
            "epoch": epoch,
            "loss": mean_loss,
            "nearest_neighbor_mse": (
                local_supervision_mean
                + float(train_config.rollout_loss_weight) * rollout_supervision_mean
            ),
            "local_step_loss": local_step_mean,
            "rollout_step_loss": rollout_step_mean,
            "local_nearest_neighbor_mse": local_supervision_mean,
            "rollout_nearest_neighbor_mse": rollout_supervision_mean,
            "loss_mode": train_config.loss_mode,
            "lr": get_current_lr(optimizer),
        })
        if train_config.print_every and epoch % int(train_config.print_every) == 0:
            print(f"[nn-mse-drift] epoch={epoch}/{train_config.n_epochs} loss={mean_loss:.6g}", flush=True)
    history_df = pd.DataFrame(history)
    save_drift_checkpoint(
        output_dir / "drift_model.pt", model, model_config, train_config, history_df=history_df
    )
    return {
        "integrator": integrator,
        "initialization_mode": "nearest_neighbor_mse_full_data_refit",
    }


__all__ = [
    "symmetric_nearest_neighbor_mse",
    "train_nearest_neighbor_drift",
]
