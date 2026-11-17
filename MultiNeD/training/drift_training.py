
from pathlib import Path as _PathForBootstrap
import sys as _sysForBootstrap

_CODE_DIR = _PathForBootstrap(__file__).resolve().parents[1]
if str(_CODE_DIR) not in _sysForBootstrap.path:
    _sysForBootstrap.path.insert(0, str(_CODE_DIR))

from dataclasses import asdict, dataclass
from pathlib import Path
import random

import numpy as np
import torch
import pandas as pd

from common.cd8_model_components import CD8DriftNet, DriftIntegrator, DriftModelConfig
from common.cd8_ot_losses import sinkhorn_divergence
from common.cd8_zdist_data import mouse_balanced_sample, sample_transition


@dataclass
class DriftTrainConfig:
    """Training configuration for the drift-only model."""
    n_epochs: int = 200
    n_max: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-5
    sinkhorn_epsilon: float = 0.08
    sinkhorn_iters: int = 60
    loss_mode: str = "local"
    rollout_loss_weight: float = 0.25
    rollout_n_max: int | None = None
    n_steps_per_day: int = 10
    time_scale: float = 15.0
    drift_l2_weight: float = 1e-4
    grad_clip: float = 5.0
    seed: int = 2026
    device: str = "auto"
    lr_patience: int = 20
    lr_factor: float = 0.5
    min_lr: float = 1e-6
    print_every: int = 1


def set_random_seed(seed):
    """Set Python, NumPy, and PyTorch random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(device="auto"):
    """Resolve `auto`, `cpu`, or `cuda` into a torch.device."""
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def build_drift_model(model_config=None, train_config=None, device="auto"):
    """Create a CD8DriftNet and matching integrator."""
    model_config = model_config or DriftModelConfig()
    train_config = train_config or DriftTrainConfig()
    device = choose_device(device if device != "auto" else train_config.device)

    model = CD8DriftNet(model_config).to(device)
    integrator = DriftIntegrator(
        model=model,
        n_steps_per_day=train_config.n_steps_per_day,
        time_scale=train_config.time_scale,
        pk_config=model_config.pk,
    ).to(device)
    return model, integrator, device


def build_lr_scheduler(optimizer, train_config):
    """Reduce the learning rate when the epoch loss stops improving."""
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=train_config.lr_factor,
        patience=train_config.lr_patience,
        min_lr=train_config.min_lr,
    )


def get_current_lr(optimizer):
    """Return the current learning rate from the first optimizer group."""
    return float(optimizer.param_groups[0]["lr"])


def tensor_from_numpy(array, device):
    """Convert a NumPy array to float32 torch tensor on device."""
    return torch.as_tensor(array, dtype=torch.float32, device=device)


def predict_transition(integrator, batch, device):
    """Roll out source particles through the drift model for one transition."""
    z0 = tensor_from_numpy(batch["Z0"], device)
    return integrator(
        z0,
        P=batch["P"],
        B=batch["B"],
        day0=batch["day0"],
        dt=batch["dt"],
    )


def transition_ot_loss(integrator, batch, train_config, device):
    """
    Compute weighted OT loss between predicted source rollout and target cells.
    """
    z_pred = predict_transition(integrator, batch, device)
    z_target = tensor_from_numpy(batch["Z1"], device)

    loss = sinkhorn_divergence(
        z_pred,
        z_target,
        epsilon=train_config.sinkhorn_epsilon,
        n_iters=train_config.sinkhorn_iters,
    )

    raw_weight = float(batch["loss_weight"])
    drift_l2 = (z_pred - tensor_from_numpy(batch["Z0"], device)).pow(2).mean()
    total = loss + train_config.drift_l2_weight * drift_l2

    metrics = {
        "loss": float(total.detach().cpu()),
        "ot_loss": float(loss.detach().cpu()),
        "drift_l2": float(drift_l2.detach().cpu()),
        "loss_weight": 1.0,
        "raw_loss_weight": raw_weight,
        "transition_id": batch["transition_id"],
        "transition_type": batch["transition_type"],
        "source_group": batch["source_group"],
        "target_group": batch["target_group"],
    }

    return total, metrics


def build_rollout_arms(group_data):
    """
    Build treatment arms for trajectory-level rollout consistency.

    Each arm contains all available groups with the same treatment coding
    (P, B), sorted by day. Later observations are targets from Control day 1.
    """
    anchor_day = float(group_data["day1__Control"]["day"])
    records = []
    for group_id, group in group_data.items():
        records.append({
            "group_id": group_id,
            "day": float(group["day"]),
            "treatment_clean": group["treatment_clean"],
            "P": float(group["P"]),
            "B": float(group["B"]),
            "loss_weight": float(group.get("loss_weight", 1.0)),
        })
    df = pd.DataFrame(records).sort_values(["P", "B", "day", "group_id"])
    arms = []
    for (P, B), cur in df.groupby(["P", "B"], sort=True):
        cur = cur.sort_values(["day", "group_id"]).reset_index(drop=True)
        if not cur["day"].gt(anchor_day).any():
            continue
        arms.append({
            "arm_id": f"P{int(P)}_B{int(B)}",
            "P": float(P),
            "B": float(B),
            "treatment_clean": str(cur["treatment_clean"].iloc[-1]),
            "groups": cur.to_dict(orient="records"),
            "days": cur["day"].tolist(),
        })
    return arms


def rollout_arm_loss(integrator, group_data, arm, train_config, device, random_state=None):
    """
    Compute trajectory consistency loss for one treatment arm.

    Sample the initial state from Control day 1 and propagate it under
    the arm's P/B conditions to each later observation.
    """
    rng = np.random.default_rng(random_state)
    n_max = train_config.rollout_n_max or train_config.n_max
    groups = sorted(arm["groups"], key=lambda x: (x["day"], x["group_id"]))
    anchor_group = "day1__Control"
    anchor_day = float(group_data[anchor_group]["day"])
    targets = [g for g in groups if float(g["day"]) > anchor_day]
    if not targets:
        raise ValueError(f"No rollout target found for arm {arm['arm_id']}")

    Z0, _ = mouse_balanced_sample(
        group_data,
        anchor_group,
        n_max=n_max,
        random_state=int(rng.integers(0, 2**31 - 1)),
    )
    z0 = tensor_from_numpy(Z0, device)

    losses = []
    drift_l2_values = []
    for target in targets:
        Zt, _ = mouse_balanced_sample(
            group_data,
            target["group_id"],
            n_max=n_max,
            random_state=int(rng.integers(0, 2**31 - 1)),
        )
        dt = float(target["day"]) - anchor_day
        z_pred = integrator(
            z0,
            P=arm["P"],
            B=arm["B"],
            day0=anchor_day,
            dt=dt,
        )
        z_target = tensor_from_numpy(Zt, device)
        losses.append(sinkhorn_divergence(
            z_pred,
            z_target,
            epsilon=train_config.sinkhorn_epsilon,
            n_iters=train_config.sinkhorn_iters,
        ))
        drift_l2_values.append((z_pred - z0).pow(2).mean())

    rollout_ot = sum(losses) / len(losses)
    drift_l2 = sum(drift_l2_values) / len(drift_l2_values)
    total = train_config.rollout_loss_weight * rollout_ot + train_config.drift_l2_weight * drift_l2

    metrics = {
        "loss": float(total.detach().cpu()),
        "rollout_loss": float(rollout_ot.detach().cpu()),
        "rollout_drift_l2": float(drift_l2.detach().cpu()),
        "rollout_weight": float(train_config.rollout_loss_weight),
        "arm_id": arm["arm_id"],
        "treatment_clean": arm["treatment_clean"],
        "P": arm["P"],
        "B": arm["B"],
        "anchor_group": anchor_group,
        "anchor_day": anchor_day,
        "n_targets": len(targets),
        "target_days": ",".join(f"{float(t['day']):g}" for t in targets),
    }
    return total, metrics


def train_drift_model(
    group_data,
    transitions,
    model_config=None,
    train_config=None,
):
    """
    Train v_theta(z, P, B, t) using temporal transitions and OT loss.

    Returns
    -------
    model, integrator, history_df
    """
    train_config = train_config or DriftTrainConfig()
    model_config = model_config or DriftModelConfig()

    set_random_seed(train_config.seed)
    model, integrator, device = build_drift_model(model_config, train_config)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.lr,
        weight_decay=train_config.weight_decay,
    )
    scheduler = build_lr_scheduler(optimizer, train_config)

    transitions = transitions.reset_index(drop=True)
    rng = np.random.default_rng(train_config.seed)
    history = []

    loss_mode = train_config.loss_mode.lower()
    if loss_mode not in ("local", "rollout", "rollout_only", "local_rollout", "hybrid", "local+rollout"):
        raise ValueError("loss_mode must be 'local', 'rollout', or 'local_rollout'.")
    use_local = loss_mode not in ("rollout", "rollout_only")
    use_rollout = loss_mode in ("local_rollout", "hybrid", "local+rollout")
    use_rollout = use_rollout or loss_mode in ("rollout", "rollout_only")
    rollout_arms = build_rollout_arms(group_data) if use_rollout else []
    if use_local and transitions.empty:
        raise ValueError("Local drift supervision requires at least one transition.")
    if use_rollout and not rollout_arms:
        raise ValueError("Rollout drift supervision requires at least one multi-day treatment arm.")

    model.train()
    for epoch in range(1, train_config.n_epochs + 1):
        epoch_records = []
        rollout_records = []

        transition_indices = []
        if use_local:
            transition_indices = np.arange(len(transitions))
            rng.shuffle(transition_indices)

        for row_idx in transition_indices:
            row = transitions.iloc[row_idx]
            seed = int(rng.integers(0, 2**31 - 1))
            batch = sample_transition(
                group_data,
                row,
                n_max=train_config.n_max,
                random_state=seed,
            )

            optimizer.zero_grad(set_to_none=True)
            loss, metrics = transition_ot_loss(integrator, batch, train_config, device)
            loss.backward()

            if train_config.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)

            optimizer.step()
            epoch_records.append(metrics)

        if use_rollout:
            for arm in rollout_arms:
                seed = int(rng.integers(0, 2**31 - 1))
                optimizer.zero_grad(set_to_none=True)
                rollout_loss, rollout_metrics = rollout_arm_loss(
                    integrator,
                    group_data,
                    arm,
                    train_config,
                    device,
                    random_state=seed,
                )
                rollout_loss.backward()

                if train_config.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)

                optimizer.step()
                rollout_records.append(rollout_metrics)

        epoch_df = pd.DataFrame(epoch_records)
        rollout_df = pd.DataFrame(rollout_records)
        local_loss_mean = float(epoch_df["loss"].mean()) if not epoch_df.empty else 0.0
        rollout_step_loss_mean = float(rollout_df["loss"].mean()) if not rollout_df.empty else 0.0
        summary = {
            "epoch": epoch,
            "loss": local_loss_mean + rollout_step_loss_mean,
            "local_loss": local_loss_mean,
            "ot_loss": float(epoch_df["ot_loss"].mean()) if not epoch_df.empty else 0.0,
            "drift_l2": float(epoch_df["drift_l2"].mean()) if not epoch_df.empty else 0.0,
            "rollout_loss": float(rollout_df["rollout_loss"].mean()) if not rollout_df.empty else 0.0,
            "rollout_step_loss": rollout_step_loss_mean,
        }
        scheduler.step(summary["loss"])
        summary["lr"] = get_current_lr(optimizer)
        history.append(summary)

        if train_config.print_every and epoch % train_config.print_every == 0:
            print(
                "[drift-train] "
                f"epoch={epoch:04d}/{train_config.n_epochs} "
                f"loss={summary['loss']:.6g} "
                f"ot={summary['ot_loss']:.6g} "
                f"rollout={summary['rollout_loss']:.6g} "
                f"drift_l2={summary['drift_l2']:.6g} "
                f"lr={summary['lr']:.3e}",
                flush=True,
            )

    return model, integrator, pd.DataFrame(history)


def save_drift_checkpoint(path, model, model_config, train_config, history_df=None):
    """Save model weights and configs for later full-model coupling."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "model_config": asdict(model_config),
        "train_config": asdict(train_config),
    }
    if history_df is not None:
        payload["history"] = history_df.to_dict(orient="records")
    torch.save(payload, path)
    return path


def load_drift_checkpoint(path, device="auto"):
    """
    Load a saved drift checkpoint and return model, integrator, configs, history.

    This is the intended handoff point from drift-only training to the later
    full PDE-ODE fitting code.
    """
    path = Path(path)
    checkpoint = torch.load(path, map_location=choose_device(device), weights_only=False)

    model_config = DriftModelConfig.from_saved_config(checkpoint["model_config"])

    saved_training = dict(checkpoint["train_config"])
    saved_training.pop("rollout_anchor", None)
    train_config = DriftTrainConfig(**saved_training)
    model, integrator, _ = build_drift_model(
        model_config=model_config,
        train_config=train_config,
        device=device,
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    history = pd.DataFrame(checkpoint.get("history", []))
    return model, integrator, model_config, train_config, history
