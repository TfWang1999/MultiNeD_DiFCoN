from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random
import subprocess
import sys
from types import MethodType, SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run conditional TIGON for a cellular-distribution benchmark.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--external-repo", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    return parser.parse_args()


def git_commit(repo: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(settings: dict) -> torch.device:
    device = torch.device(
        str(settings.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("TIGON adapter requested CUDA but CUDA is unavailable")
    return device


class VelocityOnly(nn.Module):
    """Expose TIGON's learned velocity without evaluating the growth trace."""

    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, t, state):
        if hasattr(self.func, "velocity"):
            return self.func.velocity(t, state)
        batch = state.shape[0]
        time = torch.as_tensor(t, dtype=state.dtype, device=state.device).expand(batch, 1)
        hidden = torch.cat((time, state), dim=1)
        value = hidden
        for layer in self.func.hyper_net1.net:
            value = layer(value)
        return self.func.hyper_net1.out(value)


def _device_safe_velocity_forward(self, t, x):
    """Official TIGON hypernetwork forward with time on the state device."""

    time = torch.as_tensor(t, dtype=x.dtype, device=x.device).reshape(1, 1)
    value = torch.cat((time.expand(x.shape[0], 1), x), dim=1)
    for layer in self.net:
        value = layer(value)
    return self.out(value)


def _device_safe_growth_forward(self, t, x):
    time = torch.as_tensor(t, dtype=x.dtype, device=x.device).reshape(1, 1)
    state = torch.cat((time.expand(x.shape[0], 1), x), dim=1)
    return self.net(state)


class ConditionalUOT(nn.Module):
    """Shared TIGON v(z,t,P,B), retaining the official velocity/growth layers."""

    def __init__(self, official_core, z_dim: int, trace_function):
        super().__init__()
        self.z_dim = int(z_dim)
        self.hyper_net1 = official_core.hyper_net1
        self.hyper_net2 = official_core.hyper_net2
        hidden_dim = int(self.hyper_net1.out.in_features)
        self.hyper_net1.out = nn.Linear(hidden_dim, self.z_dim)
        self.trace_function = trace_function
        self.register_buffer("condition", torch.zeros(2, dtype=torch.float32), persistent=False)

    def set_condition(self, p_value: float, b_value: float) -> None:
        self.condition.copy_(
            torch.as_tensor([p_value, b_value], dtype=self.condition.dtype, device=self.condition.device)
        )

    def _augmented(self, state):
        condition = self.condition.to(dtype=state.dtype, device=state.device).expand(state.shape[0], 2)
        return torch.cat((state, condition), dim=1)

    def velocity(self, t, state):
        return self.hyper_net1(t, self._augmented(state))

    def forward(self, t, states):
        z = states[0]
        batch_size = z.shape[0]
        with torch.set_grad_enabled(True):
            z.requires_grad_(True)
            augmented = self._augmented(z)
            dz_dt = self.hyper_net1(t, augmented)
            growth = self.hyper_net2(t, augmented)
            dlogp_z_dt = growth - self.trace_function(dz_dt, z).view(batch_size, 1)
        return dz_dt, growth, dlogp_z_dt


def reconstruct_group_values(data) -> tuple[list[str], dict[str, np.ndarray]]:
    group_ids = [str(value) for value in data["training_group_ids"]]
    offsets = np.asarray(data["training_group_offsets"], dtype=np.int64)
    values = np.asarray(data["training_group_values"], dtype=np.float32)
    if offsets.shape[0] != len(group_ids) + 1:
        raise ValueError("TIGON group offsets do not match group IDs")
    groups = {
        group_id: values[int(offsets[index]):int(offsets[index + 1])]
        for index, group_id in enumerate(group_ids)
    }
    return group_ids, groups


def reconstruct_ragged_values(
    data,
    *,
    values_key: str,
    offsets_key: str,
    expected_count: int,
) -> list[np.ndarray]:
    values = np.asarray(data[values_key], dtype=np.float32)
    offsets = np.asarray(data[offsets_key], dtype=np.int64)
    if values.ndim != 2:
        raise ValueError(f"TIGON {values_key} must be two-dimensional")
    if offsets.ndim != 1 or offsets.shape[0] != expected_count + 1:
        raise ValueError(f"TIGON {offsets_key} does not match task IDs")
    if offsets[0] != 0 or offsets[-1] != values.shape[0] or np.any(np.diff(offsets) <= 0):
        raise ValueError(f"TIGON {offsets_key} is invalid")
    return [
        values[int(offsets[index]):int(offsets[index + 1])]
        for index in range(expected_count)
    ]


def pack_ragged_values(blocks: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    if not blocks:
        raise ValueError("TIGON produced no predictions")
    arrays = [np.asarray(block, dtype=np.float32) for block in blocks]
    if any(block.ndim != 2 or block.shape[0] == 0 for block in arrays):
        raise ValueError("TIGON predictions must be non-empty two-dimensional arrays")
    state_dims = {int(block.shape[1]) for block in arrays}
    if len(state_dims) != 1:
        raise ValueError("TIGON predictions have inconsistent state dimensions")
    offsets = np.asarray(
        [0, *np.cumsum([block.shape[0] for block in arrays], dtype=np.int64)],
        dtype=np.int64,
    )
    return np.concatenate(arrays, axis=0), offsets


def build_config(settings: dict) -> SimpleNamespace:
    return SimpleNamespace(
        niters=int(settings.get("iterations", 2000)),
        lr=float(settings.get("learning_rate", 3e-3)),
        num_samples=int(settings.get("num_samples", 64)),
        hidden_dim=int(settings.get("hidden_dim", 32)),
        n_hiddens=int(settings.get("n_hiddens", 4)),
        activation=str(settings.get("activation", "Tanh")),
    )


def integration_options() -> dict:
    return {
        "method": "Dopri5",
        "h": None,
        "rtol": 1e-3,
        "atol": 1e-5,
        "print_neval": False,
        "neval_max": 1000000,
        "safety": None,
    }


def write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)




def build_joint_trajectories(
    data,
    group_ids,
    normalized_groups,
    device,
    *,
    allow_disconnected: bool = False,
) -> list[dict]:
    source_indices = np.asarray(data["transition_source_group_indices"], dtype=np.int64)
    target_indices = np.asarray(data["transition_target_group_indices"], dtype=np.int64)
    source_days = np.asarray(data["transition_source_days"], dtype=float)
    target_days = np.asarray(data["transition_target_days"], dtype=float)
    p_values = np.asarray(data["transition_P"], dtype=float)
    b_values = np.asarray(data["transition_B"], dtype=float)
    origin = float(min(np.asarray(data["training_group_days"], dtype=float)))
    condition_keys = sorted({(float(p), float(b)) for p, b in zip(p_values, b_values)})
    trajectories = []
    for p_value, b_value in condition_keys:
        indices = [
            index for index in range(len(p_values))
            if np.isclose(p_values[index], p_value) and np.isclose(b_values[index], b_value)
        ]
        indices.sort(key=lambda index: (source_days[index], target_days[index], index))
        segments: list[list[int]] = []
        current_segment: list[int] = []
        current_target = None
        for index in indices:
            source_group = group_ids[int(source_indices[index])]
            if current_segment and source_group != current_target:
                if not allow_disconnected:
                    raise ValueError(
                        f"TIGON conditional trajectory P={p_value}, B={b_value} "
                        f"is not a chain at {source_group}"
                    )
                segments.append(current_segment)
                current_segment = []
            current_segment.append(index)
            current_target = group_ids[int(target_indices[index])]
        if current_segment:
            segments.append(current_segment)

        for segment_index, segment in enumerate(segments):
            first = segment[0]
            ordered_groups = [group_ids[int(source_indices[first])]]
            times = [float(source_days[first] - origin)]
            for index in segment:
                source_group = group_ids[int(source_indices[index])]
                target_group = group_ids[int(target_indices[index])]
                if source_group != ordered_groups[-1]:
                    raise AssertionError("Disconnected TIGON segment construction failed")
                ordered_groups.append(target_group)
                times.append(float(target_days[index] - origin))
            if any(next_time <= time for time, next_time in zip(times, times[1:])):
                raise ValueError(f"TIGON trajectory times must increase for P={p_value}, B={b_value}")
            trajectories.append({
                "P": p_value,
                "B": b_value,
                "segment_index": segment_index,
                "group_ids": ordered_groups,
                "times": times,
                "data_train": [
                    torch.as_tensor(normalized_groups[group_id], dtype=torch.float32, device=device)
                    for group_id in ordered_groups
                ],
                "train_time": list(range(len(ordered_groups))),
                "transition_count": len(segment),
                "sigma": 1.0,
            })
    return trajectories


def run_conditional(args, data, settings, repo, official, odeint) -> None:
    task_manifest_separated = bool(
        int(np.asarray(data["evaluation_task_manifest_separated"]).reshape(-1)[0])
    ) if "evaluation_task_manifest_separated" in data.files else False
    allow_disconnected = bool(
        int(np.asarray(data["allow_disconnected_training_trajectories"]).reshape(-1)[0])
    ) if "allow_disconnected_training_trajectories" in data.files else False
    UOT, initialize_weights, train_model, trace_df_dz = official
    group_ids, raw_groups = reconstruct_group_values(data)
    all_values = np.concatenate([raw_groups[group_id] for group_id in group_ids], axis=0)
    z_dim = int(all_values.shape[1])
    mean = all_values.mean(axis=0)
    std = all_values.std(axis=0)
    std = np.where(std > 1e-8, std, 1.0).astype(np.float32)
    rng = np.random.default_rng(args.seed)
    max_points = int(settings.get("max_points_per_timepoint", 256))
    normalized_groups = {}
    for group_id in group_ids:
        snapshot = ((raw_groups[group_id] - mean) / std).astype(np.float32)
        if snapshot.shape[0] > max_points:
            snapshot = snapshot[rng.choice(snapshot.shape[0], max_points, replace=False)]
        normalized_groups[group_id] = snapshot

    device = resolve_device(settings)
    config = build_config(settings)
    official_core = UOT(
        in_out_dim=z_dim + 2,
        hidden_dim=config.hidden_dim,
        n_hiddens=config.n_hiddens,
        activation=config.activation,
    )
    func = ConditionalUOT(official_core, z_dim=z_dim, trace_function=trace_df_dz).to(device)
    func.hyper_net1.forward = MethodType(_device_safe_velocity_forward, func.hyper_net1)
    func.hyper_net2.forward = MethodType(_device_safe_growth_forward, func.hyper_net2)
    func.apply(initialize_weights)
    trajectories = build_joint_trajectories(
        data,
        group_ids,
        normalized_groups,
        device,
        allow_disconnected=allow_disconnected,
    )
    if not allow_disconnected and len(trajectories) != 4:
        raise ValueError(f"Expected four P/B trajectories for global TIGON, observed {len(trajectories)}")
    if not trajectories:
        raise ValueError("TIGON conditional training requires at least one transition trajectory")

    optimizer = optim.Adam(func.parameters(), lr=config.lr, weight_decay=0.01)
    milestones = sorted({max(1, config.niters - 400), max(1, config.niters - 200)})
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.5)
    mse = nn.MSELoss()
    rows = []
    options = integration_options()
    n_trajectories = len(trajectories)
    for iteration in range(1, config.niters + 1):
        optimizer.zero_grad(set_to_none=True)
        loss_values = []
        transport_values = []
        first_values = []
        adjacent_values = []
        for trajectory in trajectories:
            func.set_condition(trajectory["P"], trajectory["B"])
            loss, transport, sigma, local_first, local_adjacent = train_model(
                mse,
                func,
                config,
                trajectory["data_train"],
                trajectory["train_time"],
                trajectory["times"],
                trajectory["sigma"],
                options.copy(),
                device,
                iteration,
            )
            (loss / n_trajectories).backward()
            trajectory["sigma"] = float(sigma)
            loss_values.append(float(loss.detach().cpu()))
            transport_values.append(float(transport[-1].mean().detach().cpu()))
            first_values.append(float(local_first.mean().detach().cpu()))
            adjacent_values.append(float(local_adjacent.mean().detach().cpu()))
        optimizer.step()
        scheduler.step()
        rows.append({
            "iteration": iteration,
            "loss": float(np.mean(loss_values)),
            "transport": float(np.mean(transport_values)),
            "sigma": float(np.mean([trajectory["sigma"] for trajectory in trajectories])),
            "local_first": float(np.mean(first_values)),
            "local_adjacent": float(np.mean(adjacent_values)),
        })

    task_ids = [str(value) for value in data["task_transition_ids"]]
    task_sources = reconstruct_ragged_values(
        data,
        values_key="task_source_values",
        offsets_key="task_source_offsets",
        expected_count=len(task_ids),
    )
    task_source_days = np.asarray(data["task_source_days"], dtype=float)
    task_target_days = np.asarray(data["task_target_days"], dtype=float)
    task_p = np.asarray(data["task_P"], dtype=float)
    task_b = np.asarray(data["task_B"], dtype=float)
    origin = float(min(np.asarray(data["training_group_days"], dtype=float)))
    velocity = VelocityOnly(func).to(device).eval()
    predictions = []
    for index in range(len(task_ids)):
        func.set_condition(float(task_p[index]), float(task_b[index]))
        source_tensor = torch.as_tensor(
            (task_sources[index] - mean) / std, dtype=torch.float32, device=device
        )
        with torch.no_grad():
            times = torch.tensor(
                [task_source_days[index] - origin, task_target_days[index] - origin],
                dtype=torch.float32,
                device=device,
            )
            prediction = odeint(
                velocity,
                source_tensor,
                times,
                method="midpoint",
                options={"step_size": 0.1},
            )[-1]
        predictions.append((prediction.cpu().numpy() * std + mean).astype(np.float32))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    prediction_values, prediction_offsets = pack_ragged_values(predictions)
    np.savez_compressed(
        args.output,
        transition_ids=np.asarray(task_ids),
        prediction_values=prediction_values,
        prediction_offsets=prediction_offsets,
    )
    torch.save(
        {
            "func_state_dict": func.state_dict(),
            "mean": mean,
            "std": std,
            "settings": settings,
            "z_dim": z_dim,
            "conditioning_variables": ["P", "B", "time"],
            "evaluation_task_manifest_separated": task_manifest_separated,
            "allow_disconnected_training_trajectories": allow_disconnected,
        },
        args.output.parent / "tigon_model.pt",
    )
    write_rows(args.output.parent / "training_history.csv", rows)
    args.metadata.write_text(
        json.dumps(
            {
                "method": "tigon",
                "external_commit": git_commit(repo),
                "expected_release": "v1.0.0",
                "trainable_parameters": int(sum(parameter.numel() for parameter in func.parameters())),
                "training_group_count": len(group_ids),
                "training_transition_count": int(len(data["transition_ids"])),
                "training_trajectory_count": len(trajectories),
                "prediction_target_count": len(task_ids),
                "global_joint_fit": True,
                "treatment_conditioned": True,
                "conditioning_variables": ["P", "B", "time"],
                "evaluation_task_manifest_separated": task_manifest_separated,
                "allow_disconnected_training_trajectories": allow_disconnected,
                "conditional_adapter_extension": (
                    "official TIGON velocity/growth layer depth and per-trajectory train_model objective retained; "
                    "shared velocity/growth inputs extended to (z,time,P,B), with mean gradient over "
                    "all available conditional trajectory segments"
                ),
                "joint_objective_reduction": "mean_over_condition_trajectory_objectives",
                "official_train_model_objective_reused": True,
                "official_implementation_unmodified": False,
                "same_day_treatment_snapshots_merged": False,
                "one_shared_parameter_model": True,
                "growth_output_used_for_scoring": False,
                "cell_count_interpreted_as_population_growth_truth": False,
                "adapter_device_patch": "time tensor follows the state device",
            },
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    repo = args.external_repo.resolve()
    sys.path.insert(0, str(repo))
    from utility import UOT, initialize_weights, trace_df_dz, train_model
    from torchdiffeq import odeint

    settings = json.loads(args.settings.read_text(encoding="utf-8"))
    seed_everything(args.seed)
    official = UOT, initialize_weights, train_model, trace_df_dz
    with np.load(args.input) as data:
        run_conditional(args, data, settings, repo, official, odeint)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
