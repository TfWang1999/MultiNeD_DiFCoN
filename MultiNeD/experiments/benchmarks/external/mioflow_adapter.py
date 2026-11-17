from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
import torch
import torch.nn as nn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the MIOFlow core for a B1 benchmark fit unit.")
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
        raise RuntimeError("MIOFlow adapter requested CUDA but CUDA is unavailable")
    return device


def write_history(path: Path, history: dict[str, list]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history))
        writer.writeheader()
        for index in range(len(history["epoch"])):
            writer.writerow({key: history[key][index] for key in history})


def reconstruct_group_values(data) -> tuple[list[str], dict[str, np.ndarray]]:
    group_ids = [str(value) for value in data["training_group_ids"]]
    offsets = np.asarray(data["training_group_offsets"], dtype=np.int64)
    values = np.asarray(data["training_group_values"], dtype=np.float32)
    if offsets.shape[0] != len(group_ids) + 1:
        raise ValueError("MIOFlow group offsets do not match group IDs")
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
        raise ValueError(f"MIOFlow {values_key} must be two-dimensional")
    if offsets.ndim != 1 or offsets.shape[0] != expected_count + 1:
        raise ValueError(f"MIOFlow {offsets_key} does not match task IDs")
    if offsets[0] != 0 or offsets[-1] != values.shape[0] or np.any(np.diff(offsets) <= 0):
        raise ValueError(f"MIOFlow {offsets_key} is invalid")
    return [
        values[int(offsets[index]):int(offsets[index + 1])]
        for index in range(expected_count)
    ]


def pack_ragged_values(blocks: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    if not blocks:
        raise ValueError("MIOFlow produced no predictions")
    arrays = [np.asarray(block, dtype=np.float32) for block in blocks]
    if any(block.ndim != 2 or block.shape[0] == 0 for block in arrays):
        raise ValueError("MIOFlow predictions must be non-empty two-dimensional arrays")
    state_dims = {int(block.shape[1]) for block in arrays}
    if len(state_dims) != 1:
        raise ValueError("MIOFlow predictions have inconsistent state dimensions")
    offsets = np.asarray(
        [0, *np.cumsum([block.shape[0] for block in arrays], dtype=np.int64)],
        dtype=np.int64,
    )
    return np.concatenate(arrays, axis=0), offsets


class ConditionalMIOFlowODE(nn.Module):
    """Minimal f(z, t, P, B) extension around the official MIOFlow ODEFunc."""

    def __init__(self, official_ode, z_dim: int):
        super().__init__()
        self.official_ode = official_ode
        self.z_dim = int(z_dim)
        hidden_dim = int(self.official_ode.model[-1].in_features)
        output = nn.Linear(hidden_dim, self.z_dim)
        nn.init.kaiming_normal_(output.weight, mode="fan_in", nonlinearity="relu")
        nn.init.zeros_(output.bias)
        self.official_ode.model[-1] = output
        self.register_buffer("condition", torch.zeros(2, dtype=torch.float32), persistent=False)

    def set_condition(self, p_value: float, b_value: float) -> None:
        self.condition.copy_(
            torch.as_tensor([p_value, b_value], dtype=self.condition.dtype, device=self.condition.device)
        )

    def reset_momentum(self) -> None:
        self.official_ode.reset_momentum()

    def forward(self, t, state):
        condition = self.condition.to(dtype=state.dtype, device=state.device).expand(state.shape[0], 2)
        augmented = torch.cat((state, condition), dim=1)
        return self.official_ode(t, augmented)


class ConditionalTransitionDataset:
    """Distinct conditioned transitions consumed by one shared MIOFlow model."""

    def __init__(self, items: list[dict], model: ConditionalMIOFlowODE):
        self.items = items
        self.model = model

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict:
        item = self.items[index]
        self.model.set_condition(item["P"], item["B"])
        return {
            "X_start": torch.as_tensor(item["source"], dtype=torch.float32),
            "X_end": torch.as_tensor(item["target"], dtype=torch.float32),
            "t_start": float(item["source_day"]),
            "t_end": float(item["target_day"]),
            "interval_idx": int(index),
        }


def train_kwargs(settings: dict, device: torch.device) -> dict:
    return {
        "num_epochs": int(settings.get("epochs", 300)),
        "batch_size": int(settings.get("batch_size", 256)),
        "learning_rate": float(settings.get("learning_rate", 1e-3)),
        "device": str(device),
        "lambda_ot": float(settings.get("lambda_ot", 1.0)),
        "lambda_density": float(settings.get("lambda_density", 0.1)),
        "lambda_energy": float(settings.get("lambda_energy", 0.01)),
        "energy_time_steps": int(settings.get("energy_time_steps", 10)),
        "grad_clip": float(settings.get("grad_clip", 1.0)),
    }




def run_conditional(args, data, settings, repo, ODEFunc, train_mioflow, odeint) -> None:
    task_manifest_separated = bool(
        int(np.asarray(data["evaluation_task_manifest_separated"]).reshape(-1)[0])
    ) if "evaluation_task_manifest_separated" in data.files else False
    group_ids, raw_groups = reconstruct_group_values(data)
    all_values = np.concatenate([raw_groups[group_id] for group_id in group_ids], axis=0)
    z_dim = int(all_values.shape[1])
    mean = all_values.mean(axis=0)
    std = all_values.std(axis=0)
    std = np.where(std > 1e-8, std, 1.0).astype(np.float32)
    rng = np.random.default_rng(args.seed)
    max_points = int(settings.get("max_points_per_timepoint", 512))
    normalized_groups = {}
    for group_id in group_ids:
        snapshot = ((raw_groups[group_id] - mean) / std).astype(np.float32)
        if snapshot.shape[0] > max_points:
            snapshot = snapshot[rng.choice(snapshot.shape[0], max_points, replace=False)]
        normalized_groups[group_id] = snapshot

    transition_ids = [str(value) for value in data["transition_ids"]]
    source_indices = np.asarray(data["transition_source_group_indices"], dtype=np.int64)
    target_indices = np.asarray(data["transition_target_group_indices"], dtype=np.int64)
    source_days = np.asarray(data["transition_source_days"], dtype=float)
    target_days = np.asarray(data["transition_target_days"], dtype=float)
    p_values = np.asarray(data["transition_P"], dtype=float)
    b_values = np.asarray(data["transition_B"], dtype=float)
    n_transitions = len(transition_ids)
    lengths = {
        len(source_indices), len(target_indices), len(source_days), len(target_days),
        len(p_values), len(b_values), n_transitions,
    }
    if len(lengths) != 1:
        raise ValueError("MIOFlow transition arrays have inconsistent lengths")

    origin = float(min(np.asarray(data["training_group_days"], dtype=float)))
    device = resolve_device(settings)
    official = ODEFunc(
        input_dim=z_dim + 2,
        hidden_dim=int(settings.get("hidden_dim", 64)),
        momentum_beta=float(settings.get("momentum_beta", 0.0)),
    )
    model = ConditionalMIOFlowODE(official, z_dim=z_dim).to(device)
    items = []
    for index in range(n_transitions):
        items.append({
            "transition_id": transition_ids[index],
            "source": normalized_groups[group_ids[int(source_indices[index])]],
            "target": normalized_groups[group_ids[int(target_indices[index])]],
            "source_day": float(source_days[index] - origin),
            "target_day": float(target_days[index] - origin),
            "P": float(p_values[index]),
            "B": float(b_values[index]),
        })
    dataset = ConditionalTransitionDataset(items, model)
    history = train_mioflow(model=model, dataset=dataset, **train_kwargs(settings, device))

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
    predictions = []
    model.eval()
    for index in range(len(task_ids)):
        model.set_condition(float(task_p[index]), float(task_b[index]))
        source_tensor = torch.as_tensor(
            (task_sources[index] - mean) / std, dtype=torch.float32, device=device
        )
        model.reset_momentum()
        with torch.no_grad():
            times = torch.tensor(
                [task_source_days[index] - origin, task_target_days[index] - origin],
                dtype=torch.float32,
                device=device,
            )
            prediction = odeint(model, source_tensor, times)[-1]
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
            "state_dict": model.state_dict(),
            "mean": mean,
            "std": std,
            "settings": settings,
            "z_dim": z_dim,
            "conditioning_variables": ["P", "B", "time"],
            "evaluation_task_manifest_separated": task_manifest_separated,
        },
        args.output.parent / "mioflow_model.pt",
    )
    write_history(args.output.parent / "training_history.csv", history)
    args.metadata.write_text(
        json.dumps(
            {
                "method": "mioflow",
                "external_commit": git_commit(repo),
                "trainable_parameters": int(sum(parameter.numel() for parameter in model.parameters())),
                "training_group_count": len(group_ids),
                "training_transition_count": n_transitions,
                "prediction_target_count": len(task_ids),
                "global_joint_fit": True,
                "treatment_conditioned": True,
                "conditioning_variables": ["P", "B", "time"],
                "evaluation_task_manifest_separated": task_manifest_separated,
                "conditional_adapter_extension": (
                    "official ODEFunc depth and train_mioflow loss/optimizer loop retained; "
                    "velocity input extended from (z,time) to (z,time,P,B)"
                ),
                "official_training_loop_reused": True,
                "official_implementation_unmodified": False,
                "same_day_treatment_snapshots_merged": False,
                "one_shared_parameter_model": True,
                "gaga_embedding_used": False,
                "input_space": "predefined program-score state space",
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
    from mioflow.core.models.ode_model import ODEFunc
    from mioflow.mioflow import train_mioflow
    from torchdiffeq import odeint

    settings = json.loads(args.settings.read_text(encoding="utf-8"))
    seed_everything(args.seed)
    with np.load(args.input) as data:
        run_conditional(args, data, settings, repo, ODEFunc, train_mioflow, odeint)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
