from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
import torch
import torch.nn as nn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run official TorchCFM OT-CFM for one shared conditional B1 fit unit."
    )
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
        raise RuntimeError("official_ot_cfm requested CUDA but CUDA is unavailable")
    return device


def reconstruct_group_values(data) -> tuple[list[str], dict[str, np.ndarray]]:
    group_ids = [str(value) for value in data["training_group_ids"]]
    offsets = np.asarray(data["training_group_offsets"], dtype=np.int64)
    values = np.asarray(data["training_group_values"], dtype=np.float32)
    if offsets.ndim != 1 or offsets.shape[0] != len(group_ids) + 1:
        raise ValueError("OT-CFM group offsets do not match group IDs")
    if offsets[0] != 0 or offsets[-1] != values.shape[0] or np.any(np.diff(offsets) <= 0):
        raise ValueError("OT-CFM group offsets are invalid")
    return group_ids, {
        group_id: values[int(offsets[index]):int(offsets[index + 1])]
        for index, group_id in enumerate(group_ids)
    }


def reconstruct_task_sources(data, expected_count: int) -> list[np.ndarray]:
    values = np.asarray(data["task_source_values"], dtype=np.float32)
    offsets = np.asarray(data["task_source_offsets"], dtype=np.int64)
    if values.ndim != 2 or offsets.ndim != 1 or offsets.shape[0] != expected_count + 1:
        raise ValueError("OT-CFM task source ragged arrays are invalid")
    if offsets[0] != 0 or offsets[-1] != values.shape[0] or np.any(np.diff(offsets) <= 0):
        raise ValueError("OT-CFM task source offsets are invalid")
    return [
        values[int(offsets[index]):int(offsets[index + 1])]
        for index in range(expected_count)
    ]


def pack_predictions(blocks: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    if not blocks:
        raise ValueError("OT-CFM produced no predictions")
    arrays = [np.asarray(block, dtype=np.float32) for block in blocks]
    if any(block.ndim != 2 or block.shape[0] == 0 for block in arrays):
        raise ValueError("OT-CFM predictions must be non-empty two-dimensional arrays")
    if len({int(block.shape[1]) for block in arrays}) != 1:
        raise ValueError("OT-CFM predictions have inconsistent state dimensions")
    offsets = np.asarray(
        [0, *np.cumsum([block.shape[0] for block in arrays], dtype=np.int64)],
        dtype=np.int64,
    )
    return np.concatenate(arrays, axis=0), offsets


class SharedConditionalVelocity(nn.Module):
    """One v(z, physical_time, P, B) shared by every training transition."""

    def __init__(self, z_dim: int, hidden_dim: int, depth: int):
        super().__init__()
        layers: list[nn.Module] = []
        width = int(z_dim) + 3
        for _ in range(int(depth)):
            layers.extend([nn.Linear(width, int(hidden_dim)), nn.SiLU()])
            width = int(hidden_dim)
        layers.append(nn.Linear(width, int(z_dim)))
        self.network = nn.Sequential(*layers)

    def forward(
        self,
        state: torch.Tensor,
        physical_time: torch.Tensor,
        p_value: float,
        b_value: float,
    ) -> torch.Tensor:
        if physical_time.ndim == 1:
            physical_time = physical_time[:, None]
        condition = torch.empty(
            (state.shape[0], 2), dtype=state.dtype, device=state.device
        )
        condition[:, 0] = float(p_value)
        condition[:, 1] = float(b_value)
        return self.network(torch.cat((state, physical_time, condition), dim=1))


def convert_tau_velocity_to_physical_time(
    tau: torch.Tensor,
    velocity_tau: torch.Tensor,
    day0: float,
    day1: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert TorchCFM's dz/dtau target into dz/d(physical day)."""

    dt = float(day1) - float(day0)
    if dt <= 0:
        raise ValueError("OT-CFM physical-time conversion requires day1 > day0")
    return float(day0) + tau * dt, velocity_tau / dt


def write_history(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "loss"])
        writer.writeheader()
        writer.writerows(rows)


def run_conditional(
    args,
    data,
    settings: dict,
    repo: Path,
    matcher_class,
    package_version: str | None,
) -> None:
    separated = bool(
        int(np.asarray(data["evaluation_task_manifest_separated"]).reshape(-1)[0])
    ) if "evaluation_task_manifest_separated" in data.files else False
    if not separated:
        raise ValueError(
            "official_ot_cfm currently supports only separated shared-conditional LOTO tasks"
        )

    group_ids, raw_groups = reconstruct_group_values(data)
    task_ids = [str(value) for value in data["task_transition_ids"]]
    task_source_group_ids = [str(value) for value in data["task_source_group_ids"]]
    task_target_group_ids = [str(value) for value in data["task_target_group_ids"]]
    if not task_ids or len(task_ids) != len(set(task_ids)):
        raise ValueError("OT-CFM prediction task IDs must be non-empty and unique")
    if set(task_target_group_ids) & set(group_ids):
        raise ValueError("OT-CFM held-out target groups must not occur in training groups")
    if any(group_id not in raw_groups for group_id in task_source_group_ids):
        raise ValueError("OT-CFM prediction source must occur in training groups")

    source_indices = np.asarray(data["transition_source_group_indices"], dtype=np.int64)
    target_indices = np.asarray(data["transition_target_group_indices"], dtype=np.int64)
    source_days = np.asarray(data["transition_source_days"], dtype=float)
    target_days = np.asarray(data["transition_target_days"], dtype=float)
    p_values = np.asarray(data["transition_P"], dtype=float)
    b_values = np.asarray(data["transition_B"], dtype=float)
    n_transitions = len(data["transition_ids"])
    lengths = {
        len(source_indices), len(target_indices), len(source_days), len(target_days),
        len(p_values), len(b_values), n_transitions,
    }
    if lengths != {n_transitions} or n_transitions == 0:
        raise ValueError("OT-CFM transition arrays have inconsistent lengths")
    if np.any(target_days <= source_days):
        raise ValueError("OT-CFM requires every training transition to move forward in time")

    all_training_values = np.concatenate([raw_groups[group_id] for group_id in group_ids], axis=0)
    mean = all_training_values.mean(axis=0).astype(np.float32)
    std = all_training_values.std(axis=0).astype(np.float32)
    std = np.where(std > 1e-8, std, 1.0).astype(np.float32)
    normalized_groups = {
        group_id: ((values - mean) / std).astype(np.float32)
        for group_id, values in raw_groups.items()
    }

    device = resolve_device(settings)
    z_dim = int(all_training_values.shape[1])
    model = SharedConditionalVelocity(
        z_dim=z_dim,
        hidden_dim=int(settings.get("hidden_dim", 64)),
        depth=int(settings.get("depth", 2)),
    ).to(device)
    matcher = matcher_class(sigma=float(settings.get("sigma", 0.0)))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(settings.get("learning_rate", 1e-3)),
        weight_decay=float(settings.get("weight_decay", 1e-5)),
    )
    epochs = int(settings.get("epochs", 300))
    pairs_per_transition = int(settings.get("pairs_per_transition", 256))
    print_every = int(settings.get("print_every", 25))
    grad_clip = float(settings.get("grad_clip", 5.0))
    if epochs <= 0 or pairs_per_transition <= 0:
        raise ValueError("OT-CFM epochs and pairs_per_transition must be positive")

    rng = np.random.default_rng(int(args.seed) + 1701)
    history = []
    model.train()
    for epoch in range(1, epochs + 1):
        losses = []
        for transition_index in rng.permutation(n_transitions):
            index = int(transition_index)
            source = normalized_groups[group_ids[int(source_indices[index])]]
            target = normalized_groups[group_ids[int(target_indices[index])]]
            batch_size = min(pairs_per_transition, source.shape[0], target.shape[0])
            if batch_size < 2:
                raise ValueError("OT-CFM needs at least two cells at both transition endpoints")
            source_batch = source[rng.choice(source.shape[0], batch_size, replace=False)]
            target_batch = target[rng.choice(target.shape[0], batch_size, replace=False)]
            x0 = torch.as_tensor(source_batch, dtype=torch.float32, device=device)
            x1 = torch.as_tensor(target_batch, dtype=torch.float32, device=device)

            tau, xt, velocity_tau = matcher.sample_location_and_conditional_flow(x0, x1)
            physical_time, target_velocity = convert_tau_velocity_to_physical_time(
                tau,
                velocity_tau,
                float(source_days[index]),
                float(target_days[index]),
            )
            prediction = model(
                xt,
                physical_time,
                float(p_values[index]),
                float(b_values[index]),
            )
            loss = (prediction - target_velocity).pow(2).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        row = {"epoch": epoch, "loss": float(np.mean(losses))}
        history.append(row)
        if print_every and epoch % print_every == 0:
            print(
                f"[official-ot-cfm] epoch={epoch}/{epochs} loss={row['loss']:.6g}",
                flush=True,
            )

    task_sources = reconstruct_task_sources(data, len(task_ids))
    task_source_days = np.asarray(data["task_source_days"], dtype=float)
    task_target_days = np.asarray(data["task_target_days"], dtype=float)
    task_p = np.asarray(data["task_P"], dtype=float)
    task_b = np.asarray(data["task_B"], dtype=float)
    if len({len(task_ids), len(task_source_days), len(task_target_days), len(task_p), len(task_b)}) != 1:
        raise ValueError("OT-CFM prediction task arrays have inconsistent lengths")

    integration_steps_per_day = int(settings.get("integration_steps_per_day", 4))
    if integration_steps_per_day <= 0:
        raise ValueError("OT-CFM integration_steps_per_day must be positive")
    predictions = []
    model.eval()
    with torch.no_grad():
        for index, source in enumerate(task_sources):
            state = torch.as_tensor(
                (source - mean) / std, dtype=torch.float32, device=device
            )
            day0 = float(task_source_days[index])
            day1 = float(task_target_days[index])
            steps = max(1, int(math.ceil(abs(day1 - day0) * integration_steps_per_day)))
            step = (day1 - day0) / steps
            for step_index in range(steps):
                day = day0 + step_index * step
                time0 = torch.full((state.shape[0],), day, dtype=state.dtype, device=device)
                k1 = model(state, time0, float(task_p[index]), float(task_b[index]))
                midpoint = state + 0.5 * step * k1
                time_mid = torch.full(
                    (state.shape[0],), day + 0.5 * step, dtype=state.dtype, device=device
                )
                k2 = model(midpoint, time_mid, float(task_p[index]), float(task_b[index]))
                state = state + step * k2
            predictions.append((state.cpu().numpy() * std + mean).astype(np.float32))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    prediction_values, prediction_offsets = pack_predictions(predictions)
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
            "conditioning_variables": ["P", "B", "physical_time"],
            "matcher_class": matcher_class.__name__,
        },
        args.output.parent / "official_ot_cfm_model.pt",
    )
    write_history(args.output.parent / "training_history.csv", history)
    args.metadata.write_text(
        json.dumps(
            {
                "method": "official_ot_cfm",
                "external_commit": git_commit(repo),
                "external_package_version": package_version,
                "official_matcher_class": matcher_class.__name__,
                "official_matcher_unmodified": True,
                "trainable_parameters": int(sum(p.numel() for p in model.parameters())),
                "training_group_count": len(group_ids),
                "training_transition_count": n_transitions,
                "prediction_target_count": len(task_ids),
                "global_joint_fit": True,
                "one_shared_parameter_model": True,
                "treatment_conditioned": True,
                "conditioning_variables": ["P", "B", "physical_time"],
                "raw_treatment_conditioning": True,
                "physical_time_velocity": True,
                "tau_velocity_divided_by_dt": True,
                "coupling_scope": "within_each_observed_transition",
                "evaluation_task_manifest_separated": True,
                "normalization_scope": "training_groups_only",
                "heldout_target_used_for_training": False,
                "pk_module_used": False,
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
    import torchcfm
    from torchcfm.conditional_flow_matching import (
        ExactOptimalTransportConditionalFlowMatcher,
    )

    settings = json.loads(args.settings.read_text(encoding="utf-8"))
    seed_everything(args.seed)
    with np.load(args.input, allow_pickle=False) as data:
        run_conditional(
            args,
            data,
            settings,
            repo,
            ExactOptimalTransportConditionalFlowMatcher,
            getattr(torchcfm, "__version__", None),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
