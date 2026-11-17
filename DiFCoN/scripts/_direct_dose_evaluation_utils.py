"""Restore saved DiFCoN policies and cohorts for evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.cohort import build_patient_specs
from core.experiment import cohort_csv_path
from core.losses import DirectLossConfig
from core.policy import DirectDoseTreatmentCNN, load_policy_from_checkpoint
from core.reaction_runtime import REACTION_RUN_NAME, load_project_group_data, load_trained_reaction_model
from core.rollout import DirectRolloutConfig
from core.training import seed_everything


def resolve_from_root(value: str | Path) -> Path:
    """Resolve a CLI path relative to DiFCoN."""
    path = Path(value).expanduser()
    return (ROOT / path).resolve()


@dataclass
class SavedPolicyContext:
    checkpoint_path: Path
    seeds: dict[str, int]
    reaction_model: Any
    policy: DirectDoseTreatmentCNN
    rollout_config: DirectRolloutConfig
    loss_config: DirectLossConfig
    group_data: dict
    device: torch.device


def load_saved_policy_context(run_dir: str | Path, device: str = "auto") -> SavedPolicyContext:
    """Load a frozen policy and its checkpoint-matched Reaction environment."""
    run_dir = resolve_from_root(run_dir)
    config = json.loads((run_dir / "resolved_config.json").read_text(encoding="utf-8"))
    checkpoint_path = run_dir / "checkpoints" / config["output"]["checkpoint_filename"]
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    identity = checkpoint["model_identity"]
    if identity["reaction_run_name"] != REACTION_RUN_NAME:
        raise ValueError(f"Expected Reaction run {REACTION_RUN_NAME!r} in {checkpoint_path}.")

    seeds = identity["seeds"]
    seed_everything(seeds["model"])
    group_data = load_project_group_data()
    reaction_model, model_device = load_trained_reaction_model(device=device, **config["model"])
    return SavedPolicyContext(
        checkpoint_path=checkpoint_path,
        seeds=seeds,
        reaction_model=reaction_model,
        policy=load_policy_from_checkpoint(checkpoint, model_device),
        rollout_config=DirectRolloutConfig(**checkpoint["rollout_config"]),
        loss_config=DirectLossConfig(**checkpoint["loss_config"]),
        group_data=group_data,
        device=model_device,
    )


def load_frozen_cohort(cohort_dir: str | Path, split: str) -> tuple[pd.DataFrame, Path]:
    table_path = cohort_csv_path(resolve_from_root(cohort_dir), split)
    return pd.read_csv(table_path), table_path


def limit_patients(frame: pd.DataFrame, max_patients: int | None) -> pd.DataFrame:
    if max_patients is None:
        return frame.copy().reset_index(drop=True)
    if int(max_patients) < 1:
        raise ValueError("--max-patients must be >= 1 when provided.")
    return frame.iloc[: int(max_patients)].copy().reset_index(drop=True)


def make_specs(
    context: SavedPolicyContext,
    cohort_df: pd.DataFrame,
    n_particles: int,
    rollout_seed: int,
) -> list[dict[str, Any]]:
    return build_patient_specs(
        cohort_df, context.group_data, context.reaction_model, n_particles, rollout_seed,
    )


def make_output_dir(path: str | Path) -> Path:
    path = resolve_from_root(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
