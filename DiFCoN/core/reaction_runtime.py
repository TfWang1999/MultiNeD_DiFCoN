from __future__ import annotations

import copy
import sys
from pathlib import Path

import torch


PKG_DIR = Path(__file__).resolve().parents[2]
CODE_DIR = PKG_DIR / "MultiNeD"

for project_path in (PKG_DIR, CODE_DIR):
    if str(project_path) not in sys.path:
        sys.path.insert(0, str(project_path))


from training.drift_training import load_drift_checkpoint
from training.coupled_dynamics import ReactionCoupledModel, load_reaction_checkpoint_configs
from common.cd8_zdist_data import load_training_package
from MultiNeD.scripts.reproduce.upstream_paths import (
    DEFAULT_DRIFT_CHECKPOINT,
    DEFAULT_REACTION_CHECKPOINT,
)


# Training identity recorded in the saved DiFCoN policy checkpoints.
REACTION_RUN_NAME = "all_with_qin_michaelis_b_michaelis_add"


def load_project_group_data():
    group_data, _, _, _ = load_training_package(PKG_DIR)
    return group_data


def load_trained_reaction_model(
    device: str | torch.device = "auto",
    drift_checkpoint: str | Path | None = None,
    reaction_checkpoint: str | Path | None = None,
):
    """Load the frozen differentiable environment and its saved PK settings."""
    drift_ckpt_path = PKG_DIR / (drift_checkpoint or DEFAULT_DRIFT_CHECKPOINT)
    reaction_ckpt_path = PKG_DIR / (reaction_checkpoint or DEFAULT_REACTION_CHECKPOINT)

    reaction_config, pk_config, payload = load_reaction_checkpoint_configs(reaction_ckpt_path)

    _, integrator, _, _, _ = load_drift_checkpoint(drift_ckpt_path, device=device)
    device = next(integrator.parameters()).device
    integrator.pk_config = pk_config

    reaction_model = ReactionCoupledModel(
        drift_integrator=integrator,
        q_in=payload["state_dict"]["q_in"],
        config=reaction_config,
    ).to(device)
    reaction_model.drift_integrator.pk_config = copy.deepcopy(pk_config)
    reaction_model.load_state_dict(payload["state_dict"], strict=True)
    reaction_model.eval()
    reaction_model.loaded_reaction_checkpoint_path = reaction_ckpt_path
    reaction_model.requires_grad_(False)
    return reaction_model, device
