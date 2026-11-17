"""Formal drift pretraining and multiscale joint calibration."""
from dataclasses import asdict
from pathlib import Path
import copy
import json

import numpy as np
import pandas as pd

from common.cd8_zdist_data import load_training_package
from training import drift_training as drift
from training import coupled_dynamics as reaction
from .paper_settings import drift_configs, reaction_configs
from .upstream_paths import (
    DEFAULT_DRIFT_CHECKPOINT, DRIFT_NAME, OBSERVED_VOLUME, PARAMETER_NAME,
    QIN_NAME, REACTION_NAME, VOLUME_TABLE,
)

CODE = Path(__file__).resolve().parents[2]
REPO = CODE.parent


def run(task, destination, device=None, *, drift_checkpoint=None):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    groups, _, transitions, _ = load_training_package(REPO)
    if task == "notebook_drift_run":
        model_config, config = drift_configs()
        if device:
            config.device = device
        model, _, history = drift.train_drift_model(
            groups, transitions, model_config=model_config, train_config=config)
        drift.save_drift_checkpoint(
            destination / DRIFT_NAME, model, model_config, config, history_df=history
        )
        history.to_csv(destination / "drift_training_history.csv", index=False)
        settings = {"model": asdict(model_config), "training": asdict(config), "seed": 2026}
    else:
        volume = pd.read_csv(REPO / OBSERVED_VOLUME)
        q_in, q_info = reaction.build_q_in_distribution(groups)
        np.save(destination / QIN_NAME, q_in)
        volume_path = destination / VOLUME_TABLE
        volume_path.parent.mkdir(parents=True, exist_ok=True)
        volume.to_csv(volume_path, index=False)
        _, integrator, _, _, _ = drift.load_drift_checkpoint(
            drift_checkpoint or REPO / DEFAULT_DRIFT_CHECKPOINT, device=device or "auto"
        )
        config, pk, parameters = reaction_configs(groups)
        if device:
            config.device = device
        integrator.pk_config = pk
        model = reaction.ReactionCoupledModel(copy.deepcopy(integrator), q_in=q_in, config=config).to(
            next(integrator.parameters()).device)
        model.drift_integrator.pk_config = pk
        applied = reaction.apply_reaction_parameter_settings(model, parameters, config=config)
        drift.set_random_seed(2026)
        history = reaction.train_reaction_coupled_model(
            model, groups, volume, treatment_clean="all", config=config, random_state=2026)
        (destination / PARAMETER_NAME).write_text(
            json.dumps({"parameter_settings": parameters, "applied": applied}, indent=2), encoding="utf-8")
        reaction.save_reaction_checkpoint(
            destination / REACTION_NAME, model, history_df=history, q_in_info=q_info
        )
        history.to_csv(destination / "reaction_coupled_history.csv", index=False)
        settings = {"training": asdict(config), "pk": asdict(pk), "qin": q_info,
                    "parameters": parameters, "seed": 2026}
    (destination / "resolved_settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    (destination / "run_status.json").write_text(json.dumps({"status": "completed", "mode": "full"}), encoding="utf-8")
