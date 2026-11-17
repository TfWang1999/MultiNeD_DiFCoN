"""Restore the selected full MultiNeD weights and their model configuration."""
from dataclasses import asdict, fields
import json
import sys

import numpy as np
import torch

PROGRAMS = ["Precursor", "Activation", "Early effector", "Cytotoxic", "Dysfunction"]
SCORE_COLS = ["P_precursor_zstd", "Activation_zstd", "Eff_Early_zstd", "Eff_Cytox_zstd", "Dysf_zstd"]
TREATMENTS = ["Control", "Anti-PD1", "Anti-4-1BB", "Combination"]
ARMS = [(0, 0), (1, 0), (0, 1), (1, 1)]





def load_final_reaction(repo, cfg):
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(repo / "MultiNeD"))
    from common.cd8_model_components import CD8DriftNet, DriftIntegrator, DriftModelConfig
    from training.coupled_dynamics import (
        ReactionCoupledModel, choose_device, load_reaction_checkpoint_configs,
    )

    reaction_path = repo / cfg["reaction_checkpoint"]
    rc, pk, payload = load_reaction_checkpoint_configs(reaction_path, map_location="cpu")
    if not {"state_dict", "reaction_config", "pk_config"}.issubset(payload):
        raise ValueError("A fully configured Reaction checkpoint is required; no default-config fallback.")
    # Reaction checkpoints omit the drift architecture and integration settings.
    # For a selected training run, read those attributes from its matching drift
    # checkpoint; all fitted tensors still come from the full Reaction checkpoint.
    if cfg.get("drift_checkpoint"):
        configuration_source = cfg["drift_checkpoint"]
        drift = torch.load(repo / configuration_source, map_location="cpu", weights_only=False)
        mc = dict(drift["model_config"])
        mc["pk"] = payload["pk_config"]
        train_settings = drift["train_config"]
        tc = {key: train_settings[key] for key in ("n_steps_per_day", "time_scale")}
    else:
        configuration_source = cfg["final_reaction_metadata"]
        metadata = json.loads((repo / configuration_source).read_text(encoding="utf-8"))
        if metadata.get("schema_version") != 1:
            raise ValueError("Unsupported final Reaction metadata schema")
        if metadata["reaction_checkpoint"]["path"] != cfg["reaction_checkpoint"]:
            raise ValueError("Final metadata belongs to a different Reaction checkpoint")
        mc = dict(metadata["drift_model_config"])
        tc = dict(metadata["integrator"])
    if not {f.name for f in fields(DriftModelConfig)}.issubset(mc):
        raise ValueError("Final drift metadata must explicitly specify every model-config field")
    if json.dumps(mc["pk"], sort_keys=True) != json.dumps(payload["pk_config"], sort_keys=True):
        raise ValueError("Final metadata PK does not match the authoritative Reaction payload")
    model_cfg = DriftModelConfig.from_saved_config(mc)
    model_cfg.pk = pk
    if (model_cfg.z_dim, model_cfg.treatment_dim, model_cfg.time_dim) != (5, 2, 1):
        raise ValueError("This analysis requires five ordered programs and two exposure inputs.")
    if set(tc) != {"n_steps_per_day", "time_scale"}:
        raise ValueError("Final metadata must explicitly specify all integrator attributes")
    if (not isinstance(tc["n_steps_per_day"], int) or tc["n_steps_per_day"] <= 0
            or not np.isfinite(tc["time_scale"]) or tc["time_scale"] <= 0):
        raise ValueError("Invalid final integrator metadata")
    integrator = DriftIntegrator(
        CD8DriftNet(model_cfg), n_steps_per_day=tc["n_steps_per_day"],
        time_scale=tc["time_scale"], pk_config=pk,
    )
    # Strict state loading below checks tensor names and shapes.
    state = payload["state_dict"]
    if "q_in" not in state:
        raise ValueError("Reaction checkpoint does not contain its q_in buffer.")
    resolved = choose_device(cfg["device"])
    model = ReactionCoupledModel(integrator, q_in=state["q_in"].numpy(), config=rc).to(resolved)
    model.load_state_dict(state, strict=True)
    model.eval()
    model.requires_grad_(False)
    info = {
        "model_weights_source": cfg["reaction_checkpoint"],
        "checkpoint_loaded_strictly": True,
        "standalone_drift_weights_loaded": False,
        "model_configuration_source": configuration_source,
        "drift_model_config": asdict(model_cfg), "reaction_config": asdict(model.config),
        "pk_config": asdict(model.drift_integrator.pk_config),
        "integrator": {"time_scale": integrator.time_scale,
                       "n_steps_per_day": integrator.n_steps_per_day},
        "killing_weights": model.killing_weights.detach().cpu().tolist(),
        "activity_method": "training.coupled_dynamics.ReactionCoupledModel.effector_activity",
        "positive_reaction_parameters": {k: float(v.detach().cpu()) for k, v in model.positive_params().items()},
        "reaction_effect_state": {k: float(v.detach().cpu()) for k, v in model.reaction_effect.state_dict().items()},
        "device": str(resolved),
    }
    return model, info
