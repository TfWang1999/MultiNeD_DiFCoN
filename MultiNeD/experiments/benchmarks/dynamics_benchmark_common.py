from __future__ import annotations

import copy
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import pandas as pd


CODE_DIR = Path(__file__).resolve().parents[2]
PROJECT_ROOT = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from common.cd8_zdist_data import mouse_balanced_sample  # noqa: E402
from training.drift_training import (  # noqa: E402
    load_drift_checkpoint,
    save_drift_checkpoint,
    train_drift_model,
)
from training.coupled_dynamics import (  # noqa: E402
    ReactionCoupledModel,
    apply_reaction_parameter_settings,
    build_z_transform_params,
    load_reaction_checkpoint_configs,
    save_reaction_checkpoint,
    train_reaction_coupled_model,
)
from experiments.volume_loto.config import project_path
from experiments.volume_loto.model_runner import _apply_overrides, load_parameter_settings, _json_default as json_default, set_seed  # noqa: E402


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()




def write_json(path: str | Path, value: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=json_default) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return path




def environment_record(device: str) -> dict[str, Any]:
    """Small environment record; intentionally contains no file hashes."""
    return {
        "created_at_utc": utc_now(),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "requested_device": str(device),
        "gpu_names": [
            torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
        ] if torch.cuda.is_available() else [],
    }


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    config = json.loads(path.read_text(encoding="utf-8"))
    config["config_source"] = str(path)
    return config


def load_base_validation_config(path: str | Path) -> dict[str, Any]:
    return load_config(project_path(path, PROJECT_ROOT))


def prepare_run_directory(root: str | Path, run_id: str) -> Path:
    path = project_path(root, PROJECT_ROOT) / str(run_id)
    path.mkdir(parents=True, exist_ok=True)
    return path


def fit_full_project_model(
    base_config: dict[str, Any],
    group_data: dict,
    transitions: pd.DataFrame,
    volume: pd.DataFrame,
    output_dir: str | Path,
    seed: int,
    *,
    fixed_q_in: np.ndarray,
    fixed_q_in_info: dict[str, Any],
    drift_overrides: dict[str, Any] | None = None,
    reaction_overrides: dict[str, Any] | None = None,
    initialize_drift_randomly: bool = False,
    prefit_drift: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fit the full-data loss ablation with the saved notebook q_in."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)
    inputs = base_config["inputs"]
    source_drift = project_path(inputs["drift_checkpoint"], PROJECT_ROOT)
    _, _, drift_model_config, drift_train_config, _ = load_drift_checkpoint(source_drift, device="cpu")
    drift_values = copy.deepcopy(base_config.get("drift", {}))
    drift_values.update(copy.deepcopy(drift_overrides or {}))
    drift_values["seed"] = int(seed)
    drift_train_config = _apply_overrides(drift_train_config, drift_values, "drift")

    if prefit_drift is not None:
        integrator = prefit_drift["integrator"]
        drift_initialization = prefit_drift["initialization_mode"]
    elif initialize_drift_randomly:
        from training.drift_training import build_drift_model

        _, integrator, _ = build_drift_model(
            model_config=drift_model_config,
            train_config=drift_train_config,
            device=drift_train_config.device,
        )
        drift_initialization = "random"
    else:
        drift_model, integrator, drift_history = train_drift_model(
            group_data,
            transitions,
            model_config=drift_model_config,
            train_config=drift_train_config,
        )
        save_drift_checkpoint(
            output_dir / "drift_model.pt",
            drift_model,
            drift_model_config,
            drift_train_config,
            history_df=drift_history,
        )
        drift_initialization = "full_data_or_split_refit"

    reference_reaction = project_path(inputs["reaction_checkpoint"], PROJECT_ROOT)
    reaction_config, pk_config, _ = load_reaction_checkpoint_configs(reference_reaction, map_location="cpu")
    reaction_values = copy.deepcopy(base_config.get("reaction", {}))
    reaction_values.update(copy.deepcopy(reaction_overrides or {}))
    reaction_config = _apply_overrides(reaction_config, reaction_values, "reaction")
    reaction_config.z_transform_params = build_z_transform_params(
        group_data,
        q_low=float(base_config["evaluation"].get("robust_quantiles", [0.01, 0.99])[0]),
        q_high=float(base_config["evaluation"].get("robust_quantiles", [0.01, 0.99])[1]),
    )
    integrator.pk_config = pk_config
    q_in = np.asarray(fixed_q_in, dtype=np.float32).copy()
    if q_in.ndim != 2:
        raise ValueError(f"fixed_q_in must be a 2D array, got shape {q_in.shape}")
    expected_z_dim = len(next(iter(group_data.values()))["z_cols"])
    if q_in.shape[1] != expected_z_dim:
        raise ValueError(f"fixed_q_in has {q_in.shape[1]} columns; expected {expected_z_dim}")
    q_in_info = copy.deepcopy(fixed_q_in_info)
    q_in_info["source_mode"] = "fixed_notebook_empirical_distribution"
    q_in_info["n_particles"] = int(q_in.shape[0])
    q_in_info["z_dim"] = int(q_in.shape[1])
    device = next(integrator.parameters()).device
    model = ReactionCoupledModel(integrator, q_in=q_in, config=reaction_config).to(device)
    settings = load_parameter_settings(base_config, PROJECT_ROOT)
    applied = apply_reaction_parameter_settings(model, settings, config=reaction_config)
    history = train_reaction_coupled_model(
        model,
        group_data,
        volume,
        treatment_clean="all",
        config=reaction_config,
        random_state=int(seed),
    )
    save_reaction_checkpoint(
        output_dir / "reaction_model.pt", model, history_df=history, q_in_info=q_in_info
    )
    write_json(output_dir / "applied_parameter_settings.json", applied)
    write_json(output_dir / "model_sources.json", {
        "source_drift_checkpoint": str(source_drift),
        "source_reaction_checkpoint": str(reference_reaction),
        "drift_random_initialization": bool(initialize_drift_randomly),
        "prefit_drift_supplied": bool(prefit_drift is not None),
        "drift_initialization": drift_initialization,
        "reaction_trains_drift": bool(reaction_config.train_drift),
        "q_in_source_mode": q_in_info.get("source_mode"),
        "seed": int(seed),
    })
    return {"integrator": integrator, "reaction_model": model}


def sample_group(group_data: dict, group_id: str, n: int, seed: int) -> np.ndarray:
    values, _ = mouse_balanced_sample(group_data, group_id, n_max=int(n), random_state=int(seed))
    return values


__all__ = [
    "CODE_DIR",
    "PROJECT_ROOT",
    "environment_record",
    "fit_full_project_model",
    "load_base_validation_config",
    "load_config",
    "prepare_run_directory",
    "project_path",
    "sample_group",
    "set_seed",
    "utc_now",
    "write_json",
]
