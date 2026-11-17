from __future__ import annotations

import copy
import json
import platform
import random
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import pandas as pd


CODE_DIR = Path(__file__).resolve().parents[2]
PROJECT_ROOT = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from common.cd8_zdist_data import (
    DEFAULT_CELL_SCORE_CSV,
    load_raw_group_scores,
    load_training_package,
    mouse_balanced_sample,
)
from training.drift_training import (
    load_drift_checkpoint,
    save_drift_checkpoint,
    train_drift_model,
)
from training.coupled_dynamics import (
    ReactionCoupledModel,
    apply_reaction_parameter_settings,
    build_q_in_distribution,
    build_z_transform_params,
    load_reaction_checkpoint_configs,
    save_reaction_checkpoint,
    train_reaction_coupled_model,
)
from experiments.volume_loto.config import project_path
from experiments.volume_loto.splits import HoldoutSplit


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if is_dataclass(value):
        return asdict(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def write_json(path: str | Path, value: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    return path


def package_versions() -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def load_inputs(
    config: dict[str, Any],
    project_root: str | Path = PROJECT_ROOT,
    *,
    raw_scores: bool = False,
) -> dict[str, Any]:
    inputs = config["inputs"]
    group_data, group_info, transitions, paths = load_training_package(
        project_root,
        group_pkl=project_path(inputs["group_pkl"], project_root),
        group_info_csv=project_path(inputs["group_info_csv"], project_root),
        transition_csv=project_path(inputs["transition_csv"], project_root),
    )
    if raw_scores:
        score_path = project_path(inputs.get("cell_score_csv", DEFAULT_CELL_SCORE_CSV), project_root)
        group_data = load_raw_group_scores(group_data, score_path)
        paths["cell_score_csv"] = score_path
    volume = pd.read_csv(project_path(inputs["volume_csv"], project_root))
    return {
        "group_data": group_data,
        "group_info": group_info,
        "transitions": transitions,
        "volume": volume,
        "paths": {**paths, "volume_csv": project_path(inputs["volume_csv"], project_root)},
    }


def _apply_overrides(dataclass_obj: Any, overrides: dict[str, Any], label: str) -> Any:
    result = copy.deepcopy(dataclass_obj)
    fields = set(getattr(result, "__dataclass_fields__", {}))
    unknown = sorted(set(overrides) - fields)
    if unknown:
        raise KeyError(f"Unknown {label} config fields: {unknown}")
    for key, value in overrides.items():
        setattr(result, key, copy.deepcopy(value))
    return result


def load_parameter_settings(config: dict[str, Any], project_root: str | Path = PROJECT_ROOT) -> dict[str, Any]:
    path = project_path(config["inputs"]["reaction_parameter_settings"], project_root)
    payload = json.loads(path.read_text(encoding="utf-8"))
    settings = copy.deepcopy(payload["parameter_settings"])
    # The supplied historical settings initialize the abundance scale at one.
    settings.pop("initial_n", None)
    return settings


def train_fresh_split_model(
    config: dict[str, Any],
    split: HoldoutSplit,
    output_dir: str | Path,
    seed: int,
    project_root: str | Path = PROJECT_ROOT,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs = config["inputs"]
    set_seed(seed)

    source_drift_path = project_path(inputs["drift_checkpoint"], project_root)
    refit_drift = bool(split.heldout_groups)
    if refit_drift:
        _, _, drift_model_config, drift_train_config, _ = load_drift_checkpoint(
            source_drift_path,
            device="cpu",
        )
        drift_overrides = copy.deepcopy(config["drift"])
        drift_overrides["seed"] = int(seed)
        drift_train_config = _apply_overrides(drift_train_config, drift_overrides, "drift")
        drift_model, integrator, drift_history = train_drift_model(
            split.train_group_data,
            split.train_transitions,
            model_config=drift_model_config,
            train_config=drift_train_config,
        )
        drift_path = save_drift_checkpoint(
            output_dir / "drift_model.pt",
            drift_model,
            drift_model_config,
            drift_train_config,
            history_df=drift_history,
        )
    else:
        _, integrator, _, _, _ = load_drift_checkpoint(
            source_drift_path,
            device=config["drift"].get("device", "auto"),
        )
        drift_path = source_drift_path
    write_json(output_dir / "drift_provenance.json", {
        "strategy": "matching_z_only",
        "refit": refit_drift,
        "heldout_z_groups": list(split.heldout_groups),
        "source_checkpoint": str(source_drift_path),
        "effective_checkpoint": str(drift_path),
    })

    reference_reaction = project_path(inputs["reaction_checkpoint"], project_root)
    reaction_config, pk_config, _ = load_reaction_checkpoint_configs(reference_reaction, map_location="cpu")
    reaction_config = _apply_overrides(reaction_config, config["reaction"], "reaction")
    reaction_config.z_transform_params = build_z_transform_params(
        split.train_group_data,
        q_low=float(config["evaluation"]["robust_quantiles"][0]),
        q_high=float(config["evaluation"]["robust_quantiles"][1]),
    )
    integrator.pk_config = pk_config
    q_in, q_in_info = build_q_in_distribution(
        split.train_group_data,
        random_state=int(seed),
    )
    device = next(integrator.parameters()).device
    set_seed(seed)
    reaction_model = ReactionCoupledModel(integrator, q_in=q_in, config=reaction_config).to(device)
    parameter_settings = load_parameter_settings(config, project_root)
    applied = apply_reaction_parameter_settings(reaction_model, parameter_settings, config=reaction_config)
    reaction_history = train_reaction_coupled_model(
        reaction_model,
        split.train_group_data,
        split.train_volume,
        treatment_clean="all",
        config=reaction_config,
        random_state=int(seed),
    )
    save_reaction_checkpoint(
        output_dir / "reaction_model.pt",
        reaction_model,
        history_df=reaction_history,
        q_in_info=q_in_info,
    )
    write_json(output_dir / "applied_parameter_settings.json", applied)
    return {"integrator": integrator, "reaction_model": reaction_model}


def _target_groups_for_split(split: HoldoutSplit, group_data: dict) -> list[str]:
    return sorted(split.heldout_groups, key=lambda group_id: (float(group_data[group_id]["day"]), group_id))


def evaluate_heldout_split(
    config: dict[str, Any],
    split: HoldoutSplit,
    full_group_data: dict,
    full_volume: pd.DataFrame,
    trained: dict[str, Any],
    seed: int,
) -> pd.DataFrame:
    # Fix cell sampling and q_in draws independently of the preceding training.
    set_seed(seed)
    integrator = trained["integrator"]
    reaction_model = trained["reaction_model"]
    device = next(integrator.parameters()).device
    n_particles = int(config["evaluation"]["n_particles"])
    rng = np.random.default_rng(int(seed))

    control_candidates = sorted(
        (
            (float(group["day"]), group_id)
            for group_id, group in split.train_group_data.items()
            if group["treatment_clean"] == "Control"
        )
    )
    if not control_candidates:
        raise ValueError("M1 requires a Control z-distribution anchor.")
    control_group_id = control_candidates[0][1]

    # Preserve the cell and q_in draws that precede the volume prediction.
    for target_group_id in _target_groups_for_split(split, full_group_data):
        target = full_group_data[target_group_id]
        source = split.train_group_data[control_group_id]
        source_z, _ = mouse_balanced_sample(
            split.train_group_data,
            control_group_id,
            n_max=n_particles,
            random_state=int(rng.integers(0, 2**31 - 1)),
        )
        mouse_balanced_sample(
            full_group_data,
            target_group_id,
            n_max=n_particles,
            random_state=int(rng.integers(0, 2**31 - 1)),
        )
        with torch.no_grad():
            reaction_model.simulate(
                torch.as_tensor(source_z, dtype=torch.float32, device=device),
                P=float(target["P"]),
                B=float(target["B"]),
                day0=float(source["day"]),
                time_points=[float(target["day"])],
                volume0=100.0,
                n0=None,
            )

    volume_rows = []
    for treatment in split.heldout_treatments:
        treatment_groups = [
            group
            for group in full_group_data.values()
            if group["treatment_clean"] == treatment
        ]
        if not treatment_groups:
            continue
        target_meta = treatment_groups[0]
        target_days = sorted(split.heldout_days)
        init_part = split.train_volume[split.train_volume["treatment_clean"] == treatment]
        if init_part.empty:
            continue
        init_day = float(init_part["day"].astype(float).min())
        init_volume = float(init_part.loc[init_part["day"].astype(float) == init_day, "tumor_volume"].mean())
        z0, _ = mouse_balanced_sample(
            split.train_group_data,
            control_group_id,
            n_max=n_particles,
            random_state=int(rng.integers(0, 2**31 - 1)),
        )
        z0_tensor = torch.as_tensor(z0, dtype=torch.float32, device=device)
        valid_days = [day for day in target_days if day >= init_day]
        with torch.no_grad():
            sim = reaction_model.simulate(
                z0_tensor,
                P=float(target_meta["P"]),
                B=float(target_meta["B"]),
                day0=init_day,
                time_points=valid_days,
                volume0=init_volume,
                n0=None,
            )
        pred_by_day = {float(rec["time"]): float(rec["volume"].detach().cpu()) for rec in sim}
        observed = full_volume[full_volume["treatment_clean"] == treatment]
        for target_day in valid_days:
            target_values = observed.loc[observed["day"].astype(float) == float(target_day), "tumor_volume"].dropna()
            if target_values.empty:
                continue
            observed_mean = float(target_values.mean())
            prediction = pred_by_day[float(target_day)]
            volume_rows.append({
                "split_id": split.split_id,
                "holdout_kind": split.kind,
                "treatment_clean": treatment,
                "day": float(target_day),
                "model": "reaction",
                "initial_day": init_day,
                "initial_volume": init_volume,
                "predicted_volume": prediction,
                "observed_mean_volume": observed_mean,
                "absolute_error": abs(prediction - observed_mean),
                "squared_error": (prediction - observed_mean) ** 2,
                "n_observed": int(target_values.shape[0]),
            })
    return pd.DataFrame(volume_rows)
