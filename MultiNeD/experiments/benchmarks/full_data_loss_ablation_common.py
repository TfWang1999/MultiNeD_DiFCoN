from __future__ import annotations

from typing import Any

import numpy as np
import torch
import pandas as pd

from common.cd8_zdist_data import mouse_balanced_sample
from training.drift_training import load_drift_checkpoint
from training.coupled_dynamics import (
    distribution_metrics,
    reaction_rollout_evaluation,
)

from .dynamics_benchmark_common import PROJECT_ROOT, project_path, set_seed


def _mean_volume_at(volume: pd.DataFrame, treatment: str, day: float) -> float:
    part = volume[volume["treatment_clean"].astype(str) == str(treatment)]
    means = part.groupby("day")["tumor_volume"].mean().sort_index()
    if means.empty:
        return 100.0
    return float(np.interp(float(day), means.index.to_numpy(float), means.to_numpy(float)))


def load_notebook_pretrained_drift(base_config: dict[str, Any]) -> dict[str, Any]:
    """Load the exact drift checkpoint used by 02_calibrate_MultiNeD.ipynb."""
    source = project_path(base_config["inputs"]["drift_checkpoint"], PROJECT_ROOT)
    _, integrator, _, _, _ = load_drift_checkpoint(
        source,
        device=base_config.get("drift", {}).get("device", "auto"),
    )
    return {
        "integrator": integrator,
        "initialization_mode": "notebook_pretrained_checkpoint",
    }


def load_notebook_q_in(base_config: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Load the fixed q_in array saved by 02_calibrate_MultiNeD.ipynb."""
    value = base_config["inputs"].get("q_in_npy")
    if not value:
        raise KeyError("The notebook-aligned ablation requires inputs.q_in_npy")
    source = project_path(value, PROJECT_ROOT)
    if not source.exists():
        raise FileNotFoundError(source)
    q_in = np.asarray(np.load(source), dtype=np.float32)
    if q_in.ndim != 2:
        raise ValueError(f"Saved notebook q_in must be 2D, got shape {q_in.shape}")
    return q_in, {
        "source": str(source),
        "source_mode": "fixed_notebook_empirical_distribution",
        "notebook_random_state": 2026,
        "n_particles": int(q_in.shape[0]),
        "z_dim": int(q_in.shape[1]),
    }


def evaluate_full_data_model(
    trained: dict[str, Any],
    group_data: dict,
    transitions: pd.DataFrame,
    volume: pd.DataFrame,
    evaluation: dict[str, Any],
    seed: int,
    variant: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate every variant on common full-data transitions and anchored rollouts."""
    # Use the same evaluation random stream regardless of each variant's training.
    set_seed(seed)
    integrator = trained["integrator"]
    reaction_model = trained["reaction_model"]
    integrator.eval()
    reaction_model.eval()
    device = next(reaction_model.parameters()).device
    n_particles = int(evaluation["n_particles"])
    z_cols = list(next(iter(group_data.values()))["z_cols"])
    transform = reaction_model.config.z_transform_params
    rng = np.random.default_rng(int(seed))
    transition_rows: list[dict[str, Any]] = []
    for transition_index, row in enumerate(transitions.itertuples(index=False)):
        source_meta = group_data[str(row.source_group)]
        target_meta = group_data[str(row.target_group)]
        source, _ = mouse_balanced_sample(
            group_data,
            str(row.source_group),
            n_max=n_particles,
            random_state=int(rng.integers(0, 2**31 - 1)),
        )
        target, _ = mouse_balanced_sample(
            group_data,
            str(row.target_group),
            n_max=n_particles,
            random_state=int(rng.integers(0, 2**31 - 1)),
        )
        source_t = torch.as_tensor(source, dtype=torch.float32, device=device)
        with torch.no_grad():
            drift_prediction = integrator(
                source_t,
                P=float(row.P),
                B=float(row.B),
                day0=float(row.day0),
                dt=float(row.dt),
            ).detach().cpu().numpy()
            reaction_prediction = reaction_model.simulate(
                source_t,
                P=float(row.P),
                B=float(row.B),
                day0=float(row.day0),
                time_points=[float(row.day1)],
                volume0=_mean_volume_at(
                    volume, str(target_meta["treatment_clean"]), float(row.day0)
                ),
                n0=None,
            )[-1]
        for model_level, prediction, weights in (
            ("drift", drift_prediction, None),
            ("reaction", reaction_prediction["z"].cpu().numpy(), reaction_prediction["particle_weights"]),
        ):
            record = {
                "evaluation_scope": "full_data_in_sample",
                "variant": variant,
                "seed": int(seed),
                "model_level": model_level,
                "transition_id": str(row.transition_id),
                "source_group": str(row.source_group),
                "target_group": str(row.target_group),
                "treatment_clean": str(target_meta["treatment_clean"]),
                "source_day": float(source_meta["day"]),
                "target_day": float(target_meta["day"]),
            }
            record.update(
                distribution_metrics(
                    prediction,
                    target,
                    z_cols,
                    transform,
                    evaluation,
                    seed=int(seed) + transition_index,
                    predicted_weights=weights,
                )
            )
            transition_rows.append(record)

    rollout_parts, volume_parts = [], []
    treatments = sorted(
        {str(group["treatment_clean"]) for group in group_data.values()}
        | set(volume["treatment_clean"].dropna().astype(str))
    )
    for treatment_index, treatment in enumerate(treatments):
        group_days = [
            float(group["day"])
            for group in group_data.values()
            if str(group["treatment_clean"]) == treatment
        ]
        volume_days = volume.loc[
            volume["treatment_clean"].astype(str) == treatment, "day"
        ].astype(float).tolist()
        result = reaction_rollout_evaluation(
            reaction_model,
            group_data,
            volume,
            treatment_clean=treatment,
            config=reaction_model.config,
            n_particles=n_particles,
            random_state=int(seed) + treatment_index,
            time_points=sorted(set(group_days + volume_days)),
        )
        rollout = result["ot"].copy()
        if not rollout.empty:
            rollout = rollout.rename(columns={"ot_loss": "sinkhorn_divergence"})
            rollout["evaluation_scope"] = "full_data_in_sample"
            rollout["variant"] = variant
            rollout["seed"] = int(seed)
            rollout_parts.append(rollout)
        volume_table = result["volume"].copy()
        predicted = volume_table[volume_table["kind"] == "predicted"][
            ["treatment_clean", "day", "volume"]
        ].rename(columns={"volume": "predicted_volume"})
        observed = (
            volume_table[volume_table["kind"] == "observed"]
            .groupby(["treatment_clean", "day"], as_index=False)["volume"]
            .agg(observed_mean_volume="mean", observed_sd_volume="std", n_observed="count")
        )
        matched = predicted.merge(observed, on=["treatment_clean", "day"], how="inner")
        if not matched.empty:
            matched["absolute_error"] = (
                matched["predicted_volume"] - matched["observed_mean_volume"]
            ).abs()
            matched["squared_error"] = (
                matched["predicted_volume"] - matched["observed_mean_volume"]
            ).pow(2)
            matched["evaluation_scope"] = "full_data_in_sample"
            matched["variant"] = variant
            matched["seed"] = int(seed)
            volume_parts.append(matched)
    return (
        pd.DataFrame(transition_rows),
        pd.concat(rollout_parts, ignore_index=True) if rollout_parts else pd.DataFrame(),
        pd.concat(volume_parts, ignore_index=True) if volume_parts else pd.DataFrame(),
    )


__all__ = [
    "evaluate_full_data_model",
    "load_notebook_pretrained_drift",
    "load_notebook_q_in",
]
