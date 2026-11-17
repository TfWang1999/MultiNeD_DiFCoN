"""Table S1: reconstruct B16 distributions with two existing checkpoints.

The complete MultiNeD simulation includes volume, cell abundance and q_in.
Both models share mouse-balanced initial/target samples and robust01 scaling.
"""
import copy
from pathlib import Path

import pandas as pd
import torch

from common.cd8_zdist_data import load_training_package, get_z_columns, mouse_balanced_sample
from training.drift_training import load_drift_checkpoint
from training.coupled_dynamics import (
    ReactionCoupledModel, load_reaction_checkpoint_configs,
    build_z_transform_params, distribution_metrics, transform_z_torch,
)

from .reproduce.upstream_paths import DEFAULT_DRIFT_CHECKPOINT, DEFAULT_REACTION_CHECKPOINT, OBSERVED_VOLUME


ROOT = Path(__file__).resolve().parents[2]
SEEDS = (20260816,)
TREATMENTS = ("Control", "Anti-PD1", "Anti-41BB", "Anti-PD1+Anti-41BB")
METRICS = ("debiased_transport_cost", "energy_distance", "program_mean_mae", "wasserstein_1")
N_PARTICLES = 512


def run(output, device="auto", drift_checkpoint=None, reaction_checkpoint=None):
    drift_checkpoint = Path(drift_checkpoint or ROOT / DEFAULT_DRIFT_CHECKPOINT)
    reaction_checkpoint = Path(reaction_checkpoint or ROOT / DEFAULT_REACTION_CHECKPOINT)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    groups, _, _, _ = load_training_package(ROOT)
    columns = get_z_columns(groups)
    scaling = build_z_transform_params(groups, columns, q_low=0.01, q_high=0.99)
    _, drift, _, _, _ = load_drift_checkpoint(drift_checkpoint, device=device)
    device = next(drift.parameters()).device
    config, pk, payload = load_reaction_checkpoint_configs(reaction_checkpoint)
    coupled_drift = copy.deepcopy(drift)
    coupled_drift.pk_config = copy.deepcopy(pk)
    reaction = ReactionCoupledModel(coupled_drift, q_in=payload["state_dict"]["q_in"], config=config).to(device)
    reaction.load_state_dict(payload["state_dict"], strict=True)
    drift.eval()
    reaction.eval()
    volume = pd.read_csv(ROOT / OBSERVED_VOLUME)
    evaluation = {
        "z_scale": None,
        "sinkhorn_epsilon": 0.08,
        "sinkhorn_iters": 60,
        "energy_max_points": N_PARTICLES,
    }
    rows = []
    with torch.inference_mode():
        for seed in SEEDS:
            initial, _ = mouse_balanced_sample(groups, "day1__Control", N_PARTICLES, seed)
            z0 = torch.as_tensor(initial, dtype=torch.float32, device=device)
            for arm, treatment in enumerate(TREATMENTS):
                targets = sorted(((key, g) for key, g in groups.items()
                                  if g["treatment_clean"] == treatment and float(g["day"]) > 1),
                                 key=lambda item: float(item[1]["day"]))
                times = [float(g["day"]) for _, g in targets]
                P, B = targets[0][1]["P"], targets[0][1]["B"]
                volume0 = volume.loc[volume.treatment_clean.eq(treatment) & volume.day.eq(1), "tumor_volume"].mean()
                torch.manual_seed(seed + arm)
                simulation = reaction.simulate(z0.clone(), P, B, 1.0, times, volume0=volume0)
                for (group_id, group), rec in zip(targets, simulation):
                    day = float(group["day"])
                    z = drift(z0.clone(), P=P, B=B, day0=1.0, dt=day-1.0)
                    target, _ = mouse_balanced_sample(groups, group_id, N_PARTICLES, seed)
                    y = transform_z_torch(torch.as_tensor(target, device=device), columns, "robust01", scaling)
                    for label, particles, weights in (
                        ("Pretrained model", z, None), ("MultiNeD", rec["z"], rec["particle_weights"]),
                    ):
                        x = transform_z_torch(particles, columns, "robust01", scaling)
                        scores = distribution_metrics(x, y, None, None, evaluation, seed, predicted_weights=weights)
                        rows.append(dict(
                            seed=seed, treatment=treatment, day=day, model=label,
                            debiased_transport_cost=scores["sinkhorn_divergence"],
                            energy_distance=scores["energy_distance"],
                            program_mean_mae=scores["score_mean_mae"],
                            wasserstein_1=scores["wasserstein_1"],
                        ))
                print(f"Evaluated seed={seed} treatment={treatment}: {len(times)} time points", flush=True)
    detail = pd.DataFrame(rows)
    per_seed = detail.groupby(["seed", "treatment", "model"], sort=False)[list(METRICS)].mean().reset_index()
    overall = per_seed.groupby(["seed", "model"], sort=False)[list(METRICS)].mean().reset_index()
    overall["treatment"] = "Overall"
    per_seed = pd.concat([per_seed, overall], ignore_index=True)
    summary = per_seed.groupby(["treatment", "model"], sort=False)[list(METRICS)].mean().reset_index()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    detail.to_csv(output / "distribution_metrics.csv", index=False)
    summary.to_csv(output / "reconstruction_summary.csv", index=False)
    print(summary.to_string(index=False))
    return summary
