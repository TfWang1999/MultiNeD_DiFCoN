"""Export the fixed predicted particles used by the downstream density plots.

The model loader and trajectories use the original notebook's sampling settings.
Both particle sampling and stochastic inflow use the fixed export seed.
"""
from dataclasses import replace
from pathlib import Path
import argparse
import json
import sys

import numpy as np
import pandas as pd
import torch

CODE_DIR = Path(__file__).resolve().parents[1]
PKG_DIR = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from common.cd8_zdist_data import get_z_columns, load_training_package
from scripts.reproduce.upstream_paths import (
    DEFAULT_DRIFT_CHECKPOINT, DEFAULT_REACTION_CHECKPOINT, DRIFT_NAME, REACTION_NAME, VOLUME_TABLE, checkpoint_path,
)
from training.drift_training import load_drift_checkpoint
from training.coupled_dynamics import (
    ReactionCoupledModel, load_reaction_checkpoint_configs,
    _build_reaction_training_context,
)

TREATMENT_ORDER = ["Control", "Anti-PD1", "Anti-41BB", "Anti-PD1+Anti-41BB"]
BASELINE_N_TIME = 121
BASELINE_N_PARTICLES = 256
BASELINE_RANDOM_STATE = 6200


def load_reaction_model(drift_checkpoint, reaction_checkpoint, device):
    """Restore the fitted model and its saved inflow particles."""
    print("Loading drift and reaction checkpoints...")
    _, integrator, _, _, _ = load_drift_checkpoint(drift_checkpoint, device=device)
    device = next(integrator.parameters()).device
    reaction_config, pk_config, payload = load_reaction_checkpoint_configs(reaction_checkpoint)
    reaction_config.train_drift = False
    reaction_config.print_every = 0
    integrator.pk_config = pk_config
    reaction_model = ReactionCoupledModel(
        drift_integrator=integrator, q_in=payload["state_dict"]["q_in"], config=reaction_config,
    ).to(device)
    reaction_model.load_state_dict(payload["state_dict"], strict=True)
    reaction_model.eval()
    return reaction_model


def simulate_expression_trajectories(
    model,
    group_data,
    volume_df,
    initial_volume,
):
    """Simulate the program-score particles used by the supplementary density plot."""
    device = next(model.parameters()).device
    config = replace(model.config, n_particles=BASELINE_N_PARTICLES)
    z_cols = get_z_columns(group_data)
    particle_frames = []
    torch.manual_seed(BASELINE_RANDOM_STATE)

    for t_i, treatment in enumerate(TREATMENT_ORDER):
        context = _build_reaction_training_context(
            group_data,
            volume_df,
            treatment,
            config,
            device,
            BASELINE_RANDOM_STATE + t_i,
        )

        anchor = context["anchor"]
        t0 = float(anchor["day"])
        t1 = float(max(context["target_days"]))
        run_time_points = np.linspace(t0, t1, BASELINE_N_TIME)
        run_time_points = np.array(sorted(set(np.round(run_time_points, 10))), dtype=float)
        run_time_points = run_time_points[run_time_points >= float(anchor["day"])]

        with torch.no_grad():
            sim = model.simulate(
                context["z0"],
                P=anchor["P"],
                B=anchor["B"],
                day0=anchor["day"],
                time_points=run_time_points,
                volume0=initial_volume,
                n0=None,
            )

        for rec in sim:
            frame = pd.DataFrame(rec["z"].cpu().numpy(), columns=z_cols)
            frame.insert(0, "treatment_clean", treatment)
            frame.insert(0, "time", float(rec["time"]))
            frame.insert(0, "kind", "predicted")
            frame["particle_weight"] = rec["particle_weights"].cpu().numpy()
            particle_frames.append(frame)

    return pd.concat(particle_frames, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional separate export destination")
    parser.add_argument("--drift-source", help="Checkpoint or exact drift run/result directory")
    parser.add_argument("--reaction-source", help="Checkpoint or exact joint run/result directory")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    drift_checkpoint = (
        checkpoint_path(args.drift_source, DRIFT_NAME, PKG_DIR)
        if args.drift_source else PKG_DIR / DEFAULT_DRIFT_CHECKPOINT
    )
    reaction_checkpoint = (
        checkpoint_path(args.reaction_source, REACTION_NAME, PKG_DIR)
        if args.reaction_source else PKG_DIR / DEFAULT_REACTION_CHECKPOINT
    )
    default_destination = CODE_DIR / "figure_work/source_data"
    destination = args.output_dir.resolve() if args.output_dir else default_destination
    output_path = destination / "downstream_trajectory_particles.csv"
    volume_path = reaction_checkpoint.parent / VOLUME_TABLE
    required = (drift_checkpoint, reaction_checkpoint, volume_path)
    for path in required:
        if not path.is_file():
            parser.error(f"Missing input: {path}")
    settings = {"random_state": BASELINE_RANDOM_STATE, "n_time": BASELINE_N_TIME,
                "n_particles": BASELINE_N_PARTICLES,
                "initial_volume": "Control day-1 mean"}
    group_data, _, _, _ = load_training_package(PKG_DIR)
    volume_df = pd.read_csv(volume_path)
    reaction_model = load_reaction_model(drift_checkpoint, reaction_checkpoint, device=args.device)
    settings["n_inflow_particles"] = reaction_model.config.n_inflow_particles
    control_day1 = volume_df.loc[
        volume_df["treatment_clean"].astype(str).eq("Control")
        & np.isclose(volume_df["day"].astype(float), 1.0), "tumor_volume"
    ].dropna()
    if control_day1.empty:
        raise ValueError("No Control day-1 tumour volume observations.")
    common_c0 = float(control_day1.mean())
    trajectory = simulate_expression_trajectories(
        reaction_model,
        group_data=group_data,
        volume_df=volume_df,
        initial_volume=common_c0,
    )
    destination.mkdir(parents=True, exist_ok=True)
    trajectory.to_csv(output_path, index=False)
    print(json.dumps({"files": [str(output_path)], "settings": settings,
                      "rows": {"particles": len(trajectory)}}, indent=2))


if __name__ == "__main__":
    main()
