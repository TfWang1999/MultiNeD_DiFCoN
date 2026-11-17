"""E2a: compare frozen closed-loop policies with validation-derived open loop.

For every policy seed, this driver *only* averages actions observed on the
frozen validation cohort.  The locked test cohort is never read while making
the open-loop schedule.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import sys
from pathlib import Path


import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
SCRIPTS_DIR = HERE.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from _direct_dose_evaluation_utils import (
    load_frozen_cohort,
    load_saved_policy_context,
    make_output_dir,
    make_specs,
    resolve_from_root,
)
from core.io import write_json  # noqa: E402
from core.policy_common import FixedSchedulePolicy, fixed_regimen_policies  # noqa: E402
from core.training import action_frame, evaluate_strategies, evaluation_rollout, trajectory_frame  # noqa: E402

DEFAULT_RUN_ROOT = "outputs/dose_weight_sweep"
DEFAULT_OUTPUT = "outputs/openloop_comparison"
DEFAULT_COHORT = "cohorts"
SEEDS = (2031, 2032, 2033)
STRATEGY_ORDER = [
    "No treatment",
    "Fixed Combine",
    "Validation-derived open-loop",
    "Closed-loop",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="E2a: frozen lambda_d=1 closed-loop versus validation-derived open-loop control."
    )
    parser.add_argument("--run-root", type=Path, default=Path(DEFAULT_RUN_ROOT))
    parser.add_argument("--cohort-dir", type=Path, default=Path(DEFAULT_COHORT))
    parser.add_argument("--output-dir", type=Path, default=Path(DEFAULT_OUTPUT))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--device", default="auto", help="Torch device, for example cpu or cuda:0.")
    parser.add_argument("--n-particles", type=int, default=96, help="Particles per patient; the paper uses 96.")
    return parser.parse_args()


def run_dir_for_seed(run_root: Path, seed: int) -> Path:
    path = run_root / f"dose_weight_1p0_seed_{int(seed)}"
    if not path.exists():
        raise FileNotFoundError(f"Frozen lambda_d=1 run not found: {path}")
    return path


def main() -> None:
    args = parse_args()
    n_particles = args.n_particles
    if n_particles < 1:
        raise ValueError("--n-particles must be >= 1.")
    output_dir = make_output_dir(args.output_dir)
    run_root = resolve_from_root(args.run_root)
    cohort_dir = resolve_from_root(args.cohort_dir)
    validation_cohort, _ = load_frozen_cohort(cohort_dir, "validation")
    test_cohort, _ = load_frozen_cohort(cohort_dir, "test")

    schedule_parts: list[pd.DataFrame] = []
    test_summary_parts: list[pd.DataFrame] = []
    test_trajectory_parts: list[pd.DataFrame] = []
    resolved_records: list[dict] = []

    for seed in args.seeds:
        run_dir = run_dir_for_seed(run_root, int(seed))
        context = load_saved_policy_context(run_dir, device=args.device)
        dose_weight = float(context.loss_config.total_dose_weight)
        if not np.isclose(dose_weight, 1.0):
            raise ValueError(f"Expected lambda_d=1 for seed {seed}; checkpoint has {dose_weight}.")
        validation_seed = int(context.seeds["validation_rollout"])
        test_seed = int(context.seeds["test_rollout"])
        validation_specs = make_specs(context, validation_cohort, n_particles, validation_seed)
        test_specs = make_specs(context, test_cohort, n_particles, test_seed)

        validation_actions = pd.concat([
            action_frame(
                spec, "Closed-loop",
                evaluation_rollout(context.reaction_model, context.policy, spec, context.rollout_config),
            )
            for spec in validation_specs
        ], ignore_index=True)
        schedule = (
            validation_actions.groupby("day", observed=True)[["action_PD1", "action_B41BB", "daily_total_dose"]]
            .mean()
            .reset_index()
        )
        schedule.insert(0, "model_seed", int(seed))
        schedule.insert(1, "schedule_source", "validation_closed_loop_mean")
        schedule.insert(2, "n_validation_patients", int(validation_cohort["patient_id"].nunique()))
        schedule_parts.append(schedule)
        open_loop = FixedSchedulePolicy(
            schedule[["action_PD1", "action_B41BB"]].to_numpy(dtype=np.float32)
        ).to(context.device)
        strategies = fixed_regimen_policies(
            context.rollout_config.decision_days(),
            {"no_treatment": "No treatment", "combine": "Fixed Combine"},
            context.device,
        )
        strategies["Validation-derived open-loop"] = open_loop
        strategies["Closed-loop"] = context.policy
        summary_rows, trajectory_parts = [], []
        for summary, trajectory in evaluate_strategies(
            context.reaction_model,
            strategies,
            test_specs,
            context.rollout_config,
            context.loss_config,
        ):
            summary_rows.append(summary)
            trajectory_parts.append(trajectory_frame(summary, summary["strategy"], trajectory))
        summary = pd.DataFrame(summary_rows)
        trajectories = pd.concat(trajectory_parts, ignore_index=True)
        for frame in (summary, trajectories):
            frame.insert(0, "model_seed", int(seed))
        test_summary_parts.append(summary)
        test_trajectory_parts.append(trajectories)
        resolved_records.append(
            {
                "model_seed": int(seed),
                "run_dir": str(run_dir),
                "checkpoint": str(context.checkpoint_path),
                "rollout_config": asdict(context.rollout_config),
                "n_particles": n_particles,
                "validation_rollout_seed": validation_seed,
                "test_rollout_seed": test_seed,
            }
        )
        print(f"E2a seed {seed}: completed ({len(test_specs)} locked test patients).")


    schedule_df = pd.concat(schedule_parts, ignore_index=True)
    summary_df = pd.concat(test_summary_parts, ignore_index=True)
    trajectory_df = pd.concat(test_trajectory_parts, ignore_index=True)
    schedule_df.to_csv(output_dir / "validation_derived_openloop_schedule.csv", index=False)
    summary_df.to_csv(output_dir / "test_patient_level_results_long.csv", index=False)
    trajectory_df.to_csv(output_dir / "test_trajectories.csv", index=False)

    resolved = {
        "experiment_id": "E2a",
        "description": "Closed-loop vs validation-derived open-loop; test cohort was not used to construct schedules.",
        "lambda_d": 1.0,
        "seeds_requested": [int(seed) for seed in args.seeds],
        "seeds_completed": sorted(summary_df["model_seed"].unique().tolist()),
        "cohort_dir": str(cohort_dir),
        "n_particles": int(args.n_particles),
        "strategy_order": STRATEGY_ORDER,
        "per_seed": resolved_records,
    }
    write_json(output_dir / "resolved_evaluation_config.json", resolved)
    print(f"E2a outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
