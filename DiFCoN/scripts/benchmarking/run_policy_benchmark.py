from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts._direct_dose_evaluation_utils import (
    limit_patients,
    load_frozen_cohort,
    load_saved_policy_context,
    make_specs,
    resolve_from_root,
)
from scripts.benchmarking.classical_optimizers import OPTIMIZERS
from scripts.benchmarking.policy_benchmark_common import (
    ArraySchedulePolicy,
    load_config,
)
from scripts.benchmarking.rl_env import load_sac_policy, train_sac
from core.io import write_json
from core.policy_common import fixed_regimen_policies
from core.training import action_frame, evaluate_patient


DEFAULT_CONFIG = ROOT / "configs" / "benchmarking" / "locked_test_benchmark.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run matched-compute direct-dose policy benchmarks.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--cohort", choices=("validation", "test"), default=None)
    parser.add_argument(
        "--confirm-hyperparameters-frozen",
        action="store_true",
        help="Required acknowledgement before the locked test cohort can be evaluated.",
    )
    parser.add_argument("--methods", default=None, help="Comma-separated configured methods.")
    parser.add_argument("--model-seed", type=int, default=None)
    parser.add_argument("--max-patients", type=int, default=None)
    parser.add_argument("--device", default=None)
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--run-id")
    target.add_argument("--output-dir", type=Path, help="Directory for this benchmark run")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--policy-root", type=Path)
    parser.add_argument("--openloop-source", type=Path)
    parser.add_argument("--sac-root", type=Path)
    return parser.parse_args()


def resolve_config(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config)
    if args.output_root:
        config["output_root"] = str(args.output_root)
    if args.policy_root:
        for run in config["policy_runs"]:
            run["run_dir"] = str(args.policy_root / f"dose_weight_1p0_seed_{run['model_seed']}")
    if args.openloop_source:
        config["validation_derived_schedule_csv"] = str(args.openloop_source / "validation_derived_openloop_schedule.csv")
    if args.sac_root:
        sac_root = resolve_from_root(args.sac_root)
        config["sac_checkpoints"] = {}
        for run in config["policy_runs"]:
            seed = run["model_seed"]
            seed_dir = sac_root / f"seed_{seed}"
            checkpoint = seed_dir / f"model_seed_{seed}/sac_actor.pt"
            legacy_checkpoint = seed_dir / "sac_actor.pt"
            if not checkpoint.is_file() and legacy_checkpoint.is_file():
                checkpoint = legacy_checkpoint
            config["sac_checkpoints"][str(seed)] = str(checkpoint)
    if args.cohort is not None:
        config["cohort"] = args.cohort
    if args.confirm_hyperparameters_frozen:
        config["hyperparameters_frozen"] = True
    if args.device is not None:
        config["device"] = args.device
    if args.max_patients is not None:
        config["max_patients"] = int(args.max_patients)
    if args.methods:
        requested = [value.strip() for value in args.methods.split(",") if value.strip()]
        unknown = sorted(set(requested) - set(config["methods"]))
        if unknown:
            raise ValueError(f"Requested methods are not configured: {unknown}")
        config["methods"] = requested
    if args.model_seed is not None:
        matches = [value for value in config["policy_runs"] if int(value["model_seed"]) == args.model_seed]
        if not matches:
            raise ValueError(f"No configured policy run for model seed {args.model_seed}")
        config["policy_runs"] = matches
    if config["cohort"] == "test" and not bool(config.get("hyperparameters_frozen", False)):
        raise ValueError(
            "Locked test evaluation requires hyperparameters_frozen=true after validation-only selection."
        )
    sac_settings = config.get("method_settings", {}).get("sac", {})
    if config["cohort"] == "test" and "sac" in config["methods"]:
        if str(sac_settings.get("mode", "train")).lower() != "load":
            raise ValueError("Locked test requires method_settings.sac.mode='load'.")
        checkpoints = config.get("sac_checkpoints", {})
        missing = [
            int(run["model_seed"])
            for run in config["policy_runs"]
            if str(int(run["model_seed"])) not in checkpoints
        ]
        if missing:
            raise ValueError(f"Missing frozen SAC checkpoints for model seeds: {missing}")
    return config


def prepare_run_dir(config: dict[str, Any], run_id: str | None, output_dir: Path | None = None) -> Path:
    root = resolve_from_root(config["output_root"])
    runs = config["policy_runs"]
    default_run = f"seed_{runs[0]['model_seed']}" if len(runs) == 1 else f"{config['cohort']}__{config['profile']}"
    path = output_dir.resolve() if output_dir else root / (run_id or default_run)
    path.mkdir(parents=True, exist_ok=True)
    return path


def fixed_and_amortized_policies(context, config: dict[str, Any], model_seed: int):
    policies = {}
    if "fixed_controls" in config["methods"]:
        policies.update(fixed_regimen_policies(
            context.rollout_config.decision_days(),
            {
                "anti_pd1": "Anti-PD1", "anti_b41bb": "Anti-4-1BB",
                "combine": "Combine", "no_treatment": "No treatment",
            },
            context.device,
        ))
    if "validation_derived_openloop" in config["methods"]:
        schedule_table = pd.read_csv(resolve_from_root(config["validation_derived_schedule_csv"]))
        selected = schedule_table[schedule_table["model_seed"].astype(int) == int(model_seed)].sort_values("day")
        if selected.empty:
            raise ValueError(f"No validation-derived open-loop schedule for seed {model_seed}")
        schedule = selected[["action_PD1", "action_B41BB"]].to_numpy(np.float32)
        policies["Validation-derived open loop"] = ArraySchedulePolicy(schedule).to(context.device)
    if "amortized_cnn" in config["methods"]:
        policies["Amortized 1D-CNN"] = context.policy
    return policies


def evaluate_policy_set(context, specs, policies, method_group: str):
    summary_rows, actions, compute = [], [], []
    for name, policy in policies.items():
        started = time.perf_counter()
        for spec in specs:
            summary, trajectory = evaluate_patient(
                context.reaction_model, policy, spec, context.rollout_config,
                context.loss_config, name,
            )
            action = action_frame(spec, name, trajectory)
            summary["method_group"] = method_group
            action["method_group"] = method_group
            summary_rows.append(summary)
            actions.append(action)
        compute.append({
            "strategy": name,
            "method_group": method_group,
            "patient_count": len(specs),
            "wall_seconds": float(time.perf_counter() - started),
            "forward_evaluations": len(specs),
            "gradient_evaluations": 0,
            "marginal_seconds_per_patient": float(time.perf_counter() - started) / max(len(specs), 1),
        })
    return summary_rows, actions, compute


def run_patient_optimizer(method: str, context, specs, settings: dict[str, Any], seed: int):
    label = {
        "gradient_open_loop": "Per-patient gradient open loop",
        "rolling_horizon_mpc": "Rolling-horizon MPC",
        "diagonal_cmaes": "Diagonal CMA-ES",
    }[method]
    summary_rows, actions, accounting = [], [], []
    for patient_index, spec in enumerate(specs):
        result = OPTIMIZERS[method](
            context.reaction_model, spec, context.rollout_config, context.loss_config,
            settings, seed=int(seed) + patient_index,
        )
        summary, trajectory = evaluate_patient(
            context.reaction_model, result.policy, spec, context.rollout_config,
            context.loss_config, label,
        )
        action = action_frame(spec, label, trajectory)
        summary["method_group"] = method
        action["method_group"] = method
        summary_rows.append(summary)
        actions.append(action)
        accounting.append({
            "strategy": label,
            "method_group": method,
            "patient_id": spec["patient_id"],
            **result.accounting,
        })
    return summary_rows, actions, accounting


def run_sac_method(context, train_specs, eval_specs, settings: dict[str, Any], seed: int, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    result = train_sac(
        context.reaction_model, train_specs, context.rollout_config, context.loss_config,
        settings, seed,
    )
    result.history.to_csv(output_dir / "sac_training_history.csv", index=False)
    torch.save({
        "actor_state_dict": result.agent.actor.state_dict(),
        "settings": settings,
        "observation_dim": result.agent.actor.trunk[0].in_features,
    }, output_dir / "sac_actor.pt")
    rows, actions, compute = evaluate_policy_set(
        context, eval_specs, {"SAC": result.policy}, "sac"
    )
    compute[0].update(result.accounting)
    return rows, actions, compute


def run_frozen_sac_method(
    context,
    eval_specs,
    checkpoint_path: Path,
):
    expected_observation_dim = (
        int(context.rollout_config.history_length)
        * len(context.rollout_config.feature_names)
    )
    started = time.perf_counter()
    policy, payload = load_sac_policy(
        checkpoint_path,
        expected_observation_dim=expected_observation_dim,
        device=context.device,
    )
    load_seconds = float(time.perf_counter() - started)
    rows, actions, compute = evaluate_policy_set(
        context, eval_specs, {"SAC": policy}, "sac"
    )
    compute[0].update({
        "checkpoint_load_seconds": load_seconds,
        "offline_training_in_current_run": False,
        "environment_steps": 0,
        "sac_training_episodes": int(payload["settings"].get("episodes", 0)),
    })
    return rows, actions, compute


def run_one_policy_seed(
    run_spec: dict[str, Any], config: dict[str, Any], seed_dir: Path,
) -> list[pd.DataFrame]:
    model_seed = int(run_spec["model_seed"])
    context = load_saved_policy_context(run_spec["run_dir"], device=config.get("device", "auto"))
    cohort_df, _ = load_frozen_cohort(config["cohort_dir"], config["cohort"])
    cohort_df = limit_patients(cohort_df, config.get("max_patients"))
    specs = make_specs(
        context, cohort_df, int(config["n_particles"]),
        int(run_spec.get("rollout_seed", 100000 + model_seed)),
    )
    if "sac" in config["methods"]:
        sac_settings = config["method_settings"]["sac"]
        sac_mode = str(sac_settings.get("mode", "train")).lower()
        if sac_mode == "train":
            train_df, _ = load_frozen_cohort(config["cohort_dir"], "train")
            train_df = limit_patients(train_df, config.get("sac_train_patients"))
            train_specs = make_specs(
                context, train_df, int(config["n_particles"]),
                int(run_spec.get("train_rollout_seed", 12000 + model_seed)),
            )
    summary_rows, actions, accounting = [], [], []
    static = fixed_and_amortized_policies(context, config, model_seed)
    if static:
        values = evaluate_policy_set(context, specs, static, "existing_controls")
        summary_rows.extend(values[0]); actions.extend(values[1]); accounting.extend(values[2])
    for method in ("gradient_open_loop", "rolling_horizon_mpc", "diagonal_cmaes"):
        if method not in config["methods"]:
            continue
        values = run_patient_optimizer(method, context, specs, config["method_settings"][method], model_seed)
        summary_rows.extend(values[0]); actions.extend(values[1]); accounting.extend(values[2])
    if "sac" in config["methods"]:
        if sac_mode == "load":
            checkpoint_path = resolve_from_root(config["sac_checkpoints"][str(model_seed)])
            values = run_frozen_sac_method(context, specs, checkpoint_path)
        elif sac_mode == "train":
            values = run_sac_method(
                context, train_specs, specs, sac_settings, model_seed, seed_dir,
            )
        else:
            raise ValueError(f"Unknown SAC mode: {sac_mode!r}")
        summary_rows.extend(values[0]); actions.extend(values[1]); accounting.extend(values[2])
    summary = pd.DataFrame(summary_rows)
    action = pd.concat(actions, ignore_index=True) if actions else pd.DataFrame()
    compute = pd.DataFrame(accounting)
    for frame in (summary, action, compute):
        frame["model_seed"] = model_seed
    return [summary, action, compute]


def main() -> int:
    args = parse_args()
    config = resolve_config(args)
    run_dir = prepare_run_dir(config, args.run_id, args.output_dir)
    config_path = run_dir / "resolved_config.json"
    write_json(config_path, config)
    all_frames = [[], [], []]
    for run_spec in config["policy_runs"]:
        model_seed = int(run_spec["model_seed"])
        seed_dir = run_dir / f"model_seed_{model_seed}"
        frames = run_one_policy_seed(run_spec, config, seed_dir)
        for target, frame in zip(all_frames, frames):
            target.append(frame)
    output_names = ("all_patient_metrics.csv", "all_actions.csv", "all_compute_accounting.csv")
    for parts, name in zip(all_frames, output_names):
        pd.concat(parts, ignore_index=True).to_csv(run_dir / name, index=False)
    write_json(run_dir / "run_status.json", {
        "status": "completed",
        "cohort": config["cohort"],
        "completed_model_seeds": len(config["policy_runs"]),
    })
    print(f"[policy-benchmark] output={run_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
