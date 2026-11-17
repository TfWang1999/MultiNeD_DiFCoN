from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


CODE_DIR = Path(__file__).resolve().parents[2]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from experiments.benchmarks.dynamics_baselines import build_method
from experiments.benchmarks.dynamics_full_data_splits import (
    FULL_DATA_ROLLOUT_SCOPE,
    build_distribution_full_data_joint_split,
    distribution_full_data_joint_summary,
    full_data_rollout_targets,
)

from experiments.benchmarks.dynamics_shared_conditional_loto_splits import (
    build_shared_conditional_loto_splits,
    shared_conditional_loto_summary,
)
from experiments.benchmarks.dynamics_benchmark_common import (
    PROJECT_ROOT,
    environment_record,
    load_base_validation_config,
    load_config,
    prepare_run_directory,
    sample_group,
    set_seed,
    write_json,
)
from training.coupled_dynamics import build_z_transform_params, distribution_metrics
from experiments.volume_loto.model_runner import load_inputs
from experiments.volume_loto.splits import standardize_holdout_split


DEFAULT_CONFIG = CODE_DIR / "configs" / "benchmarking" / "cell_loto_benchmark.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run matched unpaired-snapshot dynamics benchmarks.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--run-id")
    target.add_argument("--output-dir", type=Path, help="Directory for this benchmark run")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--methods", default=None, help="Comma-separated configured method IDs.")
    parser.add_argument("--keep-other-methods", action="store_true",
                        help="Merge a method subset into the saved single-seed cell-LOTO table")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    if args.keep_other_methods and (not args.methods or args.seed is None or args.output_dir is None):
        parser.error("--keep-other-methods requires --methods, --seed and --output-dir")
    return args


def resolve_config(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    config = load_config(args.config)
    base = load_base_validation_config(config["base_validation_config"])
    if args.device is not None:
        config.setdefault("evaluation", {})["device"] = str(args.device)
        base["drift"]["device"] = str(args.device)
        base["reaction"]["device"] = str(args.device)
        config.setdefault("method_settings", {}).setdefault(
            "project_balanced_sinkhorn_drift", {}
        )["device"] = str(args.device)
        for method_id in ("mioflow", "tigon", "official_ot_cfm"):
            config.setdefault("method_settings", {}).setdefault(method_id, {})["device"] = str(args.device)
    configured = list(config["methods"])
    if args.methods:
        requested = [value.strip() for value in args.methods.split(",") if value.strip()]
        unknown = sorted(set(requested) - set(configured))
        if unknown:
            raise ValueError(f"Requested methods are not configured: {unknown}")
        config["methods"] = requested
    if args.seed is not None:
        config["seeds"] = [int(args.seed)]
    config["base_validation_resolved"] = base
    return config, base


def build_full_data_joint_tasks(split, n_particles: int, seed: int):
    """Score all later observations from one Control day 1 sample."""
    groups = split.train_group_data
    source_group_id = "day1__Control"
    source_day = float(groups[source_group_id]["day"])
    rng = np.random.default_rng(int(seed))
    source_z = sample_group(groups, source_group_id, n_particles, int(rng.integers(0, 2**31 - 1)))
    tasks = []
    for target_group_id in full_data_rollout_targets(split, source_group_id):
        target = groups[target_group_id]
        target_z = sample_group(groups, target_group_id, n_particles, int(rng.integers(0, 2**31 - 1)))
        tasks.append({
            "transition_id": f"{source_group_id}__to__{target_group_id}",
            "source_group": source_group_id,
            "target_group": target_group_id,
            "source_day": source_day,
            "target_day": float(target["day"]),
            "target_treatment": str(target["treatment_clean"]),
            "P": float(target["P"]),
            "B": float(target["B"]),
            "source_z": source_z,
            "target_z": target_z,
            "prediction_scope": FULL_DATA_ROLLOUT_SCOPE,
            "global_joint_fit": True,
            "training_task_manifest_separated": True,
        })
    return tasks


def build_shared_conditional_loto_tasks(
    split,
    full_group_data: dict,
    n_particles: int,
    seed: int,
    *,
    source_group: str,
):
    """Build one held-out prediction task from the fixed Control anchor."""

    if len(split.heldout_groups) != 1:
        raise ValueError("shared_conditional_loto requires exactly one held-out group per fold")
    source_group_id = str(source_group)
    target_group_id = str(split.heldout_groups[0])
    if source_group_id not in split.train_group_data:
        raise ValueError(f"Source anchor {source_group_id!r} is not available in this fold")
    source = full_group_data[source_group_id]
    target = full_group_data[target_group_id]
    rng = np.random.default_rng(int(seed))
    source_z = sample_group(
        split.train_group_data,
        source_group_id,
        n_particles,
        int(rng.integers(0, 2**31 - 1)),
    )
    target_z = sample_group(
        full_group_data,
        target_group_id,
        n_particles,
        int(rng.integers(0, 2**31 - 1)),
    )
    return [
        {
            "transition_id": split.split_id,
            "source_group": source_group_id,
            "target_group": target_group_id,
            "source_day": float(source["day"]),
            "target_day": float(target["day"]),
            "target_treatment": str(target["treatment_clean"]),
            "P": float(target["P"]),
            "B": float(target["B"]),
            "source_z": source_z,
            "target_z": target_z,
            "prediction_scope": "heldout_treatment_day_from_control_anchor",
            "global_joint_fit": True,
            "shared_conditional_loto": True,
            "training_task_manifest_separated": True,
        }
    ]


def run_one_method(
    method_id: str,
    method_settings: dict[str, Any],
    base: dict[str, Any],
    split,
    tasks: list[dict[str, Any]],
    seed: int,
    method_dir: Path,
    evaluation: dict[str, Any],
    transform: dict[str, Any],
) -> pd.DataFrame:
    set_seed(seed)
    method = build_method(method_id, method_settings, base)
    fit_kwargs = {"tasks": tasks} if bool(getattr(method, "requires_tasks", False)) else {}
    method.fit(
        split.train_group_data, split.train_transitions, seed, method_dir, **fit_kwargs
    )
    z_cols = list(next(iter(split.train_group_data.values()))["z_cols"])
    rows = []
    for index, task in enumerate(tasks):
        predict_kwargs = {
            "day0": task["source_day"],
            "day1": task["target_day"],
            "P": task["P"],
            "B": task["B"],
        }
        if bool(getattr(method, "requires_tasks", False)):
            predict_kwargs["transition_id"] = task.get("transition_id")
        predicted = method.predict(task["source_z"], **predict_kwargs)
        row = {
            "seed": int(seed),
            "split_id": split.split_id,
            "holdout_kind": split.kind,
            "method": method_id,
            "source_group": task["source_group"],
            "target_group": task["target_group"],
            "source_day": task["source_day"],
            "target_day": task["target_day"],
            "target_treatment": task["target_treatment"],
            "prediction_scope": task["prediction_scope"],
            "P": task["P"],
            "B": task["B"],
            "n_source": int(task["source_z"].shape[0]),
            "n_target": int(task["target_z"].shape[0]),
            "n_predicted": int(predicted.shape[0]),
            "z_scale": evaluation["z_scale"],
        }
        row.update(distribution_metrics(
            predicted, task["target_z"], z_cols, transform, evaluation,
            seed=int(seed) + index,
        ))
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    config, base = resolve_config(args)
    split_config = config["splits"]
    split_mode = str(split_config["mode"])
    if args.keep_other_methods and split_mode != "shared_conditional_loto":
        raise ValueError("--keep-other-methods is for cell-LOTO. Full-data rollout needs new scores for all methods.")
    bundle = load_inputs(base, PROJECT_ROOT, raw_scores=split_mode == "shared_conditional_loto")
    exclusions = []
    if split_mode == "distribution_full_data_joint":
        split = build_distribution_full_data_joint_split(
            bundle["group_data"], bundle["transitions"], bundle["volume"]
        )
        splits = [split]
        split_summaries = [distribution_full_data_joint_summary(split)]
    elif split_mode == "shared_conditional_loto":
        reaction_loto = base["experiments"]["m1"]["volume_timepoint_cv"]
        source_group = str(split_config.get("source_group", "day1__Control"))
        splits, exclusions = build_shared_conditional_loto_splits(
            bundle["group_data"],
            bundle["transitions"],
            bundle["volume"],
            treatments=split_config.get("treatments", reaction_loto["treatments"]),
            days=split_config.get("days", reaction_loto["days"]),
            source_group=source_group,
        )
        split_summaries = [
            shared_conditional_loto_summary(
                split, bundle["group_data"], source_group=source_group
            )
            for split in splits
        ]
        expected_fold_count = split_config.get("expected_fold_count")
        if expected_fold_count is not None and len(splits) != int(expected_fold_count):
            raise ValueError(
                "Shared-conditional LOTO fold count mismatch: "
                f"expected {expected_fold_count}, observed {len(splits)}"
            )
        expected_targets = split_config.get("expected_target_groups")
        observed_targets = [split.heldout_groups[0] for split in splits]
        if expected_targets is not None and set(map(str, expected_targets)) != set(observed_targets):
            raise ValueError(
                "Shared-conditional LOTO target mismatch: "
                f"expected {sorted(map(str, expected_targets))}, "
                f"observed {sorted(observed_targets)}"
            )
    else:
        raise ValueError(f"Unknown B1 split mode: {split_mode!r}")
    if split_mode == "distribution_full_data_joint":
        training_group_count = len(splits[0].train_group_data)
        training_transition_count = int(splits[0].train_transitions.shape[0])
        evaluation_target_count = split_summaries[0]["evaluation_target_count"]
        expected = split_config.get("expected_counts", {})
        observed = {
            "training_group_count": training_group_count,
            "training_transition_count": training_transition_count,
            "evaluation_target_count": evaluation_target_count,
        }
        for key, value in expected.items():
            if key in observed and int(value) != observed[key]:
                raise ValueError(
                    f"Full-data count mismatch for {key}: expected {value}, observed {observed[key]}"
                )

    if split_mode == "shared_conditional_loto":
        folds = [standardize_holdout_split(split, bundle["group_data"]) for split in splits]
        config["z_standardization_by_split"] = {
            split.split_id: params for split, _, params in folds
        }
    else:
        folds = [(split, bundle["group_data"], None) for split in splits]

    prefix = "b1_full_data" if split_mode == "distribution_full_data_joint" else "b1_shared_loto_all7"
    if args.output_dir:
        run_dir = args.output_dir.resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        run_name = f"{prefix}_seed{config['seeds'][0]}" if len(config["seeds"]) == 1 else prefix
        run_dir = prepare_run_directory(config["output_root"], args.run_id or run_name)
    metrics_path = run_dir / "all_distribution_metrics.csv"
    previous_metrics = None
    config_path = run_dir / "resolved_config.json"
    if args.keep_other_methods:
        previous_config = json.loads(config_path.read_text(encoding="utf-8"))
        previous_metrics = pd.read_csv(metrics_path, float_precision="round_trip")
        saved_scalers = previous_config["z_standardization_by_split"]
        current_scalers = config["z_standardization_by_split"]
        if (previous_config["seeds"] != config["seeds"]
                or previous_config["splits"] != split_config
                or saved_scalers.keys() != current_scalers.keys()):
            raise ValueError("Saved cell-LOTO seed or splits differ from the requested run.")
        for split_id, current in current_scalers.items():
            saved = saved_scalers[split_id]
            if (saved["z_cols"] != current["z_cols"]
                    or saved["n_training_cells"] != current["n_training_cells"]
                    or not np.allclose([saved["mean"], saved["scale"]],
                                       [current["mean"], current["scale"]], rtol=1e-12, atol=1e-12)):
                raise ValueError(f"Saved cell-LOTO scaling differs for {split_id}.")
        evaluation_keys = ("n_particles", "sinkhorn_epsilon", "sinkhorn_iters", "energy_max_points")
        if any(previous_config["evaluation"][key] != config["evaluation"][key] for key in evaluation_keys):
            raise ValueError("Saved cell-LOTO metric settings differ from the requested run.")
        keep = previous_metrics.loc[~previous_metrics["method"].isin(config["methods"])].copy()
        expected_splits = {split.split_id for split, _, _ in folds}
        for method in set(previous_config["methods"]) - set(config["methods"]):
            rows = keep.loc[keep["method"].eq(method)]
            if len(rows) != len(expected_splits) or set(rows["split_id"]) != expected_splits:
                raise ValueError(f"Saved cell-LOTO rows are incomplete for {method}.")
        if not keep["seed"].eq(args.seed).all() or not keep["holdout_kind"].eq(split_mode).all():
            raise ValueError("Saved rows must come from the same seed and cell-LOTO protocol.")
        previous_metrics = keep
    else:
        write_json(config_path, config)
        write_json(run_dir / "environment.json", environment_record(config["evaluation"].get("device", "auto")))
    write_json(run_dir / "run_status.json", {"status": "running"})
    pd.DataFrame(split_summaries).to_csv(run_dir / "split_registry.csv", index=False)
    if exclusions:
        pd.DataFrame(exclusions).to_csv(run_dir / "non_estimable_splits.csv", index=False)
    metric_parts = []
    completed_fits = 0
    evaluation = {**config["evaluation"], "z_scale": "robust01"}
    n_particles = int(evaluation["n_particles"])
    transforms = [build_z_transform_params(split.train_group_data) for split, _, _ in folds]
    for seed in map(int, config["seeds"]):
        for (split, fold_groups, _), transform in zip(folds, transforms):
            if split_mode == "distribution_full_data_joint":
                tasks = build_full_data_joint_tasks(split, n_particles, seed)
            else:
                tasks = build_shared_conditional_loto_tasks(
                    split,
                    fold_groups,
                    n_particles,
                    seed,
                    source_group=str(split_config.get("source_group", "day1__Control")),
                )
            for method_id in config["methods"]:
                method_dir = run_dir / f"seed_{seed}" / split.split_id / method_id
                method_dir.mkdir(parents=True, exist_ok=True)
                metrics = run_one_method(
                    method_id,
                    config.get("method_settings", {}).get(method_id, {}),
                    base, split, tasks, seed, method_dir, evaluation, transform,
                )
                metric_parts.append(metrics)
                completed_fits += 1
    metrics = pd.concat(metric_parts, ignore_index=True)
    if previous_metrics is not None:
        metrics = pd.concat([previous_metrics, metrics], ignore_index=True)
        config["method_settings"] = {
            **previous_config["method_settings"],
            **{method: config["method_settings"].get(method, {}) for method in config["methods"]},
        }
        config["methods"] = previous_config["methods"]
        write_json(config_path, config)
    pending_metrics = metrics_path.with_suffix(".csv.tmp")
    metrics.to_csv(pending_metrics, index=False)
    pending_metrics.replace(metrics_path)
    write_json(run_dir / "run_status.json", {
        "status": "completed",
        "completed_method_fits": (
            metrics[["seed", "split_id", "method"]].drop_duplicates().shape[0]
            if previous_metrics is not None else completed_fits
        ),
        "metric_rows": int(metrics.shape[0]),
    })
    print(f"[dynamics-benchmark] output={run_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
