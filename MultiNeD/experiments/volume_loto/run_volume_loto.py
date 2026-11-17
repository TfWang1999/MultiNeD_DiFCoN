from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd


CODE_DIR = Path(__file__).resolve().parents[2]
PROJECT_ROOT = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from experiments.volume_loto.config import load_validation_config, project_path
from experiments.volume_loto.model_runner import (
    load_inputs,
    package_versions,
    train_fresh_split_model,
    evaluate_heldout_split,
    write_json,
)
from experiments.volume_loto.splits import (
    build_volume_timepoint_holdout_split,
    holdout_summary,
    standardize_holdout_split,
)


DEFAULT_CONFIG = CODE_DIR / "configs" / "validation" / "volume_loto_common_days.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce held-out tumour-volume predictions.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--run-id", default=None, help="Run subdirectory under output.root")
    target.add_argument("--output-dir", type=Path, help="Directory for this CV run")
    return parser.parse_args()


def _m1_splits(config: dict[str, Any], bundle: dict[str, Any]):
    m1 = config["experiments"]["m1"]
    volume_cv = m1.get("volume_timepoint_cv", {})
    for treatment in volume_cv.get("treatments", []):
        for day in volume_cv.get("days", []):
            yield build_volume_timepoint_holdout_split(
                bundle["group_data"],
                bundle["transitions"],
                bundle["volume"],
                treatment=treatment,
                day=float(day),
            )


def run_m1(
    config: dict[str, Any],
    bundle: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    volume_parts = []
    split_rows = []
    m1 = config["experiments"]["m1"]
    volume_cv = m1.get("volume_timepoint_cv", {})
    cv_seeds = [int(value) for value in volume_cv.get("seeds", [config["seed"]])]
    split_templates = [
        standardize_holdout_split(split, bundle["group_data"])
        for split in _m1_splits(config, bundle)
    ]

    for cv_seed in cv_seeds:
        for split_index, (split, fold_groups, standardization) in enumerate(split_templates):
            split_dir = output_dir / f"seed_{cv_seed}" / split.split_id
            split_dir.mkdir(parents=True, exist_ok=True)
            summary = {
                **holdout_summary(split),
                "cv_seed": cv_seed,
                "fold_index": split_index,
                "z_standardization": standardization,
            }
            volume_path = split_dir / "heldout_volume_metrics.csv"

            write_json(split_dir / "split_summary.json", summary)
            split.train_transitions.to_csv(split_dir / "training_transitions.csv", index=False)
            trained = train_fresh_split_model(
                config,
                split,
                split_dir,
                seed=cv_seed,
                project_root=PROJECT_ROOT,
            )
            volume = evaluate_heldout_split(
                config,
                split,
                fold_groups,
                bundle["volume"],
                trained,
                seed=cv_seed,
            )
            volume["cv_seed"] = cv_seed
            volume["fold_index"] = split_index
            volume.to_csv(volume_path, index=False)
            volume_parts.append(volume)
            split_rows.append(summary)

    nonempty_volumes = [part for part in volume_parts if not part.empty]
    volumes = pd.concat(nonempty_volumes, ignore_index=True) if nonempty_volumes else pd.DataFrame()
    cv_integrity = {}
    if volume_cv:
        expected_folds = (
            len(volume_cv.get("treatments", []))
            * len(volume_cv.get("days", []))
            * len(cv_seeds)
        )
        reaction_timepoints = volumes.loc[
            volumes["model"].eq("reaction") & volumes["holdout_kind"].eq("volume_timepoint")
        ].groupby(["cv_seed", "treatment_clean"])["day"].nunique()
        cv_integrity = {
            "expected_folds": expected_folds,
            "completed_folds": len(split_rows),
            "expected_timepoints_per_treatment_seed": len(volume_cv.get("days", [])),
            "reaction_error_rows": len(reaction_timepoints),
            "all_reaction_rows_have_expected_timepoints": bool(
                not reaction_timepoints.empty
                and reaction_timepoints.eq(len(volume_cv.get("days", []))).all()
            ),
        }
        cv_integrity["passed"] = bool(
            cv_integrity["completed_folds"] == cv_integrity["expected_folds"]
            and cv_integrity["all_reaction_rows_have_expected_timepoints"]
        )
        if not cv_integrity["passed"]:
            raise RuntimeError(f"Incomplete full M1 cross-validation output: {cv_integrity}")
    summary = {
        "fold_count": len(split_rows),
        "folds_per_seed": len(split_templates),
        "cv_seed_count": len(cv_seeds),
        "cv_seeds": cv_seeds,
        "volume_metric_rows": int(volumes.shape[0]),
        "profile": config["profile"],
        "cv_integrity": cv_integrity,
        "interpretation": "cross_validated_reaction_model_evidence",
    }
    return summary


def main() -> int:
    args = parse_args()
    config = load_validation_config(args.config, project_root=PROJECT_ROOT)
    bundle = load_inputs(config, PROJECT_ROOT, raw_scores=True)
    root = project_path(config["output"]["root"], PROJECT_ROOT)
    run_dir = args.output_dir.resolve() if args.output_dir else root / (args.run_id or "common4")
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "resolved_config.json", config)
    write_json(run_dir / "package_versions.json", package_versions())
    summary = run_m1(config, bundle, run_dir / "m1")
    write_json(run_dir / "run_status.json", [{"experiment": "m1", "status": "completed", **summary}])
    print(f"[volume-loto] output={run_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
