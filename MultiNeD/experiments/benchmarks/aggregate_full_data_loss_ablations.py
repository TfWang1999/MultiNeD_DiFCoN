from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


TABLE_NAMES = (
    "transition_distribution_metrics",
    "rollout_distribution_metrics",
    "volume_metrics",
)


def _collect_variant_table(run_dirs: list[Path], filename: str) -> pd.DataFrame:
    paths = [path for run_dir in run_dirs for path in run_dir.glob(f"seed_*/*/{filename}.csv")
             if path.parent.name != "uot_local_rollout_pretrained"]
    if not paths:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(path) for path in sorted(paths)], ignore_index=True)


def aggregate_many(run_dirs: list[Path], output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        name: _collect_variant_table(run_dirs, name)
        for name in TABLE_NAMES
    }
    for name, table in tables.items():
        if not table.empty:
            table.to_csv(output_dir / f"all_{name}.csv", index=False)

    transition = tables["transition_distribution_metrics"]
    if not transition.empty:
        transition.groupby(["variant", "model_level"], as_index=False).agg(
            mean_sinkhorn=("sinkhorn_divergence", "mean"),
            sd_sinkhorn=("sinkhorn_divergence", "std"),
            mean_energy=("energy_distance", "mean"),
            sd_energy=("energy_distance", "std"),
            mean_score_mae=("score_mean_mae", "mean"),
            mean_wasserstein_1=("wasserstein_1", "mean"),
            sd_wasserstein_1=("wasserstein_1", "std"),
            transition_count=("transition_id", "count"),
            seed_count=("seed", "nunique"),
        ).to_csv(output_dir / "transition_distribution_summary.csv", index=False)

    rollout = tables["rollout_distribution_metrics"]
    if not rollout.empty:
        rollout.groupby(["variant"], as_index=False).agg(
            mean_sinkhorn=("sinkhorn_divergence", "mean"),
            sd_sinkhorn=("sinkhorn_divergence", "std"),
            mean_energy=("energy_distance", "mean"),
            sd_energy=("energy_distance", "std"),
            mean_score_mae=("score_mean_mae", "mean"),
            mean_wasserstein_1=("wasserstein_1", "mean"),
            sd_wasserstein_1=("wasserstein_1", "std"),
            treatment_count=("treatment_clean", "nunique"),
            timepoint_count=("day", "count"),
            seed_count=("seed", "nunique"),
        ).to_csv(output_dir / "rollout_distribution_summary.csv", index=False)

    volume = tables["volume_metrics"]
    if not volume.empty:
        volume.assign(
            signed_error=volume["predicted_volume"] - volume["observed_mean_volume"]
        ).groupby(["variant"], as_index=False).agg(
            mae=("absolute_error", "mean"),
            rmse=("squared_error", lambda values: float(np.sqrt(np.mean(values)))),
            mean_signed_error=("signed_error", "mean"),
            treatment_count=("treatment_clean", "nunique"),
            timepoint_count=("day", "count"),
            seed_count=("seed", "nunique"),
        ).to_csv(output_dir / "volume_summary.csv", index=False)

    report = {
        "evaluation_scope": "full_data_in_sample",
        "uses_holdouts": False,
        "interpretation_limit": "Component ablation on reused training data; not a generalization estimate.",
        "run_directories": [str(path) for path in run_dirs],
        "completed_fits": len(transition[["variant", "seed"]].drop_duplicates()),
        "variants": sorted(transition["variant"].unique().tolist()),
        "inferential_tests_included": False,
        "program_scores_vs_PCA_included": False,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate independent full-data B16 loss-ablation runs."
    )
    parser.add_argument("--run-dir", type=Path, required=True, nargs="+")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    run_dirs = [path.resolve() for path in args.run_dir]
    missing = [path for path in run_dirs if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Run directories do not exist: {missing}")
    output_dir = args.output_dir.resolve() if args.output_dir else run_dirs[0]
    print(json.dumps(aggregate_many(run_dirs, output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
