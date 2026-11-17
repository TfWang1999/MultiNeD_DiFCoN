from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


METRICS = ("sinkhorn_divergence", "energy_distance", "score_mean_mae", "wasserstein_1")
COMPARISON_METRICS = ("sinkhorn_divergence", "energy_distance", "score_mean_mae")


def holm_adjust(pvalues: list[float]) -> list[float]:
    values = np.asarray(pvalues, dtype=float)
    adjusted = np.full_like(values, np.nan)
    finite = np.flatnonzero(np.isfinite(values))
    order = finite[np.argsort(values[finite])]
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(finite) - rank) * values[index])
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def aggregate_many(
    run_dirs: list[Path],
    output_dir: Path,
    reference: str = "project_balanced_sinkhorn_drift",
) -> dict[str, int]:
    metrics = pd.concat(
        [pd.read_csv(run_dir / "all_distribution_metrics.csv") for run_dir in run_dirs],
        ignore_index=True,
    )
    missing = sorted(set(METRICS) - set(metrics.columns))
    if missing:
        raise ValueError(f"Dynamics metrics are missing required columns: {missing}")
    output_dir.mkdir(parents=True, exist_ok=True)
    keys = ["holdout_kind", "method"]
    summary_aggregations = dict(
        mean_sinkhorn=("sinkhorn_divergence", "mean"),
        sd_sinkhorn=("sinkhorn_divergence", "std"),
        mean_energy=("energy_distance", "mean"),
        sd_energy=("energy_distance", "std"),
        mean_score_mae=("score_mean_mae", "mean"),
        sd_score_mae=("score_mean_mae", "std"),
        target_count=("target_group", "count"),
        seed_count=("seed", "nunique"),
        split_count=("split_id", "nunique"),
        mean_wasserstein_1=("wasserstein_1", "mean"),
        sd_wasserstein_1=("wasserstein_1", "std"),
    )
    summary = metrics.groupby(keys, as_index=False).agg(**summary_aggregations)
    summary.to_csv(output_dir / "dynamics_method_summary.csv", index=False)
    paired = metrics.pivot_table(
        index=["seed", "split_id", "target_group"],
        columns="method",
        values=list(METRICS),
        aggfunc="first",
    )
    paired.columns = [f"{metric}__{method}" for metric, method in paired.columns]
    paired.reset_index().to_csv(output_dir / "paired_target_metrics.csv", index=False)
    fold_mean = metrics.groupby(["split_id", "target_group", "method"], as_index=False)[list(COMPARISON_METRICS)].mean()
    comparisons = []
    for metric in COMPARISON_METRICS:
        wide = fold_mean.pivot(index=["split_id", "target_group"], columns="method", values=metric)
        if reference not in wide:
            raise ValueError(f"Reference method {reference!r} is absent")
        for comparator in sorted(column for column in wide if column != reference):
            values = wide[[reference, comparator]].dropna()
            difference = values[comparator] - values[reference]
            statistic, pvalue = (
                stats.ttest_rel(values[comparator], values[reference])
                if len(values) >= 2 else (np.nan, np.nan)
            )
            comparisons.append({
                "metric": metric,
                "reference": reference,
                "comparator": comparator,
                "n_folds": int(len(values)),
                "mean_comparator_minus_reference": float(difference.mean()),
                "sd_paired_difference": float(difference.std(ddof=1)),
                "paired_t": float(statistic),
                "pvalue": float(pvalue),
            })
    comparison = pd.DataFrame(comparisons)
    comparison["pvalue_holm"] = holm_adjust(comparison["pvalue"].tolist())
    comparison.to_csv(output_dir / "paired_method_comparisons_holm.csv", index=False)
    return {"metric_rows": int(metrics.shape[0]), "summary_rows": int(summary.shape[0])}


def aggregate(run_dir: Path) -> dict[str, int]:
    return aggregate_many([run_dir], run_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate a completed B1 dynamics benchmark run.")
    parser.add_argument("--run-dir", type=Path, required=True, nargs="+")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--reference", default="project_balanced_sinkhorn_drift")
    args = parser.parse_args()
    run_dirs = [path.resolve() for path in args.run_dir]
    output_dir = args.output_dir.resolve() if args.output_dir else run_dirs[0]
    print(aggregate_many(run_dirs, output_dir, args.reference))


if __name__ == "__main__":
    main()
