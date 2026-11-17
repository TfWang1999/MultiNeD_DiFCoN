from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


PRIMARY_METRICS = ("loss", "final_C_rel", "tumor_auc_rel", "total_dose")


def holm_adjust(pvalues: list[float]) -> list[float]:
    values = np.asarray(pvalues, dtype=float)
    adjusted = np.full_like(values, np.nan)
    finite = np.flatnonzero(np.isfinite(values))
    order = finite[np.argsort(values[finite])]
    running = 0.0
    n = len(finite)
    for rank, index in enumerate(order):
        running = max(running, (n - rank) * values[index])
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def aggregate_many(run_dirs: list[Path], output_dir: Path, reference: str) -> dict:
    metrics = pd.concat([pd.read_csv(path / "all_patient_metrics.csv") for path in run_dirs], ignore_index=True)
    if metrics.empty:
        raise ValueError("Policy benchmark has no patient metrics")
    output_dir.mkdir(parents=True, exist_ok=True)
    patient_seed_mean = metrics.groupby(
        ["patient_id", "strategy"], as_index=False
    )[list(PRIMARY_METRICS)].mean()
    patient_seed_mean.to_csv(output_dir / "patient_seed_mean_metrics.csv", index=False)
    summary = patient_seed_mean.groupby("strategy", as_index=False).agg(
        **{
            f"mean_{metric}": (metric, "mean") for metric in PRIMARY_METRICS
        },
        patient_count=("patient_id", "nunique"),
    )
    for metric in PRIMARY_METRICS:
        sd = patient_seed_mean.groupby("strategy")[metric].std().rename(f"sd_{metric}")
        summary = summary.merge(sd, on="strategy", how="left")
    summary.to_csv(output_dir / "strategy_summary.csv", index=False)
    comparisons = []
    for metric in PRIMARY_METRICS:
        pivot = patient_seed_mean.pivot(index="patient_id", columns="strategy", values=metric)
        if reference not in pivot.columns:
            raise ValueError(f"Reference strategy {reference!r} is absent")
        for comparator in pivot.columns:
            if comparator == reference:
                continue
            paired = pivot[[reference, comparator]].dropna()
            difference = paired[comparator] - paired[reference]
            if paired.shape[0] >= 2:
                statistic, pvalue = stats.ttest_rel(paired[comparator], paired[reference])
            else:
                statistic, pvalue = np.nan, np.nan
            comparisons.append({
                "metric": metric,
                "reference": reference,
                "comparator": comparator,
                "n_patients": int(paired.shape[0]),
                "mean_paired_difference": float(difference.mean()),
                "sd_paired_difference": float(difference.std(ddof=1)),
                "paired_t": float(statistic),
                "pvalue": float(pvalue),
            })
    comparison_df = pd.DataFrame(comparisons)
    comparison_df["pvalue_holm"] = holm_adjust(comparison_df["pvalue"].tolist())
    comparison_df.to_csv(output_dir / "paired_primary_comparisons_holm.csv", index=False)
    report = {
        "patient_is_primary_unit": True,
        "seed_patient_rows_treated_as_independent": False,
        "patient_count": int(patient_seed_mean["patient_id"].nunique()),
        "strategy_count": int(patient_seed_mean["strategy"].nunique()),
        "reference": reference,
        "holm_family_size": int(comparison_df.shape[0]),
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate B2 policy benchmark with patient-level pairing.")
    parser.add_argument("--run-dir", type=Path, required=True, nargs="+")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--reference", default="Amortized 1D-CNN")
    args = parser.parse_args()
    run_dirs = [path.resolve() for path in args.run_dir]
    output_dir = args.output_dir.resolve() if args.output_dir else run_dirs[0]
    print(aggregate_many(run_dirs, output_dir, args.reference))


if __name__ == "__main__":
    main()
