"""Aggregate dose-weight experiments across the three training seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import DAILY_TOTAL_DOSE_CAP, TOTAL_BUDGET


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def find_run_dirs(study_dir: Path) -> list[Path]:
    return sorted(path.parent for path in study_dir.glob("*/resolved_config.json"))


def load_run(run_dir: Path) -> dict[str, Any]:
    config = read_json(run_dir / "resolved_config.json")
    loss = config["loss"]

    row: dict[str, Any] = {
        "model_seed": config["seeds"]["model"],
        "total_budget": TOTAL_BUDGET,
        "daily_total_dose_cap": DAILY_TOTAL_DOSE_CAP,
    }
    for name in (
        "final_tumor_weight",
        "tumor_auc_weight",
        "functional_activity_weight",
        "smoothness_weight",
        "total_dose_weight",
        "efficiency_weight",
        "functional_retention",
    ):
        row[name] = loss.get(name)

    for cohort in ("validation", "test"):
        values = read_json(run_dir / "summaries" / f"{cohort}_summary.json")["direct_dose_means"]
        for source, target in {
            "final_C_rel": "final_C_rel",
            "tumor_auc_rel": "tumor_auc_rel",
            "total_dose": "total_dose",
            "loss_late_activity": "late_activity",
            "loss_functional_activity_loss": "functional_activity_loss",
        }.items():
            row[f"{cohort}_{target}"] = values[source]
    return row


def main() -> None:
    args = parse_args()
    study_dir = args.study_dir.resolve()
    rows = [load_run(path) for path in find_run_dirs(study_dir)]
    if not rows:
        raise FileNotFoundError(f"No direct-dose run configs found under {study_dir}")
    all_runs = pd.DataFrame(rows).sort_values(["total_dose_weight", "efficiency_weight", "model_seed"])
    aggregate_dir = study_dir / "aggregate"
    aggregate_dir.mkdir(parents=True, exist_ok=True)

    setting_columns = [
        "final_tumor_weight",
        "tumor_auc_weight",
        "functional_activity_weight",
        "smoothness_weight",
        "total_dose_weight",
        "efficiency_weight",
        "functional_retention",
        "total_budget",
        "daily_total_dose_cap",
    ]
    metric_columns = [
        column
        for column in all_runs.columns
        if column.startswith("validation_") or column.startswith("test_")
    ]
    grouped = all_runs.groupby(setting_columns, dropna=False, observed=True)
    seed_summary = grouped[metric_columns].agg(["count", "mean", "std"]).reset_index()
    seed_summary.columns = [
        "_".join(str(part) for part in column if str(part)) if isinstance(column, tuple) else str(column)
        for column in seed_summary.columns
    ]
    seed_summary.to_csv(aggregate_dir / "seed_summary.csv", index=False)

    print(f"runs={len(all_runs)} settings={len(seed_summary)} aggregate={aggregate_dir}")


if __name__ == "__main__":
    main()
