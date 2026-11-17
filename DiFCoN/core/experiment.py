from __future__ import annotations

from pathlib import Path

import pandas as pd


def cohort_csv_path(directory: str | Path, split: str) -> Path:
    """Resolve a prepared LHS cohort or its copy in a saved run."""
    directory = Path(directory)
    table_dir = directory / "tables"
    return (table_dir if table_dir.is_dir() else directory) / f"{split}_lhs_cohort.csv"


def load_reference_cohorts(
    source_run: str | Path,
    n_train: int,
    n_validation: int,
    n_test: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the same training, validation and test patients from the fixed cohort or a prior run."""

    requested = {
        "train": int(n_train),
        "validation": int(n_validation),
        "test": int(n_test),
    }
    cohorts = {}
    for cohort, count in requested.items():
        frame = pd.read_csv(cohort_csv_path(source_run, cohort))
        if len(frame) < count:
            raise ValueError(f"{cohort} cohort only contains {len(frame)} rows, requested {count}.")
        frame = frame.iloc[:count].copy().reset_index(drop=True)
        if frame["patient_id"].duplicated().any():
            raise ValueError(f"{cohort} cohort contains duplicate patient_id values.")
        cohorts[cohort] = frame
    return cohorts["train"], cohorts["validation"], cohorts["test"]






__all__ = [
    "cohort_csv_path",
    "load_reference_cohorts",
]
