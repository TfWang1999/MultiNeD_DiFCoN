"""Read the prepared figure tables, preserving their recorded column types."""

import json
from pathlib import Path

import pandas as pd

from ._paths import SOURCE_DATA_DIR


TABLE_FILES = {
    "baseline": "baseline_strategy_statistics.csv",
    "dose_weight": "dose_weight_sensitivity.xlsx",
    "objective": "objective_ablation.xlsx",
    "observation": "observation_ablation.xlsx",
    "openloop": "openloop_comparison.xlsx",
    "locked": "locked_benchmark_analysis.xlsx",
    "attribution": "policy_input_attribution.xlsx",
}


def read_source_table(group: str, sheet: str = "paired_tests", *, source_dir: Path | None = None) -> pd.DataFrame:
    """Read one named table; all preparation is done by prepare_source_tables.py."""
    folder = SOURCE_DATA_DIR if source_dir is None else Path(source_dir)
    path = folder / TABLE_FILES[group]
    manifest = json.loads((folder / "source_tables_manifest.json").read_text(encoding="utf-8"))
    spec = manifest["tables"][group][sheet]
    if group == "baseline":
        frame = pd.read_csv(path, dtype=spec["dtypes"])
    else:
        frame = pd.read_excel(path, sheet_name=sheet, dtype=spec["dtypes"], engine="openpyxl")
    return frame
