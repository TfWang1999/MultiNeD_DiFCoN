"""Input workspaces and export locations for the supplementary notebooks."""

import os
from pathlib import Path

from DiFCoN.figure_workbench.plotting._paths import (
    BASELINE_DIR, DEFAULT_BASELINE_DIR, DEFAULT_SOURCE_DATA_DIR, EXPERIMENT_DIR, OPENLOOP_DIR, SOURCE_DATA_DIR,
)
from MultiNeD.figure_work.scripts.figure_data import find_analysis_dir

SUPP_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = SUPP_DIR.parent
MODEL_DIR = find_analysis_dir().resolve()
HUMAN_RESULTS = Path(os.environ.get("HUMAN_VALIDATION_RESULTS", REPO_ROOT / "human_validation")).resolve()


def figure_output_dir(group):
    """Route figures made from selected reproduction inputs away from the paper files."""
    new_inputs = {
        "model": MODEL_DIR != REPO_ROOT / "MultiNeD/figure_work",
        "human_validation": HUMAN_RESULTS != REPO_ROOT / "human_validation",
        "policy": (
            SOURCE_DATA_DIR != DEFAULT_SOURCE_DATA_DIR
            or BASELINE_DIR.resolve() != DEFAULT_BASELINE_DIR
            or OPENLOOP_DIR.resolve() != EXPERIMENT_DIR / "openloop_comparison"
        ),
    }
    root = REPO_ROOT / "reproduction/supplementary" if new_inputs[group] else SUPP_DIR
    return root / "figure" / group
