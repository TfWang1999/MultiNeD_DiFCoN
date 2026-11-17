"""Shared locations for the dynamics analyses."""
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
REPO = BASE.parent
INPUTS = BASE / "mechanism_aligned_dynamics/inputs"
ROBUSTNESS = BASE / "outputs/robustness"
ALL_PERTURBATION_METRICS = "program_perturbation_metrics_all_deltas.csv"


def new_output(path):
    path = Path(path).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path
