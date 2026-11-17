"""Repository-relative paths shared by human validation."""
from pathlib import Path
from dataclasses import dataclass
HUMAN_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = HUMAN_ROOT.parent
HCOP_PATH = HUMAN_ROOT / "resources/human_mouse_hcop_fifteen_column.txt.gz"


@dataclass(frozen=True)
class HumanInputs:
    results: Path = HUMAN_ROOT
    dynamics: Path = REPO_ROOT / "dynamics_analysis/outputs"
    mouse_trajectories: Path = REPO_ROOT / "dynamics_analysis/mechanism_aligned_dynamics/inputs/paired_timeseries.csv"
