"""Resolve explicitly selected inputs without modifying an upstream run."""
from pathlib import Path


DRIFT_NAME = "drift_model.pt"
REACTION_NAME = "reaction_coupled_model.pt"
PARAMETER_NAME = "reaction_parameter_settings.json"
QIN_NAME = "q_in_precursor_empirical.npy"
DEFAULT_DRIFT_CHECKPOINT = Path("MultiNeD/outputs/notebook_drift_run") / DRIFT_NAME
DEFAULT_REACTION_CHECKPOINT = Path("MultiNeD/outputs/reaction_coupled_run") / REACTION_NAME
VOLUME_TABLE = Path("volume_data/paper_tumor_volume_long.csv")
OBSERVED_VOLUME = Path("data/mouse/source_metadata") / VOLUME_TABLE.name


def absolute_path(value, repo):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else repo / path).resolve()


def require_file(path):
    if not path.is_file():
        raise ValueError(f"Required upstream file does not exist: {path}")
    return str(path)


def checkpoint_path(value, filename, repo):
    path = absolute_path(value, repo)
    if path.is_file():
        return path
    candidates = [p for p in (path / filename, path / "result" / filename) if p.is_file()]
    if len(candidates) != 1:
        raise ValueError(f"Select one checkpoint file or its exact run/result directory: {path}; found {len(candidates)} matching files")
    return candidates[0]
