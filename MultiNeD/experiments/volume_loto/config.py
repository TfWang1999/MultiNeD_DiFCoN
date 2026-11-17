from __future__ import annotations

import json
from pathlib import Path
from typing import Any


CODE_DIR = Path(__file__).resolve().parents[2]
PROJECT_ROOT = CODE_DIR.parent


def project_path(value: str | Path, project_root: str | Path = PROJECT_ROOT) -> Path:
    """Resolve a config path relative to the project root."""
    path = Path(value)
    if not path.is_absolute():
        path = Path(project_root) / path
    return path.resolve()


def validate_config(config: dict[str, Any], project_root: str | Path = PROJECT_ROOT) -> None:
    if int(config.get("schema_version", 0)) != 1:
        raise ValueError("paper validation config schema_version must be 1")

    required_inputs = (
        "group_pkl",
        "group_info_csv",
        "transition_csv",
        "volume_csv",
        "drift_checkpoint",
        "reaction_checkpoint",
        "reaction_parameter_settings",
    )
    inputs = config.get("inputs", {})
    missing = [name for name in required_inputs if name not in inputs]
    if missing:
        raise ValueError(f"Missing input paths: {missing}")
    absent = [str(project_path(inputs[name], project_root)) for name in required_inputs if not project_path(inputs[name], project_root).exists()]
    if absent:
        raise FileNotFoundError("Missing configured inputs:\n" + "\n".join(absent))

    output = config.get("output", {})
    if not output.get("root"):
        raise ValueError("output.root is required")
    for section in ("drift", "reaction", "evaluation"):
        if section not in config:
            raise ValueError(f"Missing config section: {section}")

    m1 = config.get("experiments", {}).get("m1", {})
    volume_cv = m1.get("volume_timepoint_cv")
    if volume_cv:
        treatments = volume_cv.get("treatments", [])
        days = volume_cv.get("days", [])
        seeds = volume_cv.get("seeds", [])
        if not treatments or any(not isinstance(value, str) or not value for value in treatments):
            raise ValueError("M1 volume_timepoint_cv.treatments must contain treatment names")
        if not days or any(float(value) <= 0 for value in days):
            raise ValueError("M1 volume_timepoint_cv.days must contain positive days")
        if not seeds or any(int(value) < 0 for value in seeds):
            raise ValueError("M1 volume_timepoint_cv.seeds must contain non-negative integers")
        if len(set(map(str, treatments))) != len(treatments):
            raise ValueError("M1 volume_timepoint_cv.treatments contains duplicates")
        if len(set(map(float, days))) != len(days):
            raise ValueError("M1 volume_timepoint_cv.days contains duplicates")
        if len(set(map(int, seeds))) != len(seeds):
            raise ValueError("M1 volume_timepoint_cv.seeds contains duplicates")


def load_validation_config(
    path: str | Path,
    project_root: str | Path = PROJECT_ROOT,
) -> dict[str, Any]:
    path = Path(path).resolve()
    config = json.loads(path.read_text(encoding="utf-8"))
    config["config_source"] = str(path)
    validate_config(config, project_root=project_root)
    return config
