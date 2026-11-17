from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any


SCHEMA_VERSION = 2
DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs/direct_dose/base.json"
DAILY_TOTAL_DOSE_CAP = 1.0
TOTAL_BUDGET = 3.0
BUDGET_ID = f"budget_{TOTAL_BUDGET:g}"


@dataclass(frozen=True)
class LoadedExperimentConfig:
    resolved: dict[str, Any]
    config_path: Path


def _parse_override_value(raw: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def _set_dotted(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    keys = [part for part in dotted_key.split(".") if part]
    if not keys:
        raise ValueError("Override key must not be empty.")
    current = config
    for key in keys[:-1]:
        if key not in current:
            raise KeyError(f"Unknown config key in override: {dotted_key!r}.")
        child = current[key]
        if not isinstance(child, dict):
            raise ValueError(f"Cannot set {dotted_key!r}: {key!r} is not a mapping.")
        current = child
    if keys[-1] not in current:
        raise KeyError(f"Unknown config key in override: {dotted_key!r}.")
    current[keys[-1]] = value


def _resolve_path(experiment_dir: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (experiment_dir / path).resolve()


def _float_tag(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def _automatic_run_name(config: dict[str, Any]) -> str:
    loss = config["loss"]
    seed = int(config["seeds"]["model"])
    return (
        f"wd{_float_tag(loss['total_dose_weight'])}_"
        f"we{_float_tag(loss['efficiency_weight'])}_"
        f"wf{_float_tag(loss['functional_activity_weight'])}_seed{seed}"
    )


def _sanitize_name(value: str, field: str) -> str:
    name = str(value).strip()
    if not name or name in {".", ".."} or not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
        raise ValueError(f"{field} must contain only letters, digits, '_', '-' or '.': {value!r}")
    return name


def _validate(config: dict[str, Any]) -> None:
    if int(config.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}.")
    for section, names in {
        "cohort": ("n_train", "n_validation", "n_test"),
        "training": ("n_epochs", "batch_size", "eval_every", "eval_patients"),
    }.items():
        for name in names:
            if int(config[section][name]) < 1:
                raise ValueError(f"{section}.{name} must be >= 1.")
    for name, value in config["loss"].items():
        if name.endswith("_weight") and float(value) < 0:
            raise ValueError(f"loss.{name} must be >= 0.")
    if int(config["training"]["eval_patients"]) > int(config["cohort"]["n_validation"]):
        raise ValueError("training.eval_patients cannot exceed cohort.n_validation.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train one configured direct-dose 1D-CNN experiment."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Experiment JSON; defaults to configs/direct_dose/base.json.")
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--run-name", help="Run name under the selected study.")
    target.add_argument("--output-dir", type=Path, help="Write this run directly in the selected directory")
    parser.add_argument("--study-name", help="Study directory under output.root.")
    parser.add_argument("--output-root", type=Path, help="Override output.root.")
    parser.add_argument("--device", help="For example cuda:0, cuda:1 or cpu.")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    return parser


def load_experiment_config(
    experiment_dir: str | Path,
    argv: list[str] | None = None,
) -> LoadedExperimentConfig:
    experiment_dir = Path(experiment_dir).resolve()
    args = build_parser().parse_args(argv)
    config_path = _resolve_path(experiment_dir, args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    for item in args.overrides:
        if "=" not in item:
            raise ValueError(f"--set expects KEY=VALUE, got {item!r}.")
        key, raw_value = item.split("=", 1)
        _set_dotted(config, key.strip(), _parse_override_value(raw_value.strip()))

    if args.run_name:
        config["output"]["run_name"] = args.run_name
    if args.study_name:
        config["study_name"] = args.study_name
    if args.output_root:
        config["output"]["root"] = str(args.output_root)
    if args.device:
        config["runtime"]["device"] = args.device

    config["study_name"] = _sanitize_name(config["study_name"], "study_name")
    output = config["output"]
    if args.output_dir:
        destination = args.output_dir.resolve()
        output["root"] = str(destination.parent)
        output["run_name"] = destination.name
    else:
        output["run_name"] = _sanitize_name(output.get("run_name") or _automatic_run_name(config), "output.run_name")
        output["root"] = str(_resolve_path(experiment_dir, output["root"]))
        destination = Path(output["root"]) / config["study_name"] / output["run_name"]
    output["run_dir"] = str(destination)
    config["cohort"]["source_run"] = str(_resolve_path(experiment_dir, config["cohort"]["source_run"]))
    _validate(config)
    return LoadedExperimentConfig(resolved=config, config_path=config_path)


__all__ = [
    "LoadedExperimentConfig",
    "SCHEMA_VERSION",
    "build_parser",
    "load_experiment_config",
]
