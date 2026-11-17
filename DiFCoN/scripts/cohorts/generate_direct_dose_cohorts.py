"""Generate and audit the fixed LHS direct-dose cohorts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import re
import sys

import numpy as np
import pandas as pd
import scipy

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.cohort import (
    materialize_initial_conditions,
    sample_independent_lhs_cohort,
)
from core.experiment import cohort_csv_path
from core.io import utc_now, write_json


HERE = ROOT
DEFAULT_CONFIG = HERE / "configs" / "direct_dose" / "lhs_cohort.json"
DEFAULT_OUTPUT = HERE / "cohorts"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate deterministic train/validation/test LHS cohort CSV files."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--from-checkpoints", action="store_true",
        help="Read tumour K and the abundance reference from the configured MultiNeD checkpoints.",
    )
    parser.add_argument(
        "--verify-against",
        type=Path,
        help="Saved run or fixed cohort directory whose cohort CSVs must match.",
    )
    return parser


def resolve_path(value: Path) -> Path:
    value = value.expanduser()
    return value.resolve() if value.is_absolute() else (HERE / value).resolve()


def validate_config(config: dict) -> None:
    if int(config.get("schema_version", -1)) != 1:
        raise ValueError("cohort schema_version must be 1.")
    name = str(config.get("cohort_name", ""))
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
        raise ValueError("cohort_name may contain only letters, digits, '_', '-' or '.'.")
    for cohort in ("train", "validation", "test"):
        if int(config["cohort_sizes"][cohort]) < 1:
            raise ValueError(f"cohort_sizes.{cohort} must be >= 1.")
        int(config["lhs_seeds"][cohort])
    if float(config["fitted_baseline"]["tumor_k"]) <= 0:
        raise ValueError("fitted_baseline.tumor_k must be positive.")
    if float(config["fitted_baseline"]["initial_n"]) <= 0:
        raise ValueError("fitted_baseline.initial_n must be positive.")


def expected_paths(output_dir: Path) -> dict[str, Path]:
    paths = {
        cohort: output_dir / f"{cohort}_lhs_cohort.csv"
        for cohort in ("train", "validation", "test")
    }
    paths.update(
        {
            "config": output_dir / "cohort_generation_config.json",
            "manifest": output_dir / "cohort_manifest.json",
        }
    )
    return paths


def verify_cohorts(
    generated: dict[str, pd.DataFrame],
    reference_run: Path,
) -> dict[str, dict]:
    verification: dict[str, dict] = {}
    for cohort, frame in generated.items():
        reference_path = cohort_csv_path(reference_run, cohort)
        reference = pd.read_csv(reference_path)
        pd.testing.assert_frame_equal(
            frame.reset_index(drop=True),
            reference.reset_index(drop=True),
            check_exact=False,
            rtol=1e-12,
            atol=1e-12,
        )
        verification[cohort] = {
            "matches": True,
            "reference_path": str(reference_path),
        }
    return verification


def main() -> None:
    args = build_parser().parse_args()
    config_path = resolve_path(args.config)
    output_dir = resolve_path(args.output_dir)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if args.from_checkpoints:
        from core.reaction_runtime import load_trained_reaction_model

        model, _ = load_trained_reaction_model(device="cpu", **config["model"])
        config["fitted_baseline"] = {
            "tumor_k": model.positive_params()["tumor_k"].item(),
            "initial_n": model.abundance_reference.item(),
        }
        checkpoint = model.loaded_reaction_checkpoint_path
        if checkpoint.is_relative_to(ROOT.parent):
            checkpoint = checkpoint.relative_to(ROOT.parent)
        config.setdefault("provenance", {})["reference_run"] = checkpoint.parent.as_posix()
    validate_config(config)

    cohort_name = str(config["cohort_name"])
    paths = expected_paths(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ranges = config["sampling_ranges"]
    tumor_k = float(config["fitted_baseline"]["tumor_k"])
    initial_n = float(config["fitted_baseline"]["initial_n"])
    generated: dict[str, pd.DataFrame] = {}
    for cohort in ("train", "validation", "test"):
        frame = sample_independent_lhs_cohort(
            int(config["cohort_sizes"][cohort]),
            cohort=cohort,
            seed=int(config["lhs_seeds"][cohort]),
            sampling_ranges=ranges,
        )
        if "tumor_k_abs" not in frame:
            frame["tumor_k_abs"] = tumor_k
        generated[cohort] = materialize_initial_conditions(frame, baseline_n=initial_n)

    verification = None
    if args.verify_against is not None:
        verification = verify_cohorts(generated, resolve_path(args.verify_against))

    for cohort, frame in generated.items():
        frame.to_csv(paths[cohort], index=False)
    write_json(paths["config"], config)

    file_records = {
        key: {
            "path": str(path),
            **({"rows": len(generated[key])} if key in generated else {}),
        }
        for key, path in paths.items()
        if key != "manifest"
    }
    manifest = {
        "created_at_utc": utc_now(),
        "cohort_name": cohort_name,
        "generation_mode": "independent_latin_hypercube",
        "algorithm": {
            "sampler": "scipy.stats.qmc.LatinHypercube",
            "linear_mapping": "low + u * (high - low)",
            "log_mapping": "exp(log(low) + u * (log(high) - log(low)))",
            "patient_seed": "lhs_seed * 1000 + one_based_patient_index",
        },
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
        },
        "cohort_sizes": config["cohort_sizes"],
        "lhs_seeds": config["lhs_seeds"],
        "sampling_ranges": ranges,
        "fitted_baseline": config["fitted_baseline"],
        "provenance": config.get("provenance", {}),
        "verification": verification,
        "files": file_records,
    }
    write_json(paths["manifest"], manifest)

    print(f"Generated fixed cohorts in: {output_dir}")
    if verification is not None:
        print("All generated cohort values match the reference run.")


if __name__ == "__main__":
    main()
