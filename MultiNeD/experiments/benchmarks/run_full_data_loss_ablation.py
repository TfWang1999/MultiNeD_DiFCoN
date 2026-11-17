from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any

import pandas as pd


CODE_DIR = Path(__file__).resolve().parents[2]
PROJECT_ROOT = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from experiments.benchmarks.dynamics_benchmark_common import (
    environment_record,
    fit_full_project_model,
    load_base_validation_config,
    load_config,
    prepare_run_directory,
    write_json,
)
from experiments.benchmarks.framework_ablation_common import train_nearest_neighbor_drift
from experiments.benchmarks.full_data_loss_ablation_common import (
    evaluate_full_data_model,
    load_notebook_pretrained_drift,
    load_notebook_q_in,
)
from experiments.volume_loto.model_runner import load_inputs


DEFAULT_CONFIG = CODE_DIR / "configs" / "benchmarking" / "training_loss_ablation.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run full-B16 in-sample loss and pretraining ablations without holdouts."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--variants", default=None, help="Comma-separated configured variants.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--run-id")
    target.add_argument("--output-dir", type=Path, help="Directory for this ablation run")
    return parser.parse_args()


def resolve_config(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    config = load_config(args.config)
    base = load_base_validation_config(config["base_validation_config"])
    configured = list(config["variants"])
    if args.variants:
        requested = [value.strip() for value in args.variants.split(",") if value.strip()]
        unknown = sorted(set(requested) - set(configured))
        if unknown:
            raise ValueError(f"Requested variants are not configured: {unknown}")
        config["variants"] = requested
    if args.seed is not None:
        config["seeds"] = [int(args.seed)]
    if args.device is not None:
        config["device"] = str(args.device)
    base["drift"] = {**base["drift"], **copy.deepcopy(config.get("drift", {}))}
    base["reaction"] = {**base["reaction"], **copy.deepcopy(config.get("reaction", {}))}
    base["evaluation"] = {**base["evaluation"], **copy.deepcopy(config.get("evaluation", {}))}
    base["drift"]["device"] = config.get("device", "auto")
    base["reaction"]["device"] = config.get("device", "auto")
    base["evaluation"].update(
        device=config.get("device", "auto"),
        z_scale="zstd",
        sinkhorn_epsilon=base["reaction"]["sinkhorn_epsilon"],
        sinkhorn_iters=base["reaction"]["sinkhorn_iters"],
    )
    if not bool(base["reaction"]["train_drift"]):
        raise ValueError(
            "The paper loss ablation requires reaction.train_drift=true"
        )
    config["base_validation_resolved"] = base
    return config, base


def variant_definition(variant: str) -> dict[str, Any]:
    definitions = {
        "balanced_ot_local_rollout_pretrained": {
            "drift_supervision": "balanced_sinkhorn",
            "drift_loss_mode": "local_rollout",
            "reaction_distribution_loss": "balanced_sinkhorn",
            "reaction_loss_mode": "local_rollout",
            "drift_pretrained": True,
            "drift_initialization": "notebook_pretrained_checkpoint",
            "reaction_trains_drift": True,
            "changed_from_notebook_reference": [],
        },
        "balanced_ot_local_only_pretrained": {
            "drift_supervision": "balanced_sinkhorn",
            "drift_loss_mode": "local",
            "reaction_distribution_loss": "balanced_sinkhorn",
            "reaction_loss_mode": "local",
            "drift_pretrained": True,
            "drift_initialization": "full_data_refit",
            "reaction_trains_drift": True,
            "changed_from_notebook_reference": [
                "drift_loss_mode",
                "reaction_loss_mode",
            ],
        },
        "balanced_ot_rollout_only_pretrained": {
            "drift_supervision": "balanced_sinkhorn",
            "drift_loss_mode": "rollout",
            "reaction_distribution_loss": "balanced_sinkhorn",
            "reaction_loss_mode": "rollout",
            "drift_pretrained": True,
            "drift_initialization": "full_data_refit",
            "reaction_trains_drift": True,
            "changed_from_notebook_reference": [
                "drift_loss_mode",
                "reaction_loss_mode",
            ],
        },
        "balanced_ot_local_rollout_no_pretrain": {
            "drift_supervision": "none_before_reaction",
            "drift_loss_mode": "none_before_reaction",
            "reaction_distribution_loss": "balanced_sinkhorn",
            "reaction_loss_mode": "local_rollout",
            "drift_pretrained": False,
            "drift_initialization": "random",
            "reaction_trains_drift": True,
            "changed_from_notebook_reference": ["drift_pretraining"],
        },
        "nearest_neighbor_mse_local_rollout_pretrained": {
            "drift_supervision": "symmetric_nearest_neighbor_mse",
            "drift_loss_mode": "local_rollout",
            "reaction_distribution_loss": "balanced_sinkhorn",
            "reaction_loss_mode": "local_rollout",
            "drift_pretrained": True,
            "drift_initialization": "full_data_refit",
            "reaction_trains_drift": True,
            "changed_from_notebook_reference": ["drift_distribution_loss"],
        },
    }
    if variant not in definitions:
        raise KeyError(variant)
    return definitions[variant]


def fit_variant(
    variant: str,
    base: dict[str, Any],
    bundle: dict[str, Any],
    output_dir: Path,
    seed: int,
) -> dict[str, Any]:
    if variant == "balanced_ot_local_rollout_pretrained":
        fit_options = {"prefit_drift": load_notebook_pretrained_drift(base)}
    elif variant in ("balanced_ot_local_only_pretrained", "balanced_ot_rollout_only_pretrained"):
        definition = variant_definition(variant)
        fit_options = {
            "drift_overrides": {"loss_mode": definition["drift_loss_mode"]},
            "reaction_overrides": {"loss_mode": definition["reaction_loss_mode"]},
        }
    elif variant == "balanced_ot_local_rollout_no_pretrain":
        fit_options = {
            "initialize_drift_randomly": True,
            "reaction_overrides": {"loss_mode": "local_rollout"},
        }
    elif variant == "nearest_neighbor_mse_local_rollout_pretrained":
        prefit = train_nearest_neighbor_drift(
            base,
            bundle["group_data"],
            bundle["transitions"],
            output_dir / "nearest_neighbor_drift",
            seed,
        )
        fit_options = {"prefit_drift": prefit}
    else:
        raise KeyError(variant)
    return fit_full_project_model(
        base,
        bundle["group_data"],
        bundle["transitions"],
        bundle["volume"],
        output_dir,
        seed,
        fixed_q_in=bundle["fixed_q_in"],
        fixed_q_in_info=bundle["fixed_q_in_info"],
        **fit_options,
    )


def main() -> int:
    args = parse_args()
    config, base = resolve_config(args)
    bundle = load_inputs(base, PROJECT_ROOT)
    fixed_q_in, fixed_q_in_info = load_notebook_q_in(base)
    bundle["fixed_q_in"] = fixed_q_in
    bundle["fixed_q_in_info"] = fixed_q_in_info
    if args.output_dir:
        run_dir = args.output_dir.resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        run_name = f"full_data_loss_seed{config['seeds'][0]}" if len(config["seeds"]) == 1 else "full_data_loss"
        run_dir = prepare_run_directory(config["output_root"], args.run_id or run_name)
    write_json(run_dir / "resolved_config.json", config)
    write_json(run_dir / "environment.json", environment_record(config.get("device", "auto")))
    for seed in map(int, config["seeds"]):
        for variant in config["variants"]:
            variant_dir = run_dir / f"seed_{seed}" / variant
            expected = [
                variant_dir / "transition_distribution_metrics.csv",
                variant_dir / "rollout_distribution_metrics.csv",
                variant_dir / "volume_metrics.csv",
            ]
            variant_dir.mkdir(parents=True, exist_ok=True)
            definition = variant_definition(variant)
            write_json(variant_dir / "variant_definition.json", definition)
            trained = fit_variant(variant, base, bundle, variant_dir / "fitted_model", seed)
            transition, rollout, volume = evaluate_full_data_model(
                trained,
                bundle["group_data"],
                bundle["transitions"],
                bundle["volume"],
                base["evaluation"],
                seed,
                variant,
            )
            transition.to_csv(expected[0], index=False)
            rollout.to_csv(expected[1], index=False)
            volume.to_csv(expected[2], index=False)
            del trained
    write_json(
        run_dir / "run_status.json",
        {
            "status": "completed",
            "completed": len(config["seeds"]) * len(config["variants"]),
            "uses_holdouts": False,
        },
    )
    print(f"[full-data-loss-ablations] output={run_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
