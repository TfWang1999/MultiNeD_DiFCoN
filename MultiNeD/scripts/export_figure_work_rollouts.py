"""Explicit upstream snapshot export for figure_work; fixed evaluation only, no training.

Writes seven files directly to figure_work/source_data.
Preserves the original manuscript evaluation order and random-state settings.
The plotting workspace never imports this module or the training runtime.
"""

import argparse
import copy
import json
import os
import sys
from pathlib import Path

import pandas as pd
import torch

CODE_DIR = Path(__file__).resolve().parents[1]
REPO = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from common.cd8_display_transforms import get_z_display_params, transform_prediction_summary_for_display
from common.cd8_zdist_data import get_z_columns, load_training_package
from scripts.reproduce.upstream_paths import (
    DEFAULT_DRIFT_CHECKPOINT, DEFAULT_REACTION_CHECKPOINT, DRIFT_NAME, REACTION_NAME, VOLUME_TABLE, checkpoint_path,
)
from training.coupled_dynamics import (
    ReactionCoupledModel, _inverse_softplus, apply_reaction_parameter_settings, evaluate_reaction_rollouts,
    load_reaction_checkpoint_configs, summarize_reaction_rollout_mean_predictions,
)
from training.drift_training import load_drift_checkpoint, set_random_seed


STATE_SEED = 2300
TUMOUR_SEED = 2400
N_TIME = 121


def _reaction_effect_values(model):
    effect, softplus = model.reaction_effect, torch.nn.functional.softplus
    eps = float(effect.config.eps)
    return {
        "f_b_coeff": float(effect.f_b_coeff.detach().cpu()),
        "g_b_coeff": float(effect.g_b_coeff.detach().cpu()), "g_pd1_coeff": float(effect.g_pd1_coeff.detach().cpu()),
        "K_P": float((softplus(effect.pd1_k_raw) + eps).detach().cpu()),
        "K_B": float((softplus(effect.b41bb_k_raw) + eps).detach().cpu()),
    }


def evaluate_fixed_rollouts(drift_checkpoint, reaction_checkpoint, device):
    """Evaluate the two fixed rollouts and retrieve the histories from their checkpoints."""
    group_data, _, _, _ = load_training_package(REPO)
    z_cols = get_z_columns(group_data)
    volume_path = Path(reaction_checkpoint).parent / VOLUME_TABLE
    volume_df = pd.read_csv(volume_path)
    if volume_df.empty:
        raise RuntimeError("Tumour-volume data are empty.")

    _, integrator, _, _, drift_history = load_drift_checkpoint(drift_checkpoint, device=device)
    device = next(integrator.parameters()).device
    reaction_config, pk_config, payload = load_reaction_checkpoint_configs(reaction_checkpoint, map_location="cpu")
    integrator.pk_config = copy.deepcopy(pk_config)
    q_in = payload["state_dict"]["q_in"].detach().cpu().numpy()
    reaction_model = ReactionCoupledModel(drift_integrator=copy.deepcopy(integrator), q_in=q_in, config=reaction_config).to(device)
    reaction_model.drift_integrator.pk_config = copy.deepcopy(pk_config)
    reaction_model.load_state_dict(payload["state_dict"], strict=True)
    reaction_model.eval()

    evaluation = evaluate_reaction_rollouts(
        reaction_model, group_data=group_data, volume_df=volume_df, treatment_clean="all", config=reaction_config,
        n_time=N_TIME, n_particles=reaction_config.n_particles, random_state=STATE_SEED)

    manual_model = copy.deepcopy(reaction_model)
    scalar_parameters = {name: float(value.detach().cpu()) for name, value in reaction_model.positive_params().items()}
    manual_settings = {name: {"mode": "fixed", "value": value} for name, value in scalar_parameters.items()}
    apply_reaction_parameter_settings(manual_model, manual_settings, config=manual_model.config)
    effect_parameters = _reaction_effect_values(reaction_model)
    with torch.no_grad():
        effect = manual_model.reaction_effect
        for name in ("f_b_coeff", "g_b_coeff", "g_pd1_coeff"):
            parameter = getattr(effect, name)
            parameter.copy_(torch.tensor(effect_parameters[name], device=parameter.device, dtype=parameter.dtype))
        effect.pd1_k_raw.copy_(torch.tensor(_inverse_softplus(effect_parameters["K_P"]), device=effect.pd1_k_raw.device, dtype=effect.pd1_k_raw.dtype))
        effect.b41bb_k_raw.copy_(torch.tensor(_inverse_softplus(effect_parameters["K_B"]), device=effect.b41bb_k_raw.device, dtype=effect.b41bb_k_raw.dtype))
    manual_evaluation = evaluate_reaction_rollouts(
        manual_model, group_data=group_data, volume_df=volume_df, treatment_clean="all", config=manual_model.config,
        n_time=N_TIME, n_particles=reaction_config.n_particles, random_state=TUMOUR_SEED)

    display_params = get_z_display_params(group_data, z_cols=z_cols, q_low=0.01, q_high=0.99)
    prediction_summary = summarize_reaction_rollout_mean_predictions(evaluation["summary"])
    prediction_summary = transform_prediction_summary_for_display(prediction_summary, display_params)

    return {
        "device": str(device), "z_cols": z_cols,
        "reaction_volume": evaluation["volume"],
        "tumour_volume": manual_evaluation["volume"], "tumour_doses": manual_evaluation["doses"],
        "state_summary": prediction_summary,
        "rollout_random_states": {"state": STATE_SEED, "tumour": TUMOUR_SEED},
        "drift_training_history": drift_history,
        "reaction_training_history": pd.DataFrame(payload["history"]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional separate export destination")
    parser.add_argument("--drift-source", help="Checkpoint or exact drift run/result directory")
    parser.add_argument("--reaction-source", help="Checkpoint or exact joint run/result directory")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    destination = args.output_dir.resolve() if args.output_dir else CODE_DIR / "figure_work" / "source_data"
    keys = ("reaction_volume", "tumour_volume", "tumour_doses", "state_summary")
    drift_checkpoint = (
        checkpoint_path(args.drift_source, DRIFT_NAME, REPO) if args.drift_source else REPO / DEFAULT_DRIFT_CHECKPOINT
    )
    reaction_checkpoint = (
        checkpoint_path(args.reaction_source, REACTION_NAME, REPO)
        if args.reaction_source else REPO / DEFAULT_REACTION_CHECKPOINT
    )
    checkpoints = (drift_checkpoint, reaction_checkpoint)
    required = (*checkpoints, reaction_checkpoint.parent / VOLUME_TABLE)
    for path in required:
        if not path.is_file():
            parser.error(f"Missing input: {path}")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    set_random_seed(STATE_SEED)
    result = evaluate_fixed_rollouts(*checkpoints, device=args.device)
    destination.mkdir(parents=True, exist_ok=True)
    for key in keys:
        result[key].to_csv(destination / (key + ".csv"), index=False)
    for name in ("drift_training_history", "reaction_training_history"):
        result[name].to_csv(destination / (name + ".csv"), index=False)
    metadata = {"global_seed": STATE_SEED, "rollout_random_states": result["rollout_random_states"],
                "evaluation_order": ["state", "tumour"], "device": result["device"], "n_time": N_TIME,
                "z_cols": result["z_cols"], "rows": {k: len(result[k]) for k in keys},
                "checkpoints": [Path(os.path.relpath(p, REPO)).as_posix() for p in checkpoints]}
    (destination / "evaluation_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
