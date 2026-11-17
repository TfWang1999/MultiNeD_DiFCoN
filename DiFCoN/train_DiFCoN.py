"""Train and evaluate the paper DiFCoN policy from one experiment configuration."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import torch

EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = EXPERIMENT_DIR.parent
for path in (EXPERIMENT_DIR, PROJECT_DIR, PROJECT_DIR / "MultiNeD"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from core.cohort import build_patient_specs
from core.config import BUDGET_ID, TOTAL_BUDGET, load_experiment_config
from core.experiment import load_reference_cohorts
from core.io import collect_environment, update_run_status, write_json
from core.losses import DirectLossConfig
from core.policy import DirectDosePolicyConfig, DirectDoseTreatmentCNN
from core.policy_common import fixed_regimen_policies
from core.reaction_runtime import (
    REACTION_RUN_NAME,
    load_project_group_data,
    load_trained_reaction_model,
)
from core.rollout import DEFAULT_FEATURE_NAMES, DirectRolloutConfig
from core.training import (
    DirectTrainConfig,
    action_frame,
    evaluate_strategies,
    save_policy_checkpoint,
    seed_everything,
    train_direct_policy,
    trajectory_frame,
)


def main(argv=None):
    experiment = load_experiment_config(EXPERIMENT_DIR, argv)
    config = experiment.resolved
    output = config["output"]
    run_name = output["run_name"]
    output_dir = Path(output["run_dir"])
    table_dir = output_dir / "tables"
    cohort = config["cohort"]
    reference_run = Path(cohort["source_run"])
    if not reference_run.is_dir():
        raise FileNotFoundError(f"Configured cohort.source_run does not exist: {reference_run}")

    device = config["runtime"]["device"]
    table_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "resolved_config.json", config)
    write_json(output_dir / "environment.json", collect_environment(device))
    update_run_status(output_dir, "running", run_name=run_name, study_name=config["study_name"])

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    seeds = config["seeds"]
    seed_everything(seeds["model"])
    group_data = load_project_group_data()
    reaction_model, model_device = load_trained_reaction_model(device=device, **config["model"])
    baseline = reaction_model.positive_params()
    reference_n = max(10.0, 2.0 * float(reaction_model.abundance_reference))

    policy_settings = dict(config["policy"])
    policy_settings["feature_names"] = policy_settings["feature_names"] or DEFAULT_FEATURE_NAMES
    policy_config = DirectDosePolicyConfig(**policy_settings)
    simulation = dict(config["simulation"])
    n_particles_train = simulation.pop("n_particles_train")
    n_particles_test = simulation.pop("n_particles_test")
    rollout_config = DirectRolloutConfig(
        **simulation,
        history_length=policy_config.history_length,
        feature_names=policy_config.feature_names,
        reference_C=6000.0,
        reference_N=reference_n,
    )
    loss_config = DirectLossConfig(**config["loss"])
    train_config = DirectTrainConfig(**config["training"], random_state=seeds["batch_order"])

    train_cohort, validation_cohort, test_cohort = load_reference_cohorts(
        reference_run, cohort["n_train"], cohort["n_validation"], cohort["n_test"],
    )
    for split, frame in (
        ("train", train_cohort), ("validation", validation_cohort), ("test", test_cohort),
    ):
        if "tumor_k_abs" not in frame:
            frame["tumor_k_abs"] = float(baseline["tumor_k"].detach().cpu())
        frame.to_csv(table_dir / f"{split}_lhs_cohort.csv", index=False)
    train_specs = build_patient_specs(
        train_cohort, group_data, reaction_model, n_particles_train, seeds["train_rollout"],
    )
    validation_specs = build_patient_specs(
        validation_cohort, group_data, reaction_model, n_particles_train, seeds["validation_rollout"],
    )
    test_specs = build_patient_specs(
        test_cohort, group_data, reaction_model, n_particles_test, seeds["test_rollout"],
    )

    print(f"loaded checkpoint: {reaction_model.loaded_reaction_checkpoint_path}")
    print(f"device={model_device} train={len(train_specs)} validation={len(validation_specs)} test={len(test_specs)}")
    policy = DirectDoseTreatmentCNN(policy_config).to(model_device)
    history = train_direct_policy(
        reaction_model, policy, train_specs, validation_specs, rollout_config, loss_config, train_config,
    )
    history.to_csv(table_dir / "training_history.csv", index=False)

    model_identity = {
        "reaction_run_name": REACTION_RUN_NAME,
        "reaction_checkpoint": str(reaction_model.loaded_reaction_checkpoint_path),
        "policy_type": policy.policy_type,
        "budget_id": BUDGET_ID,
        "total_budget": TOTAL_BUDGET,
        "seeds": dict(seeds),
        "best_epoch": int(history.loc[history["validation_loss"].idxmin(), "epoch"]),
        "best_validation_loss": float(history["validation_loss"].min()),
    }
    checkpoint_path = save_policy_checkpoint(
        output_dir / "checkpoints" / output["checkpoint_filename"],
        policy, policy_config, rollout_config, loss_config, train_config, model_identity, history,
    )
    print(f"saved checkpoint: {checkpoint_path}")

    def save_cohort_summary(frame: pd.DataFrame, split: str) -> None:
        metric_columns = [
            "loss", "final_C_rel", "tumor_auc_rel", "total_dose", "loss_late_activity",
            "loss_functional_activity_loss", "loss_smoothness_loss", "loss_drug_efficiency",
        ]
        means = frame.groupby("strategy", observed=True)[metric_columns].mean().reset_index()
        direct = means[means["strategy"] == "Optimal"].iloc[0]
        write_json(output_dir / "summaries" / f"{split}_summary.json", {
            "schema_version": config["schema_version"],
            "study_name": config["study_name"],
            "run_name": run_name,
            "cohort": split,
            "policy_type": policy.policy_type,
            "strategy_label": "Optimal",
            "loss_weights": loss_config.to_dict(),
            "strategy_means": means.to_dict(orient="records"),
            "direct_dose_means": direct.to_dict(),
            "n_patients": int(frame["patient_id"].nunique()),
        })

    validation_summary = pd.DataFrame(
        summary for summary, _ in evaluate_strategies(
            reaction_model, {"Optimal": policy}, validation_specs, rollout_config, loss_config,
        )
    )
    save_cohort_summary(validation_summary, "validation")

    test_strategies = fixed_regimen_policies(
        rollout_config.decision_days(),
        {"anti_b41bb": "Anti-41BB", "anti_pd1": "Anti-PD1", "combine": "Combine"},
        model_device,
    )
    test_strategies["Optimal"] = policy
    test_rows, test_trajectories, test_actions = [], [], []
    for summary, trajectory in evaluate_strategies(
        reaction_model, test_strategies, test_specs, rollout_config, loss_config,
    ):
        test_rows.append(summary)
        test_trajectories.append(trajectory_frame(summary, summary["strategy"], trajectory))
        test_actions.append(action_frame(summary, summary["strategy"], trajectory))

    test_summary = pd.DataFrame(test_rows)
    pd.concat(test_trajectories, ignore_index=True).to_csv(table_dir / "test_trajectories.csv", index=False)
    pd.concat(test_actions, ignore_index=True).to_csv(table_dir / "test_actions.csv", index=False)
    save_cohort_summary(test_summary, "test")

    def combine_response_class(final_C_rel: float) -> str:
        if final_C_rel <= 0.05:
            return "deep_response"
        if final_C_rel < 0.50:
            return "partial_response"
        return "nonresponse"

    combine_reference = test_summary[test_summary["strategy"] == "Combine"].set_index("patient_id")["final_C_rel"]
    test_summary["combine_response_class"] = test_summary["patient_id"].map(combine_reference.map(combine_response_class))
    test_summary.to_csv(table_dir / "test_strategy_summary.csv", index=False)
    update_run_status(
        output_dir, "completed", run_name=run_name, best_epoch=model_identity["best_epoch"],
        best_validation_loss=model_identity["best_validation_loss"],
    )
    print("train_DiFCoN.py completed successfully.")


if __name__ == "__main__":
    main()
