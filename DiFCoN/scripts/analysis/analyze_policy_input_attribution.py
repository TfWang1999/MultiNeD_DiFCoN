from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
PROJECT_ROOT = ROOT.parent
for _path in (ROOT, PROJECT_ROOT / "MultiNeD"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from scripts._direct_dose_evaluation_utils import (
    limit_patients,
    load_frozen_cohort,
    load_saved_policy_context,
    make_output_dir,
    make_specs,
)
from scripts.benchmarking.policy_benchmark_common import load_config
from core.io import write_json
from core.training import evaluate_patient


DEFAULT_CONFIG = ROOT / "configs" / "analysis" / "input_attribution.json"
OUTPUT_LABELS = ("daily_total_logit", "pd1_fraction_logit")


class HistoryInterventionPolicy(nn.Module):
    """Apply a pre-specified history intervention without changing policy weights."""

    def __init__(self, base_policy: nn.Module, mode: str):
        super().__init__()
        self.base_policy = base_policy
        self.mode = str(mode)

    def transform(self, history: torch.Tensor) -> torch.Tensor:
        if self.mode == "current_only":
            return history[:, -1:, :].repeat(1, history.shape[1], 1)
        if self.mode == "reverse_past":
            if history.shape[1] <= 1:
                return history
            return torch.cat([torch.flip(history[:, :-1, :], dims=(1,)), history[:, -1:, :]], dim=1)
        raise ValueError(f"Unknown history intervention: {self.mode}")

    def act(self, history: torch.Tensor, step_index: int, remaining_budget: torch.Tensor) -> torch.Tensor:
        return self.base_policy.act(self.transform(history), step_index, remaining_budget)


class RecordingPolicy(nn.Module):
    def __init__(self, base_policy: nn.Module):
        super().__init__()
        self.base_policy = base_policy
        self.histories: list[torch.Tensor] = []

    def act(self, history: torch.Tensor, step_index: int, remaining_budget: torch.Tensor) -> torch.Tensor:
        if step_index > 0:
            self.histories.append(history.detach())
        return self.base_policy.act(history, step_index, remaining_budget)


def _input_jacobian(policy: nn.Module, history: torch.Tensor) -> np.ndarray:
    value = history.detach().clone().requires_grad_(True)
    jacobian = torch.autograd.functional.jacobian(
        lambda item: policy.action_logits(item).reshape(2), value, create_graph=False
    )
    return jacobian[:, 0, :, :].detach().cpu().numpy()


def _integrated_gradients(
    policy: nn.Module,
    history: torch.Tensor,
    steps: int,
) -> np.ndarray:
    target = history.detach()
    baseline = torch.zeros_like(target)
    accumulated = torch.zeros((2, *target.shape), device=target.device, dtype=target.dtype)
    for alpha in torch.linspace(0.0, 1.0, int(steps), device=target.device, dtype=target.dtype):
        value = (baseline + alpha * (target - baseline)).detach().requires_grad_(True)
        logits = policy.action_logits(value).reshape(2)
        for output_index in range(2):
            gradient = torch.autograd.grad(
                logits[output_index], value, retain_graph=output_index == 0
            )[0]
            accumulated[output_index] += gradient
    integrated = (target - baseline).unsqueeze(0) * accumulated / float(steps)
    return integrated[:, 0, :, :].detach().cpu().numpy()


def _attribution_rows(
    policy,
    history: torch.Tensor,
    feature_names: tuple[str, ...],
    ig_steps: int,
) -> list[dict[str, Any]]:
    jacobian = _input_jacobian(policy, history)
    integrated = _integrated_gradients(policy, history, ig_steps)
    values = history.detach().cpu().numpy()[0]
    n_lags = values.shape[0]
    rows: list[dict[str, Any]] = []
    for output_index, output_name in enumerate(OUTPUT_LABELS):
        for lag_index in range(n_lags):
            for feature_index, feature_name in enumerate(feature_names):
                rows.append(
                    {
                        "output": output_name,
                        "lag_from_current": int(lag_index - n_lags + 1),
                        "feature": str(feature_name),
                        "input_value": float(values[lag_index, feature_index]),
                        "absolute_gradient": float(abs(jacobian[output_index, lag_index, feature_index])),
                        "absolute_integrated_gradient": float(abs(integrated[output_index, lag_index, feature_index])),
                    }
                )
    return rows


def _summarize_attribution(analysis: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_sd = (
        analysis.groupby("feature", observed=True)["input_value"]
        .std(ddof=1)
        .fillna(0.0)
        .rename("feature_sd")
    )
    analysis = analysis.merge(feature_sd, on="feature", how="left", validate="many_to_one")
    analysis["scaled_absolute_gradient"] = analysis["absolute_gradient"] * analysis["feature_sd"]
    summary = (
        analysis.groupby(["output", "lag_from_current", "feature"], observed=True)
        .agg(
            n=("absolute_gradient", "size"),
            absolute_gradient_mean=("absolute_gradient", "mean"),
            scaled_absolute_gradient_mean=("scaled_absolute_gradient", "mean"),
            absolute_integrated_gradient_mean=("absolute_integrated_gradient", "mean"),
            absolute_integrated_gradient_median=("absolute_integrated_gradient", "median"),
        )
        .reset_index()
    )
    lag_summary = (
        analysis.groupby(["output", "lag_from_current"], observed=True)
        .agg(
            scaled_absolute_gradient_sum=("scaled_absolute_gradient", "sum"),
            absolute_integrated_gradient_sum=("absolute_integrated_gradient", "sum"),
        )
        .reset_index()
    )
    lag_summary["history_role"] = np.where(
        lag_summary["lag_from_current"] == 0, "current", "past"
    )
    denominators = lag_summary.groupby("output", observed=True)[
        ["scaled_absolute_gradient_sum", "absolute_integrated_gradient_sum"]
    ].transform("sum")
    lag_summary["scaled_gradient_fraction"] = (
        lag_summary["scaled_absolute_gradient_sum"]
        / denominators["scaled_absolute_gradient_sum"].clip(lower=1e-12)
    )
    lag_summary["integrated_gradient_fraction"] = (
        lag_summary["absolute_integrated_gradient_sum"]
        / denominators["absolute_integrated_gradient_sum"].clip(lower=1e-12)
    )
    return summary, lag_summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Quantify how frozen policy outputs depend on past and current trajectory features."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-patients", type=int, default=None)
    parser.add_argument("--policy-root", type=Path, help="Dose-sweep study containing the three weight-1 policies")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.policy_root:
        for item in config["policy_runs"]:
            item["run_dir"] = str(args.policy_root / f"dose_weight_1p0_seed_{item['model_seed']}")
    if args.device is not None:
        config["device"] = args.device
    if args.max_patients is not None:
        config["max_patients"] = int(args.max_patients)

    output_dir = make_output_dir(args.output_dir or config["output_dir"])
    table_dir = output_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    contexts = [
        load_saved_policy_context(item["run_dir"], device=str(config["device"]))
        for item in config["policy_runs"]
    ]
    cohort, _ = load_frozen_cohort(config["cohort_dir"], config["cohort"])
    cohort = limit_patients(cohort, int(config["max_patients"]))

    attribution_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    for item, context in zip(config["policy_runs"], contexts):
        policy_seed = int(item["model_seed"])
        specs = make_specs(
            context,
            cohort,
            n_particles=int(config["n_particles"]),
            rollout_seed=int(config["evaluation_rollout_seed"]),
        )
        for spec in specs:
            recorder = RecordingPolicy(context.policy).to(context.device)
            summary, _ = evaluate_patient(
                context.reaction_model,
                recorder,
                spec,
                context.rollout_config,
                context.loss_config,
                "observed_history",
            )
            metric_rows.append({**summary, "policy_seed": policy_seed, "history_intervention": "observed"})
            for history in recorder.histories:
                attribution_rows.extend(
                    _attribution_rows(
                        context.policy,
                        history,
                        context.policy.config.feature_names,
                        int(config["integrated_gradient_steps"]),
                    )
                )

            for mode in config["history_interventions"]:
                intervention = HistoryInterventionPolicy(context.policy, str(mode)).to(context.device)
                summary, _ = evaluate_patient(
                    context.reaction_model,
                    intervention,
                    spec,
                    context.rollout_config,
                    context.loss_config,
                    str(mode),
                )
                metric_rows.append({**summary, "policy_seed": policy_seed, "history_intervention": str(mode)})

    attribution = pd.DataFrame(attribution_rows)
    metrics = pd.DataFrame(metric_rows)

    attribution_summary, lag_summary = _summarize_attribution(attribution)
    attribution_summary.to_csv(table_dir / "policy_input_attribution_summary.csv", index=False)
    lag_summary.to_csv(table_dir / "policy_input_lag_summary.csv", index=False)

    pairing_key = ["policy_seed", "patient_id"]
    baseline = metrics[metrics["history_intervention"] == "observed"][
        pairing_key + ["loss", "final_C_rel", "tumor_auc_rel", "tumor_regression", "total_dose"]
    ]
    paired = metrics.merge(
        baseline,
        on=pairing_key,
        how="left",
        validate="many_to_one",
        suffixes=("", "_observed"),
    )
    for metric in ("loss", "final_C_rel", "tumor_auc_rel", "tumor_regression", "total_dose"):
        paired[f"delta_{metric}"] = paired[metric] - paired[f"{metric}_observed"]
    paired.to_csv(table_dir / "history_intervention_paired_metrics.csv", index=False)

    write_json(output_dir / "resolved_config.json", config)
    write_json(
        output_dir / "run_status.json",
        {
            "status": "completed",
            "attribution_rows": int(attribution.shape[0]),
            "metric_rows": int(metrics.shape[0]),
            "policy_seeds": int(metrics["policy_seed"].nunique()),
            "patients": int(metrics["patient_id"].nunique()),
        },
    )
    print(f"[policy-input-attribution] completed output={output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
