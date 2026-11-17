from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch

from .config import DAILY_TOTAL_DOSE_CAP, TOTAL_BUDGET
from .policy_common import append_history, make_initial_history


DEFAULT_FEATURE_NAMES = (
    "score_precursor",
    "score_activation",
    "score_eff_early",
    "score_eff_cytox",
    "score_dysfunction",
    "previous_action_pd1",
    "previous_action_b41bb",
    "log_C_abs",
    "log_C_rel",
    "log_N_abs",
    "log_N_rel",
    "activity_scaled",
    "P_conc_scaled",
    "B_conc_scaled",
    "remaining_budget_fraction",
    "time_fraction",
)


@dataclass
class DirectRolloutConfig:
    """Settings for a differentiable policy rollout for one virtual patient."""

    start_day: float = 1.0
    end_day: float = 17.0
    decision_dt: float = 1.0
    internal_dt: float = 0.1
    history_length: int = 5
    concentration_cap: float = 10.0
    feature_names: tuple[str, ...] = DEFAULT_FEATURE_NAMES
    reference_C: float = 3000.0
    reference_N: float = 10.0
    reference_activity: float = 2.0

    def __post_init__(self):
        self.feature_names = tuple(self.feature_names)
        unknown = set(self.feature_names) - set(DEFAULT_FEATURE_NAMES)
        if unknown:
            raise ValueError(f"Unknown history features: {sorted(unknown)}")

    def decision_days(self) -> np.ndarray:
        return np.arange(self.start_day, self.end_day, self.decision_dt, dtype=float)


CORE_PARAMETER_SCALES = {
    "tumor_r_scale": "tumor_r",
    "eta_scale": "killing",
    "source_rate_scale": "source_rate",
    "lambda_act_scale": "lambda_act",
    "death_rate_scale": "death_rate",
    "stim_k_scale": "stim_k",
}


def reset_rollout_rng(seed: int) -> None:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def apply_daily_total_dose_cap(action: torch.Tensor) -> torch.Tensor:
    """Project the combined dose of both drugs onto the daily dose constraint."""
    scale = torch.clamp(action.new_tensor(DAILY_TOTAL_DOSE_CAP) / (action.sum() + 1e-8), max=1.0)
    return action * scale


def positive_params_for_patient(model, spec: dict) -> dict[str, torch.Tensor]:
    """Apply the LHS patient-specific multipliers to the fitted parameters."""

    params = dict(model.positive_params())
    for scale_name, model_name in CORE_PARAMETER_SCALES.items():
        params[model_name] = params[model_name] * float(spec.get(scale_name, 1.0))
    if "tumor_k_abs" in spec:
        params["tumor_k"] = params["tumor_k"].new_tensor(float(spec["tumor_k_abs"]))
    elif "tumor_k_scale" in spec:
        params["tumor_k"] = params["tumor_k"] * float(spec["tumor_k_scale"])
    return params


def _effect_baseline(model) -> dict[str, torch.Tensor]:
    effect = model.reaction_effect
    eps = float(effect.config.eps)
    return {
        "f_b_coeff": effect.f_b_coeff,
        "g_b_coeff": effect.g_b_coeff,
        "g_pd1_coeff": effect.g_pd1_coeff,
        "K_P": torch.nn.functional.softplus(effect.pd1_k_raw) + eps,
        "K_B": torch.nn.functional.softplus(effect.b41bb_k_raw) + eps,
    }


def _effect_value(values: dict[str, torch.Tensor], spec: dict, name: str) -> torch.Tensor:
    return values[name] * float(spec.get(f"{name}_scale", 1.0))


def patient_reaction_effects(
    model,
    pd1_conc: torch.Tensor,
    b41bb_conc: torch.Tensor,
    spec: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute patient-specific G(P, B) and F(B) using the fixed forms in the manuscript."""

    values = _effect_baseline(model)
    P, B = pd1_conc, b41bb_conc
    eps = P.new_tensor(float(model.reaction_effect.config.eps))

    F_PB = 1.0 + _effect_value(values, spec, "f_b_coeff") * B / (_effect_value(values, spec, "K_B") + B + eps)
    p_part = _effect_value(values, spec, "g_pd1_coeff") * P / (_effect_value(values, spec, "K_P") + P + eps)
    b_part = _effect_value(values, spec, "g_b_coeff") * B / (_effect_value(values, spec, "K_B") + B + eps)
    G_PB = 1.0 + p_part + b_part
    return G_PB, F_PB


def _drift_velocity(integrator, z, pd1_conc, b41bb_conc, day: float):
    treatment = torch.stack(
        [pd1_conc.expand(z.shape[0]), b41bb_conc.expand(z.shape[0])],
        dim=1,
    )
    time = torch.full(
        (z.shape[0], 1),
        float(day) / float(integrator.time_scale),
        device=z.device,
        dtype=z.dtype,
    )
    return integrator.model(z, treatment, time)


def _drift_step(integrator, z, pd1_conc, b41bb_conc, day0: float, dt: float):
    n_steps = max(1, int(round(float(dt) * int(getattr(integrator, "n_steps_per_day", 10)))))
    h = float(dt) / n_steps
    out = z
    for step in range(n_steps):
        day = float(day0) + step * h
        out = out + h * _drift_velocity(integrator, out, pd1_conc, b41bb_conc, day)
    return out


def one_internal_step(model, z, weights, C, n, P, B, day, dt, params, spec):
    G_PB, F_PB = patient_reaction_effects(model, P, B, spec)
    z_next = _drift_step(model.drift_integrator, z, P, B, day, dt)
    activity = model.effector_activity(z_next, weights)

    # A_CD8(t) = eta_eff * a(t) * n(t) is the killing rate in the tumor equation.
    A_CD8 = params["killing"] * activity * n
    dC = params["tumor_r"] * C * (1.0 - C / params["tumor_k"]) - A_CD8 * C
    tumor_stim = C / (params["stim_k"] + C + 1e-8)
    source = params["source_rate"] * F_PB
    activation = params["lambda_act"] * G_PB * tumor_stim
    C_next = torch.clamp(C + float(dt) * dC, min=1e-3)
    z_next, weights_next, n_next = model.inflow_step(
        z_next, weights, n, activation - params["death_rate"], source, dt, model.sample_inflow_indices(),
    )

    pk = model.drift_integrator.pk_config
    delta_P = float(pk.delta_pd1) * float(spec.get("pd1_clearance_scale", 1.0))
    delta_B = float(pk.delta_b41bb) * float(spec.get("b41bb_clearance_scale", 1.0))
    P_next = P * torch.exp(P.new_tensor(-delta_P * float(dt)))
    B_next = B * torch.exp(B.new_tensor(-delta_B * float(dt)))
    return z_next, weights_next, C_next, n_next, P_next, B_next


def score_features(z: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid((z * weights[:, None]).sum(dim=0))


def build_history_feature(
    scores,
    activity,
    C,
    N,
    P,
    B,
    previous_action,
    remaining_budget,
    current_day: float,
    C0,
    N0,
    config: DirectRolloutConfig,
) -> torch.Tensor:
    """Build the network input for one time point in ``feature_names`` order."""

    horizon = max(float(config.end_day) - float(config.start_day), 1e-8)
    values = {
        "score_precursor": scores[0],
        "score_activation": scores[1],
        "score_eff_early": scores[2],
        "score_eff_cytox": scores[3],
        "score_dysfunction": scores[4],
        "previous_action_pd1": previous_action[0],
        "previous_action_b41bb": previous_action[1],
        "log_C_abs": torch.log1p(C) / C.new_tensor(math.log1p(float(config.reference_C))),
        "log_C_rel": torch.log(torch.clamp(C / C0, min=1e-8)),
        "log_N_abs": torch.log1p(N) / N.new_tensor(math.log1p(float(config.reference_N))),
        "log_N_rel": torch.log(torch.clamp(N / N0, min=1e-8)),
        "activity_scaled": torch.log1p(activity) / activity.new_tensor(math.log1p(float(config.reference_activity))),
        "P_conc_scaled": P / float(config.concentration_cap),
        "B_conc_scaled": B / float(config.concentration_cap),
        "remaining_budget_fraction": remaining_budget / TOTAL_BUDGET,
        "time_fraction": C.new_tensor((float(current_day) - float(config.start_day)) / horizon),
    }
    return torch.stack([values[name].reshape(()) for name in config.feature_names])


def rollout_patient(model, policy, spec: dict, config: DirectRolloutConfig) -> dict[str, torch.Tensor]:
    """Run an end-to-end differentiable rollout with the policy using only the patient's trajectory history."""

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    z = spec["z0"].to(device=device, dtype=dtype)
    weights = z.new_full((len(z),), 1.0 / len(z))
    params = positive_params_for_patient(model, spec)
    C = torch.as_tensor(float(spec["C0"]), device=device, dtype=dtype).reshape(())
    n = torch.as_tensor(float(spec["n0"]), device=device, dtype=dtype).reshape(())
    N = n * model.abundance_reference
    C0, N0 = C.detach(), N.detach()
    P, B = C.new_tensor(0.0), C.new_tensor(0.0)
    previous_action = C.new_zeros(2)
    remaining_budget = C.new_tensor(TOTAL_BUDGET)
    current_day = float(config.start_day)
    activity = model.effector_activity(z, weights)
    scores = score_features(z, weights)
    A_CD8 = params["killing"] * activity * n

    first_feature = build_history_feature(
        scores, activity, C, N, P, B, previous_action, remaining_budget,
        current_day, C0, N0, config,
    )
    history = make_initial_history(first_feature, config.history_length).to(device=device, dtype=dtype)

    times, C_values, N_values = [C.new_tensor(current_day)], [C], [N]
    activities, A_values = [activity], [A_CD8]
    score_values = [scores]
    P_values, B_values = [P], [B]
    budgets = [remaining_budget]
    actions = []
    step_index = 0

    while current_day < float(config.end_day) - 1e-9:
        action = policy.act(history, step_index, remaining_budget)
        action = apply_daily_total_dose_cap(action)
        actions.append(action)
        remaining_budget = torch.clamp(remaining_budget - action.sum(), min=0.0)
        P = torch.clamp(P + action[0], 0.0, float(config.concentration_cap))
        B = torch.clamp(B + action[1], 0.0, float(config.concentration_cap))

        target_day = min(current_day + float(config.decision_dt), float(config.end_day))
        while current_day < target_day - 1e-9:
            dt = min(float(config.internal_dt), target_day - current_day)
            z, weights, C, n, P, B = one_internal_step(
                model, z, weights, C, n, P, B, current_day, dt, params, spec
            )
            current_day += dt

        previous_action = action
        activity = model.effector_activity(z, weights)
        scores = score_features(z, weights)
        A_CD8 = params["killing"] * activity * n
        N = n * model.abundance_reference
        feature = build_history_feature(
            scores, activity, C, N, P, B, previous_action, remaining_budget,
            current_day, C0, N0, config,
        )
        history = append_history(history, feature)
        times.append(C.new_tensor(current_day))
        C_values.append(C)
        N_values.append(N)
        activities.append(activity)
        A_values.append(A_CD8)
        score_values.append(scores)
        P_values.append(P)
        B_values.append(B)
        budgets.append(remaining_budget)
        step_index += 1

    action_tensor = torch.stack(actions) if actions else C.new_zeros((0, 2))
    return {
        "time": torch.stack(times),
        "volume": torch.stack(C_values),
        "N": torch.stack(N_values),
        "activity": torch.stack(activities),
        "A_CD8": torch.stack(A_values),
        "scores": torch.stack(score_values),
        "P_conc": torch.stack(P_values),
        "B_conc": torch.stack(B_values),
        "remaining_budget": torch.stack(budgets),
        "actions": action_tensor,
    }


__all__ = [
    "DEFAULT_FEATURE_NAMES",
    "DirectRolloutConfig",
    "apply_daily_total_dose_cap",
    "build_history_feature",
    "patient_reaction_effects",
    "positive_params_for_patient",
    "reset_rollout_rng",
    "rollout_patient",
]
