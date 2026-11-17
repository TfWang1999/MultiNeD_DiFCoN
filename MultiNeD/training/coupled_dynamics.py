
from pathlib import Path as _PathForBootstrap
import sys as _sysForBootstrap

_CODE_DIR = _PathForBootstrap(__file__).resolve().parents[1]
if str(_CODE_DIR) not in _sysForBootstrap.path:
    _sysForBootstrap.path.insert(0, str(_CODE_DIR))

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import pandas as pd

from common.cd8_model_components import PKConfig, ReactionEffectConfig, ReactionEffectTransform, treatment_values_at_day
from common.cd8_ot_losses import sinkhorn_divergence
from training.drift_training import choose_device
from common.cd8_zdist_data import mouse_balanced_sample


@dataclass
class ReactionCoupledConfig:
    """Training configuration for the reaction-coupled particle model."""
    n_epochs: int = 200
    n_particles: int = 256
    n_inflow_particles: int = 1
    dt: float = 0.1
    lr: float = 2e-3
    drift_lr: float = 1e-5
    weight_decay: float = 1e-5
    loss_mode: str = "local_rollout"
    z_loss_weight: float = 1.0
    volume_loss_weight: float = 1.0
    train_drift: bool = False
    sinkhorn_epsilon: float = 0.08
    sinkhorn_iters: int = 60
    reaction_effect: ReactionEffectConfig = field(default_factory=ReactionEffectConfig)
    stim_k: float = 100.0
    parameter_bounds: dict = field(default_factory=dict)
    z_transform_params: dict = field(default_factory=dict)
    lr_scheduler_factor: float = 0.5
    lr_scheduler_patience: int = 30
    min_lr: float = 1e-7
    device: str = "auto"
    print_every: int = 20


def build_q_in_distribution(group_data, random_state=2026):
    """Sample precursor-high cells from the earliest Control group for q_in."""
    z_cols = list(next(iter(group_data.values()))["z_cols"])
    precursor_col = "P_precursor_zstd"
    precursor_idx = z_cols.index(precursor_col)
    top_quantile = 0.70
    n_max = 2048

    candidates = [
        (float(group["day"]), group_id)
        for group_id, group in group_data.items()
        if group["treatment_clean"] == "Control"
    ]
    if not candidates:
        raise ValueError("No Control group found for q_in.")
    anchor_day, anchor_group = min(candidates)
    Z, _ = mouse_balanced_sample(group_data, anchor_group, n_max=n_max, random_state=random_state)
    cutoff = np.quantile(Z[:, precursor_idx], top_quantile)
    q_in = Z[Z[:, precursor_idx] >= cutoff].astype(np.float32)
    if q_in.shape[0] == 0:
        raise ValueError("q_in selection is empty.")

    info = {
        "strategy": "earliest_control_top_precursor",
        "anchor_group": anchor_group,
        "anchor_day": anchor_day,
        "precursor_col": precursor_col,
        "top_quantile": top_quantile,
        "n_max": n_max,
        "random_state": int(random_state),
        "cutoff": float(cutoff),
        "n_q_in": int(q_in.shape[0]),
        "z_cols": z_cols,
    }
    return q_in, info


def build_z_transform_params(group_data, z_cols=None, q_low=0.01, q_high=0.99):
    """Build per-z-dimension robust min/max parameters for 0-1 transforms."""
    first_group = next(iter(group_data.values()))
    z_cols = z_cols or list(first_group["z_cols"])
    all_z = np.vstack([np.asarray(group["Z_all"], dtype=float) for group in group_data.values()])
    params = {}
    for dim, col in enumerate(z_cols):
        lo, hi = np.quantile(all_z[:, dim], [float(q_low), float(q_high)])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = float(np.nanmin(all_z[:, dim])), float(np.nanmax(all_z[:, dim]))
        if hi <= lo:
            lo, hi = lo - 1.0, hi + 1.0
        params[col] = {"dim": int(dim), "lo": float(lo), "hi": float(hi)}
    return params


def transform_z_numpy(Z, z_cols=None, scale="zstd", transform_params=None):
    """Transform z-score arrays for loss/activity/diagnostics."""
    Z = np.asarray(Z, dtype=float)
    if scale in (None, "zstd"):
        return Z.copy()
    if scale == "sigmoid01":
        return 1.0 / (1.0 + np.exp(-Z))
    if scale == "robust01":
        if transform_params is None:
            raise ValueError("transform_params is required for scale='robust01'.")
        z_cols = z_cols or list(transform_params.keys())
        out = np.zeros_like(Z, dtype=float)
        for dim, col in enumerate(z_cols):
            p = transform_params[col]
            out[:, dim] = (Z[:, dim] - p["lo"]) / (p["hi"] - p["lo"])
        return np.clip(out, 0.0, 1.0)
    raise ValueError(f"Unknown z transform scale: {scale}")

def transform_z_torch(z, z_cols=None, scale="zstd", transform_params=None):
    """Torch version of z transform; keeps model state in zstd but losses/activity may use 0-1."""
    if scale in (None, "zstd"):
        return z
    if scale == "sigmoid01":
        return torch.sigmoid(z)
    if scale == "robust01":
        if transform_params is None:
            raise ValueError("transform_params is required for scale='robust01'.")
        z_cols = z_cols or list(transform_params.keys())
        lows = []
        highs = []
        for col in z_cols:
            p = transform_params[col]
            lows.append(float(p["lo"]))
            highs.append(float(p["hi"]))
        lo = torch.as_tensor(lows, device=z.device, dtype=z.dtype)
        hi = torch.as_tensor(highs, device=z.device, dtype=z.dtype)
        return torch.clamp((z - lo) / (hi - lo + 1e-8), 0.0, 1.0)
    raise ValueError(f"Unknown z transform scale: {scale}")

class ReactionCoupledModel(nn.Module):
    """Transport weighted CD8 particles and add new mass sampled from q_in."""

    def __init__(self, drift_integrator, q_in, config=None):
        super().__init__()
        self.drift_integrator = drift_integrator
        self.config = config or ReactionCoupledConfig()
        self.register_buffer("q_in", torch.as_tensor(q_in, dtype=torch.float32))
        self.register_buffer("abundance_reference", torch.tensor(1.0))
        self.raw_tumor_r = nn.Parameter(torch.tensor(-2.0))
        self.raw_tumor_k = nn.Parameter(torch.tensor(np.log(2000.0), dtype=torch.float32))
        self.raw_killing = nn.Parameter(torch.tensor(-5.0))
        self.raw_source_rate = nn.Parameter(torch.tensor(-3.0))
        self.raw_activation_rate = nn.Parameter(torch.tensor(0.0))
        self.raw_death_rate = nn.Parameter(torch.tensor(-4.0))
        self.raw_stim_k = nn.Parameter(
            torch.tensor(np.log(float(self.config.stim_k)), dtype=torch.float32),
            requires_grad=False,
        )
        self.killing_weights = nn.Parameter(
            torch.tensor([0.2, 0.6, 0.8, 1.0, 0.4]),
            requires_grad=False,
        )
        self.reaction_effect = ReactionEffectTransform(self.config.reaction_effect)

        if not self.config.train_drift:
            for p in self.drift_integrator.parameters():
                p.requires_grad_(False)

    def positive_params(self):
        """Return effective killing and source coefficients for n = N / N0."""
        sp = torch.nn.functional.softplus
        return {
            "tumor_r": sp(self.raw_tumor_r),
            "tumor_k": torch.exp(torch.clamp(self.raw_tumor_k, min=np.log(10.0), max=np.log(1e6))),
            "killing": sp(self.raw_killing),
            "source_rate": sp(self.raw_source_rate),
            "lambda_act": sp(self.raw_activation_rate),
            "death_rate": sp(self.raw_death_rate),
            "stim_k": torch.exp(torch.clamp(self.raw_stim_k, min=np.log(1e-6), max=np.log(1e6))),
        }

    def cell_activity(self, z):
        z_for_activity = transform_z_torch(
            z,
            scale="robust01",
            transform_params=self.config.z_transform_params,
        )
        weights = self.killing_weights.to(device=z.device, dtype=z.dtype)
        return torch.nn.functional.softplus((z_for_activity * weights).sum(dim=1))

    def effector_activity(self, z, particle_weights):
        return (self.cell_activity(z) * particle_weights).sum()

    def sample_inflow_indices(self):
        return torch.randint(self.q_in.shape[0], (self.config.n_inflow_particles,), device=self.q_in.device)

    def inflow_step(self, transported, particle_weights, n, net_growth, source_rate, dt, indices):
        """Apply the Euler reaction step to old mass and append the incoming mass."""
        old_mass = particle_weights * (n * (1.0 + dt * net_growth))
        new_mass = (dt * source_rate / indices.numel()).expand(indices.numel())
        mass = torch.cat((old_mass, new_mass))
        n_next = mass.sum()
        states = torch.cat((transported, self.q_in[indices].to(transported)))
        return states, mass / n_next, n_next

    def treatment_effects_at_day(self, P, B, day, dtype, device):
        P_t, B_t = treatment_values_at_day(P, B, day, self.drift_integrator.pk_config)
        treatment = torch.tensor([[P_t, B_t]], dtype=dtype, device=device)
        effects = self.reaction_effect(treatment)
        G_PB = effects[0, 0]
        F_B = effects[0, 1]
        return G_PB, F_B

    def simulate(self, z0, P, B, day0, time_points, volume0=100.0, n0=None, *,
                 weights0=None, inflow_indices=None):
        """Advance normalized abundance, with n0 and the returned N on the reporting scale."""
        params = self.positive_params()
        z = z0
        weights = z.new_full((len(z),), 1.0 / len(z)) if weights0 is None else weights0
        q_in_index = torch.full((len(z),), -1, device=z.device, dtype=torch.long)
        draws = iter(inflow_indices) if inflow_indices is not None else None
        V = torch.as_tensor(float(volume0), device=z.device, dtype=z.dtype)
        n = z.new_tensor(1.0) if n0 is None else z.new_tensor(float(n0)) / self.abundance_reference
        current = float(day0)
        records = []

        for target_time in time_points:
            target_time = float(target_time)
            while current < target_time - 1e-9:
                h = min(float(self.config.dt), target_time - current)
                G_PB, F_B = self.treatment_effects_at_day(P, B, current, dtype=z.dtype, device=z.device)
                source_rate_t = params["source_rate"] * F_B
                z = self.drift_integrator(z, P=P, B=B, day0=current, dt=h)
                activity = self.effector_activity(z, weights)
                dV = params["tumor_r"] * V * (1.0 - V / params["tumor_k"]) - params["killing"] * activity * n * V
                tumor_stim = V / (params["stim_k"] + V + 1e-8)
                activation_rate_t = params["lambda_act"] * G_PB * tumor_stim
                V = torch.clamp(V + h * dV, min=1e-3)
                indices = self.sample_inflow_indices() if draws is None else next(draws)
                z, weights, n = self.inflow_step(
                    z, weights, n, activation_rate_t - params["death_rate"], source_rate_t, h, indices,
                )
                q_in_index = torch.cat((q_in_index, indices))
                current += h

            records.append({
                "time": target_time,
                "z": z,
                "particle_weights": weights,
                "q_in_index": q_in_index,
                "volume": V,
                "N": n * self.abundance_reference,
                "effector_activity": self.effector_activity(z, weights),
            })
        return records


def _sample_group_tensor(group_data, group_id, n_max, device, random_state):
    Z, _ = mouse_balanced_sample(group_data, group_id, n_max=n_max, random_state=random_state)
    return torch.as_tensor(Z, dtype=torch.float32, device=device)


def _safe_label(value):
    return str(value).replace("+", "plus").replace("-", "").replace(" ", "_")


def _resolve_treatments(group_data, treatment_clean):
    if treatment_clean == "all":
        preferred = ["Control", "Anti-PD1", "Anti-41BB", "Anti-PD1+Anti-41BB"]
        available_groups = {g["treatment_clean"] for g in group_data.values()}
        return [t for t in preferred if t in available_groups]
    if isinstance(treatment_clean, (list, tuple, set)):
        return list(treatment_clean)
    return [treatment_clean]


def _summarize_volume_by_day(volume_df):
    """Average replicate tumor volumes by day for volume-loss targets."""
    if volume_df.empty:
        return pd.Series(dtype=float)
    return volume_df.groupby("day")["tumor_volume"].mean().sort_index()


def _volume_at_day(vol_summary, day, default=100.0):
    """Interpolate summarized tumor volume to the simulation anchor day."""
    if vol_summary.empty:
        return float(default)
    x = vol_summary.index.to_numpy(dtype=float)
    y = vol_summary.to_numpy(dtype=float)
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    if float(day) <= x[0]:
        return float(y[0])
    if float(day) >= x[-1]:
        return float(y[-1])
    return float(np.interp(float(day), x, y))


def _inverse_softplus(value, eps=1e-8):
    value = max(float(value), float(eps))
    return np.log(np.expm1(value)) if value < 50 else value


_REACTION_PARAM_TO_RAW = {
    "tumor_r": "raw_tumor_r",
    "tumor_k": "raw_tumor_k",
    "killing": "raw_killing",
    "source_rate": "raw_source_rate",
    "lambda_act": "raw_activation_rate",
    "death_rate": "raw_death_rate",
    "stim_k": "raw_stim_k",
}


_REACTION_EFFECT_PARAM_TO_ATTR = {
    "f_b_coeff": ("f_b_coeff", "direct"),
    "g_b_coeff": ("g_b_coeff", "direct"),
    "g_pd1_coeff": ("g_pd1_coeff", "direct"),
    "pd1_k": ("pd1_k_raw", "softplus_positive"),
    "b41bb_k": ("b41bb_k_raw", "softplus_positive"),
    "K_P": ("pd1_k_raw", "softplus_positive"),
    "K_B": ("b41bb_k_raw", "softplus_positive"),
}


def set_reaction_positive_parameter(model, name, value, trainable=None):
    """Set one positive reaction parameter on its natural scale."""
    if name not in _REACTION_PARAM_TO_RAW:
        raise KeyError(f"Unknown reaction parameter '{name}'. Options: {sorted(_REACTION_PARAM_TO_RAW)}")
    raw_name = _REACTION_PARAM_TO_RAW[name]
    raw_param = getattr(model, raw_name)
    if name in ("tumor_k", "stim_k"):
        raw_value = np.log(max(float(value), 1e-8))
    else:
        raw_value = _inverse_softplus(value)
    with torch.no_grad():
        raw_param.copy_(torch.tensor(float(raw_value), device=raw_param.device, dtype=raw_param.dtype))
    if trainable is not None:
        raw_param.requires_grad_(bool(trainable))


def _reaction_effect_value(effect, name):
    if name not in _REACTION_EFFECT_PARAM_TO_ATTR:
        raise KeyError(f"Unknown reaction-effect parameter '{name}'.")
    attr, transform = _REACTION_EFFECT_PARAM_TO_ATTR[name]
    tensor = getattr(effect, attr)
    if transform == "softplus_positive":
        return float((torch.nn.functional.softplus(tensor) + float(effect.config.eps)).detach().cpu())
    return float(tensor.detach().cpu())


def set_reaction_effect_parameter(model, name, value, trainable=None):
    """Set one reaction-effect parameter on its natural scale."""
    if name not in _REACTION_EFFECT_PARAM_TO_ATTR:
        raise KeyError(f"Unknown reaction-effect parameter '{name}'.")
    effect = model.reaction_effect
    attr, transform = _REACTION_EFFECT_PARAM_TO_ATTR[name]
    tensor = getattr(effect, attr)
    if transform == "softplus_positive":
        raw_value = _inverse_softplus(max(float(value) - float(effect.config.eps), 1e-8))
    else:
        raw_value = float(value)
    with torch.no_grad():
        tensor.copy_(torch.tensor(raw_value, device=tensor.device, dtype=tensor.dtype))
    if trainable is not None:
        if not isinstance(tensor, torch.nn.Parameter) and bool(trainable):
            raise ValueError(
                f"{name} cannot be optimized because reaction_effect_config.learnable is False. "
                "Set learnable=True before constructing the model."
            )
        tensor.requires_grad_(bool(trainable))


def clamp_reaction_parameter_bounds(model, bounds):
    """Clamp positive reaction parameters to user-specified natural-scale bounds."""
    if not bounds:
        return
    for name, bound in bounds.items():
        if bound is None:
            continue
        if name in _REACTION_EFFECT_PARAM_TO_ATTR:
            lo, hi = bound
            value = _reaction_effect_value(model.reaction_effect, name)
            if lo is not None:
                value = max(value, float(lo))
            if hi is not None:
                value = min(value, float(hi))
            set_reaction_effect_parameter(model, name, value)
            continue
        lo, hi = bound
        value = float(model.positive_params()[name].detach().cpu())
        if lo is not None:
            value = max(value, float(lo))
        if hi is not None:
            value = min(value, float(hi))
        set_reaction_positive_parameter(model, name, value)


def apply_reaction_parameter_settings(model, parameter_settings, config=None):
    """
    Apply fixed/optimized parameter choices for reaction-coupled training.

    Each setting may contain:
    - mode: "fixed" or "optimize"
    - value: fixed value on the natural scale
    - init: initial value on the natural scale for optimized parameters
    - bounds: optional (low, high) tuple on the natural scale

    Parameters not listed keep their current initialization and remain trainable.
    """
    parameter_settings = dict(parameter_settings or {})
    bounds = {}
    for name, spec in parameter_settings.items():
        if name in _REACTION_EFFECT_PARAM_TO_ATTR:
            spec = spec or {}
            mode = str(spec.get("mode", "optimize")).lower()
            if mode not in ("fixed", "optimize"):
                raise ValueError(f"{name}.mode must be 'fixed' or 'optimize'.")
            if mode == "fixed":
                if "value" not in spec:
                    raise ValueError(f"{name} is fixed, so a 'value' is required.")
                set_reaction_effect_parameter(model, name, spec["value"], trainable=False)
            else:
                init_value = spec.get("init", None)
                if init_value is not None:
                    set_reaction_effect_parameter(model, name, init_value, trainable=True)
                else:
                    attr, _ = _REACTION_EFFECT_PARAM_TO_ATTR[name]
                    getattr(model.reaction_effect, attr).requires_grad_(True)
                bound = spec.get("bounds", None)
                if bound is not None:
                    bounds[name] = tuple(bound)
            continue
        if name not in _REACTION_PARAM_TO_RAW:
            raise KeyError(
                f"Unknown reaction parameter '{name}'. Options: "
                f"{sorted(_REACTION_PARAM_TO_RAW) + sorted(_REACTION_EFFECT_PARAM_TO_ATTR)}"
            )
        spec = spec or {}
        mode = str(spec.get("mode", "optimize")).lower()
        if mode not in ("fixed", "optimize"):
            raise ValueError(f"{name}.mode must be 'fixed' or 'optimize'.")
        if mode == "fixed":
            if "value" not in spec:
                raise ValueError(f"{name} is fixed, so a 'value' is required.")
            set_reaction_positive_parameter(model, name, spec["value"], trainable=False)
        else:
            init_value = spec.get("init", None)
            if init_value is not None:
                set_reaction_positive_parameter(model, name, init_value, trainable=True)
            else:
                getattr(model, _REACTION_PARAM_TO_RAW[name]).requires_grad_(True)
            bound = spec.get("bounds", None)
            if bound is not None:
                if len(bound) != 2:
                    raise ValueError(f"{name}.bounds must be a (low, high) tuple.")
                bounds[name] = tuple(bound)
    if config is not None:
        config.parameter_bounds = bounds
        if "stim_k" in parameter_settings:
            config.stim_k = float(model.positive_params()["stim_k"].detach().cpu())
    clamp_reaction_parameter_bounds(model, bounds)
    effect_trainable = []
    effect_fixed = []
    for canonical_name in ("f_b_coeff", "g_b_coeff", "g_pd1_coeff", "K_P", "K_B"):
        attr, _ = _REACTION_EFFECT_PARAM_TO_ATTR[canonical_name]
        tensor = getattr(model.reaction_effect, attr)
        if getattr(tensor, "requires_grad", False):
            effect_trainable.append(canonical_name)
        else:
            effect_fixed.append(canonical_name)
    return {
        "trainable": [
            name for name, raw_name in _REACTION_PARAM_TO_RAW.items()
            if getattr(model, raw_name).requires_grad
        ] + effect_trainable,
        "fixed": [
            name for name, raw_name in _REACTION_PARAM_TO_RAW.items()
            if not getattr(model, raw_name).requires_grad
        ] + effect_fixed + ["killing_weights"],
        "bounds": bounds,
        "values": {k: float(v.detach().cpu()) for k, v in model.positive_params().items()},
        "reaction_effect_values": {
            "f_b_coeff": _reaction_effect_value(model.reaction_effect, "f_b_coeff"),
            "g_b_coeff": _reaction_effect_value(model.reaction_effect, "g_b_coeff"),
            "g_pd1_coeff": _reaction_effect_value(model.reaction_effect, "g_pd1_coeff"),
            "K_P": _reaction_effect_value(model.reaction_effect, "K_P"),
            "K_B": _reaction_effect_value(model.reaction_effect, "K_B"),
        },
        "killing_weights": [float(x) for x in model.killing_weights.detach().cpu().tolist()],
    }


def _select_initial_z_anchor(group_data):
    """Select the Control day-1 cell distribution shared by the treatment arms."""
    candidates = [
        (float(group["day"]), group_id)
        for group_id, group in group_data.items()
        if str(group["treatment_clean"]) == "Control" and abs(float(group["day"]) - 1.0) < 1e-8
    ]
    if not candidates:
        raise ValueError("No Control day-1 group found for the initial z distribution.")
    day, group_id = min(candidates, key=lambda candidate: candidate[0])
    return {"group_id": group_id, "day": day, "treatment_clean": "Control"}


def _build_reaction_training_context(group_data, volume_df, treatment_clean, config, device, random_state):
    groups = []
    for group_id, group in group_data.items():
        if group["treatment_clean"] == treatment_clean:
            groups.append({
                "group_id": group_id,
                "day": float(group["day"]),
                "treatment_clean": group["treatment_clean"],
                "P": float(group["P"]),
                "B": float(group["B"]),
            })
    groups = sorted(groups, key=lambda x: x["day"])
    if len(groups) < 2:
        raise ValueError(f"Need at least two z-distribution days for {treatment_clean}.")

    vol = volume_df[volume_df["treatment_clean"] == treatment_clean].dropna(subset=["tumor_volume"]).copy()
    vol_summary = _summarize_volume_by_day(vol)

    treatment_anchor = groups[0]
    z_anchor = _select_initial_z_anchor(group_data)
    start_candidates = [float(z_anchor["day"]), 1.0]
    if not vol_summary.empty:
        start_candidates.append(float(vol_summary.index.min()))
    start_day = min(start_candidates)
    anchor = dict(treatment_anchor)
    anchor["day"] = float(start_day)
    anchor["z_anchor_day"] = float(z_anchor["day"])
    anchor["z_anchor_group_id"] = z_anchor["group_id"]
    anchor["z_anchor_treatment_clean"] = z_anchor["treatment_clean"]
    anchor["treatment_first_z_day"] = float(treatment_anchor["day"])
    anchor["treatment_first_z_group_id"] = treatment_anchor["group_id"]
    z0 = _sample_group_tensor(group_data, z_anchor["group_id"], config.n_particles, device, random_state)
    volume0 = _volume_at_day(vol_summary, start_day, default=100.0)
    volume_days = [float(d) for d in vol_summary.index] if not vol_summary.empty else []
    target_days = sorted(set([g["day"] for g in groups] + volume_days))
    target_days = [d for d in target_days if d >= start_day]
    if start_day not in target_days:
        target_days = sorted([start_day] + target_days)
    return {
        "treatment_clean": treatment_clean,
        "groups": groups,
        "vol_summary": vol_summary,
        "anchor": anchor,
        "z0": z0,
        "volume0": volume0,
        "start_day": float(start_day),
        "z_anchor_day": float(z_anchor["day"]),
        "z_anchor_group_id": z_anchor["group_id"],
        "z_anchor_treatment_clean": z_anchor["treatment_clean"],
        "treatment_first_z_day": float(treatment_anchor["day"]),
        "treatment_first_z_group_id": treatment_anchor["group_id"],
        "target_days": target_days,
        "has_volume": not vol_summary.empty,
    }


def _reaction_context_loss(model, group_data, context, config, loss_mode, device, rng):
    groups = context["groups"]
    vol_summary = context["vol_summary"]
    anchor = context["anchor"]
    sim = model.simulate(
        context["z0"],
        P=anchor["P"],
        B=anchor["B"],
        day0=anchor["day"],
        time_points=context["target_days"],
        volume0=context["volume0"],
        n0=None,
    )
    sim_by_day = {float(r["time"]): r for r in sim}

    z_losses = []
    if loss_mode in ("rollout", "local_rollout"):
        z_target_groups = groups[1:] if abs(float(groups[0]["day"]) - float(context["start_day"])) < 1e-8 else groups
        for g in z_target_groups:
            if g["day"] not in sim_by_day:
                continue
            z_target = _sample_group_tensor(
                group_data,
                g["group_id"],
                config.n_particles,
                device,
                int(rng.integers(0, 2**31 - 1)),
            )
            z_losses.append(sinkhorn_divergence(
                sim_by_day[g["day"]]["z"],
                z_target,
                epsilon=config.sinkhorn_epsilon,
                n_iters=config.sinkhorn_iters,
                x_weights=sim_by_day[g["day"]]["particle_weights"],
            ))
    else:
        for src_group, tgt_group in zip(groups[:-1], groups[1:]):
            z_src = _sample_group_tensor(
                group_data,
                src_group["group_id"],
                config.n_particles,
                device,
                int(rng.integers(0, 2**31 - 1)),
            )
            z_target = _sample_group_tensor(
                group_data,
                tgt_group["group_id"],
                config.n_particles,
                device,
                int(rng.integers(0, 2**31 - 1)),
            )
            src_day = float(src_group["day"])
            tgt_day = float(tgt_group["day"])
            segment = model.simulate(
                z_src,
                P=src_group["P"],
                B=src_group["B"],
                day0=src_day,
                time_points=[tgt_day],
                volume0=_volume_at_day(vol_summary, src_day, default=context["volume0"]),
                n0=None,
            )
            z_losses.append(sinkhorn_divergence(
                segment[-1]["z"],
                z_target,
                epsilon=config.sinkhorn_epsilon,
                n_iters=config.sinkhorn_iters,
                x_weights=segment[-1]["particle_weights"],
            ))
    z_loss = sum(z_losses) / len(z_losses) if z_losses else torch.tensor(0.0, device=device)

    volume_loss = None
    if context["has_volume"]:
        volume_preds = []
        volume_obs = []
        for day, observed in vol_summary.items():
            nearest_day = min(sim_by_day, key=lambda t: abs(t - float(day)))
            pred_v = sim_by_day[nearest_day]["volume"]
            obs_v = torch.as_tensor(float(observed), device=device, dtype=pred_v.dtype)
            volume_preds.append(pred_v)
            volume_obs.append(obs_v)
        pred = torch.stack(volume_preds)
        obs = torch.stack(volume_obs)
        ss_res = torch.sum((pred - obs).pow(2))
        ss_tot = torch.sum((obs - torch.mean(obs)).pow(2))
        volume_loss = ss_res / (ss_tot + 1e-8)
    return z_loss, volume_loss


def train_reaction_coupled_model(
    model,
    group_data,
    volume_df,
    treatment_clean="Control",
    config=None,
    random_state=2026,
):
    """
    Fit reaction parameters to z distributions and tumor volume.

    `treatment_clean="all"` trains one shared model with all treatment arms;
    it does not create separate models per treatment.
    """
    config = config or model.config
    device = next(model.parameters()).device
    rng = np.random.default_rng(random_state)
    loss_mode = config.loss_mode.lower()
    if loss_mode not in ("local", "rollout", "local_rollout"):
        raise ValueError("ReactionCoupledConfig.loss_mode must be 'local', 'rollout', or 'local_rollout'.")

    treatments = _resolve_treatments(group_data, treatment_clean)
    if not treatments:
        raise ValueError("No treatment arms available for reaction-coupled training.")
    contexts = [
        _build_reaction_training_context(
            group_data,
            volume_df,
            treatment,
            config,
            device,
            random_state + i,
        )
        for i, treatment in enumerate(treatments)
    ]

    reaction_params = []
    drift_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("drift_integrator."):
            drift_params.append(param)
        else:
            reaction_params.append(param)
    param_groups = []
    if reaction_params:
        param_groups.append({"params": reaction_params, "lr": config.lr, "name": "reaction"})
    if drift_params:
        param_groups.append({"params": drift_params, "lr": config.drift_lr, "name": "drift"})
    if not param_groups:
        raise ValueError("No trainable parameters found.")
    optimizer = torch.optim.AdamW(param_groups, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(config.lr_scheduler_factor),
        patience=int(config.lr_scheduler_patience),
        min_lr=float(config.min_lr),
    )
    history = []

    for epoch in range(1, config.n_epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        per_treatment = {}
        z_loss_values = []
        volume_loss_values = []
        for context in contexts:
            z_part, volume_part = _reaction_context_loss(
                model,
                group_data,
                context,
                config,
                loss_mode,
                device,
                rng,
            )
            z_loss_values.append(z_part)
            if volume_part is not None:
                volume_loss_values.append(volume_part)
            per_treatment[context["treatment_clean"]] = (z_part, volume_part)

        z_loss = sum(z_loss_values) / len(z_loss_values)
        volume_loss = (
            sum(volume_loss_values) / len(volume_loss_values)
            if volume_loss_values
            else torch.tensor(0.0, device=device)
        )

        loss = config.z_loss_weight * z_loss + config.volume_loss_weight * volume_loss
        loss.backward()
        optimizer.step()
        clamp_reaction_parameter_bounds(model, config.parameter_bounds)
        scheduler.step(float(loss.detach().cpu()))

        row = {
            "epoch": epoch,
            "loss": float(loss.detach().cpu()),
            "loss_mode": loss_mode,
            "volume_loss_mode": "mean_r2",
            "volume_summary_stat": "mean",
            "treatment_clean": "all" if treatment_clean == "all" else ",".join(treatments),
            "n_treatments": len(treatments),
            "use_q_in": True,
            "z_loss": float(z_loss.detach().cpu()),
            "volume_loss": float(volume_loss.detach().cpu()),
        }
        for group in optimizer.param_groups:
            row[f"lr_{group.get('name', 'group')}"] = float(group["lr"])
        row["volume_mean_r2"] = 1.0 - row["volume_loss"]
        for treatment, (z_part, volume_part) in per_treatment.items():
            label = _safe_label(treatment)
            row[f"z_loss_{label}"] = float(z_part.detach().cpu())
            row[f"volume_loss_{label}"] = float(volume_part.detach().cpu()) if volume_part is not None else np.nan
            row[f"volume_r2_{label}"] = (
                1.0 - float(volume_part.detach().cpu()) if volume_part is not None else np.nan
            )
        row.update({k: float(v.detach().cpu()) for k, v in model.positive_params().items()})
        history.append(row)
        if config.print_every and epoch % config.print_every == 0:
            print(
                f"[reaction-coupled] epoch={epoch:04d}/{config.n_epochs} "
                f"loss={row['loss']:.6g} z={row['z_loss']:.6g} volume={row['volume_loss']:.6g} "
                f"lr_reaction={row.get('lr_reaction', np.nan):.3g} lr_drift={row.get('lr_drift', np.nan):.3g}",
                flush=True,
            )

    return pd.DataFrame(history)


def energy_distance(x: np.ndarray, y: np.ndarray, max_points: int = 512, seed: int = 0,
                    x_weights=None) -> float:
    rng = np.random.default_rng(int(seed))
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x_weights is None:
        if x.shape[0] > max_points:
            x = x[rng.choice(x.shape[0], size=max_points, replace=False)]
        x_weights = np.full(len(x), 1.0 / len(x))
    if y.shape[0] > max_points:
        y = y[rng.choice(y.shape[0], size=max_points, replace=False)]
    xy = np.sqrt(np.maximum(((x[:, None, :] - y[None, :, :]) ** 2).sum(axis=2), 0.0))
    xx = np.sqrt(np.maximum(((x[:, None, :] - x[None, :, :]) ** 2).sum(axis=2), 0.0))
    yy = np.sqrt(np.maximum(((y[:, None, :] - y[None, :, :]) ** 2).sum(axis=2), 0.0))
    return float(max(0.0, 2.0 * (x_weights @ xy.mean(axis=1)) - (x_weights @ xx @ x_weights) - yy.mean()))


def distribution_metrics(predicted, observed, z_cols, transform_params, evaluation, seed, predicted_weights=None):
    """Score predicted and observed distributions with their empirical masses."""
    import ot

    scale = evaluation.get("z_scale", "robust01")
    with torch.no_grad():
        if scale is None and torch.is_tensor(predicted):
            x, y = predicted, observed
            pred = x.cpu().numpy().astype(np.float64)
            obs = y.cpu().numpy().astype(np.float64)
            cost = torch.cdist(x.to(torch.float64), y.to(torch.float64)).cpu().numpy()
        else:
            pred = transform_z_numpy(predicted, z_cols=z_cols, scale=scale, transform_params=transform_params)
            obs = transform_z_numpy(observed, z_cols=z_cols, scale=scale, transform_params=transform_params)
            device = choose_device(evaluation.get("device", "auto"))
            x = torch.as_tensor(pred, dtype=torch.float32, device=device)
            y = torch.as_tensor(obs, dtype=torch.float32, device=device)
            cost = ot.dist(pred, obs, metric="euclidean")
        weights = None if predicted_weights is None else torch.as_tensor(
            predicted_weights, dtype=x.dtype, device=x.device,
        )
        sinkhorn = sinkhorn_divergence(
            x, y,
            epsilon=float(evaluation.get("sinkhorn_epsilon", 0.08)),
            n_iters=int(evaluation.get("sinkhorn_iters", 40)),
            x_weights=weights,
        )
    mass = np.full(len(pred), 1.0 / len(pred)) if weights is None else weights.detach().cpu().numpy().astype(float)
    mass /= mass.sum()
    return {
        "sinkhorn_divergence": float(sinkhorn.detach().cpu()),
        "energy_distance": energy_distance(
            pred, obs, max_points=int(evaluation.get("energy_max_points", 512)), seed=seed,
            x_weights=None if weights is None else mass,
        ),
        "score_mean_mae": float(np.abs(mass @ pred - obs.mean(axis=0)).mean()),
        "wasserstein_1": float(ot.emd2(
            mass,
            np.full(len(obs), 1.0 / len(obs)),
            cost,
        )),
    }


def reaction_rollout_evaluation(
    model,
    group_data,
    volume_df,
    treatment_clean,
    config=None,
    n_time=121,
    n_particles=None,
    random_state=2026,
    time_points=None,
):
    """Evaluate one treatment arm with the trained reaction-coupled model."""
    config = config or model.config
    n_particles = int(config.n_particles if n_particles is None else n_particles)
    device = next(model.parameters()).device
    context = _build_reaction_training_context(
        group_data,
        volume_df,
        treatment_clean,
        replace(config, n_particles=n_particles),
        device,
        random_state,
    )
    anchor = context["anchor"]
    target_days = context["target_days"]
    if time_points is None:
        time_points = np.linspace(float(anchor["day"]), max(target_days), int(n_time))
    else:
        time_points = np.asarray(time_points, dtype=float)
    time_points = np.array(sorted(set(np.round(np.concatenate([time_points, np.asarray(target_days)]), 10))), dtype=float)
    time_points = time_points[time_points >= float(anchor["day"])]

    with torch.no_grad():
        sim = model.simulate(
            context["z0"],
            P=anchor["P"],
            B=anchor["B"],
            day0=anchor["day"],
            time_points=time_points,
            volume0=context["volume0"],
            n0=None,
        )

    z_cols = list(group_data[anchor["group_id"]]["z_cols"])
    summary_records = []
    volume_records = []
    pk_records = []
    dose_records = []
    ot_records = []
    sim_z_arrays = [rec["z"].detach().cpu().numpy() for rec in sim]
    quantile_levels = np.array([0.10, 0.25, 0.50, 0.75, 0.90])

    for rec, z_np in zip(sim, sim_z_arrays):
        t = float(rec["time"])
        weights = rec["particle_weights"].detach().cpu().numpy()
        for dim, z_col in enumerate(z_cols):
            values = z_np[:, dim]
            order = np.argsort(values)
            cumulative_mass = np.cumsum(weights[order], dtype=float)
            quantiles = values[order[np.searchsorted(cumulative_mass, quantile_levels * cumulative_mass[-1])]]
            summary_records.append({
                "kind": "predicted",
                "time": t,
                "day": t,
                "treatment_clean": treatment_clean,
                "P": anchor["P"],
                "B": anchor["B"],
                "z_dim": dim,
                "z_col": z_col,
                "n_cells": int(values.shape[0]),
                "mean": float(np.average(values, weights=weights)),
                **dict(zip(("q10", "q25", "q50", "q75", "q90"), quantiles)),
            })
        volume_records.append({
            "kind": "predicted",
            "time": t,
            "day": t,
            "treatment_clean": treatment_clean,
            "volume": float(rec["volume"].detach().cpu()),
            "N": float(rec["N"].detach().cpu()),
            "effector_activity": float(rec["effector_activity"].detach().cpu()),
        })
        P_t, B_t = treatment_values_at_day(anchor["P"], anchor["B"], t, model.drift_integrator.pk_config)
        G_PB, F_B = model.treatment_effects_at_day(anchor["P"], anchor["B"], t, dtype=rec["z"].dtype, device=rec["z"].device)
        pk_records.append({
            "time": t,
            "day": t,
            "treatment_clean": treatment_clean,
            "P": anchor["P"],
            "B": anchor["B"],
            "P_t": float(P_t),
            "B_t": float(B_t),
            "G_PB": float(G_PB.detach().cpu()),
            "F_B": float(F_B.detach().cpu()),
        })

    evaluation = {
        "z_scale": "zstd",
        "device": device,
        "sinkhorn_epsilon": config.sinkhorn_epsilon,
        "sinkhorn_iters": config.sinkhorn_iters,
    }
    for group in context["groups"]:
        seed = random_state + int(round(float(group["day"]) * 1000))
        Z_obs, _ = mouse_balanced_sample(
            group_data,
            group["group_id"],
            n_max=n_particles,
            random_state=seed,
        )
        nearest_rec, Z_pred = min(
            zip(sim, sim_z_arrays),
            key=lambda pair: abs(float(pair[0]["time"]) - float(group["day"])),
        )
        metrics = distribution_metrics(
            Z_pred, Z_obs, z_cols, None, evaluation, seed, predicted_weights=nearest_rec["particle_weights"],
        )
        ot_records.append({
            "day": float(group["day"]),
            "time": float(nearest_rec["time"]),
            "group_id": group["group_id"],
            "treatment_clean": treatment_clean,
            "P": group["P"],
            "B": group["B"],
            "ot_loss": metrics.pop("sinkhorn_divergence"),
            "z_loss_scale": "zstd",
            **metrics,
        })
        for dim, z_col in enumerate(z_cols):
            values = Z_obs[:, dim]
            summary_records.append({
                "kind": "observed",
                "time": float(group["day"]),
                "day": float(group["day"]),
                "group_id": group["group_id"],
                "treatment_clean": treatment_clean,
                "P": group["P"],
                "B": group["B"],
                "z_dim": dim,
                "z_col": z_col,
                "n_cells": int(values.shape[0]),
                "mean": float(np.mean(values)),
                "q10": float(np.quantile(values, 0.10)),
                "q25": float(np.quantile(values, 0.25)),
                "q50": float(np.quantile(values, 0.50)),
                "q75": float(np.quantile(values, 0.75)),
                "q90": float(np.quantile(values, 0.90)),
            })

    vol = volume_df[volume_df["treatment_clean"] == treatment_clean].dropna(subset=["tumor_volume"]).copy()
    if not vol.empty:
        for _, row in vol.iterrows():
            volume_records.append({
                "kind": "observed",
                "time": float(row["day"]),
                "day": float(row["day"]),
                "treatment_clean": treatment_clean,
                "volume": float(row["tumor_volume"]),
                "replicate_id": row.get("replicate_id", ""),
            })
        med = vol.groupby("day")["tumor_volume"].median().sort_index()
        for day, value in med.items():
            volume_records.append({
                "kind": "observed_median",
                "time": float(day),
                "day": float(day),
                "treatment_clean": treatment_clean,
                "volume": float(value),
            })

    pk_config = model.drift_integrator.pk_config
    if bool(anchor["P"]):
        for dose_day in pk_config.pd1_dose_days:
            dose_records.append({
                "treatment_clean": treatment_clean,
                "drug": "Anti-PD1",
                "day": float(dose_day),
                "dose_amount": float(pk_config.pd1_dose_amount),
            })
    if bool(anchor["B"]):
        for dose_day in pk_config.b41bb_dose_days:
            dose_records.append({
                "treatment_clean": treatment_clean,
                "drug": "Anti-41BB",
                "day": float(dose_day),
                "dose_amount": float(pk_config.b41bb_dose_amount),
            })

    return {
        "summary": pd.DataFrame(summary_records),
        "ot": pd.DataFrame(ot_records),
        "volume": pd.DataFrame(volume_records),
        "pk": pd.DataFrame(pk_records),
        "doses": pd.DataFrame(dose_records),
        "context": context,
    }


def evaluate_reaction_rollouts(
    model,
    group_data,
    volume_df,
    treatment_clean="all",
    config=None,
    n_time=121,
    n_particles=None,
    random_state=2026,
):
    """Evaluate one shared reaction model over one or more treatment arms."""
    config = config or model.config
    treatments = _resolve_treatments(group_data, treatment_clean)
    outputs = []
    for i, treatment in enumerate(treatments):
        outputs.append(reaction_rollout_evaluation(
            model,
            group_data,
            volume_df,
            treatment_clean=treatment,
            config=config,
            n_time=n_time,
            n_particles=n_particles,
            random_state=random_state + i,
        ))

    def concat(key):
        frames = [out[key] for out in outputs if key in out and not out[key].empty]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    return {
        "summary": concat("summary"),
        "ot": concat("ot"),
        "volume": concat("volume"),
        "pk": concat("pk"),
        "doses": concat("doses"),
        "per_treatment": outputs,
    }


def summarize_reaction_rollout_mean_predictions(summary_df):
    """
    Convert reaction rollout summaries into predicted-vs-target mean rows.

    `summary_df` is the `summary` table returned by evaluate_reaction_rollouts().
    Each output row compares the predicted mean at an observed day against the
    observed mean for one treatment arm and one z dimension.
    """
    if summary_df is None or summary_df.empty:
        return pd.DataFrame()
    required = {"kind", "treatment_clean", "time", "day", "z_col", "mean"}
    missing = required - set(summary_df.columns)
    if missing:
        raise ValueError(f"summary_df missing columns: {sorted(missing)}")

    pred = summary_df[summary_df["kind"] == "predicted"].copy()
    obs = summary_df[summary_df["kind"] == "observed"].copy()
    if pred.empty or obs.empty:
        return pd.DataFrame()

    pred = pred.sort_values(["treatment_clean", "z_col", "time"])
    obs = obs.sort_values(["treatment_clean", "z_col", "day"])
    records = []
    for _, row in obs.iterrows():
        treatment = row["treatment_clean"]
        z_col = row["z_col"]
        day = float(row["day"])
        part = pred[(pred["treatment_clean"] == treatment) & (pred["z_col"] == z_col)]
        if part.empty:
            continue
        nearest_idx = (part["time"].astype(float) - day).abs().idxmin()
        p_row = part.loc[nearest_idx]
        source_day = float(part["time"].astype(float).min())
        source_part = part[np.isclose(part["time"].astype(float), source_day)]
        source_mean = float(source_part["mean"].iloc[0]) if not source_part.empty else np.nan
        predicted_mean = float(p_row["mean"])
        target_mean = float(row["mean"])
        records.append({
            "level": "rollout",
            "transition_id": f"{treatment}__day{day:g}__{z_col}",
            "transition_type": "control_day1_rollout",
            "source_group": "Control day1 rollout anchor",
            "target_group": row.get("group_id", f"{treatment} day {day:g}"),
            "day0": source_day,
            "day1": day,
            "dt": day - source_day,
            "P": float(row.get("P", p_row.get("P", 0.0))),
            "B": float(row.get("B", p_row.get("B", 0.0))),
            "treatment_clean": treatment,
            "loss_weight": 1.0,
            "z_dim": int(row.get("z_dim", p_row.get("z_dim", 0))),
            "z_col": z_col,
            "source_mean": source_mean,
            "predicted_mean": predicted_mean,
            "target_mean": target_mean,
            "error": predicted_mean - target_mean,
            "abs_error": abs(predicted_mean - target_mean),
            "predicted_time": float(p_row["time"]),
        })

    return pd.DataFrame(records)


def save_reaction_checkpoint(path, model, history_df=None, q_in_info=None):
    """Save reaction-coupled model parameters."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "reaction_config": asdict(model.config),
        "pk_config": asdict(model.drift_integrator.pk_config),
    }
    if history_df is not None:
        payload["history"] = history_df.to_dict(orient="records")
    if q_in_info is not None:
        payload["q_in_info"] = q_in_info
    torch.save(payload, path)
    return path


def _read_reaction_checkpoint(path, map_location):
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    state = payload["state_dict"]
    if "raw_initial_n" in state:
        reference = torch.nn.functional.softplus(state.pop("raw_initial_n"))
        state["abundance_reference"] = reference
        bounds = payload["reaction_config"].get("parameter_bounds", {})
        bounds.pop("initial_n", None)
        for name, scale in (("killing", float(reference)), ("source_rate", 1.0 / float(reference))):
            raw = state[f"raw_{name}"]
            value = torch.nn.functional.softplus(raw) * scale
            state[f"raw_{name}"] = value + torch.log(-torch.expm1(-value))
            if name in bounds:
                bounds[name] = tuple(None if bound is None else bound * scale for bound in bounds[name])
    return payload


def load_reaction_checkpoint(path, model, map_location=None, strict=True):
    """Load reaction-coupled model parameters into an existing model instance."""
    path = Path(path)
    payload = _read_reaction_checkpoint(path, map_location)
    model.load_state_dict(payload["state_dict"], strict=strict)
    history_df = None
    if "history" in payload:
        history_df = pd.DataFrame(payload["history"])
    return {
        "model": model,
        "payload": payload,
        "history_df": history_df,
        "path": path,
    }


def load_reaction_checkpoint_configs(path, map_location="cpu"):
    """Restore model-structure configs saved with a reaction checkpoint."""
    payload = _read_reaction_checkpoint(path, map_location)
    reaction_data = dict(payload["reaction_config"])
    effect_data = reaction_data.pop("reaction_effect")
    reaction_config = ReactionCoupledConfig(
        reaction_effect=ReactionEffectConfig(**effect_data), **reaction_data,
    )
    pk_config = PKConfig(**payload["pk_config"])
    return reaction_config, pk_config, payload
