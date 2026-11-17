from __future__ import annotations

from dataclasses import asdict, dataclass

import torch

from .config import TOTAL_BUDGET


@dataclass
class DirectLossConfig:
    """Policy trajectory objective configured by the formal experiment entry."""

    final_tumor_weight: float = 1.0
    tumor_auc_weight: float = 1.0
    functional_activity_weight: float = 0.50
    smoothness_weight: float = 0.02
    total_dose_weight: float = 1.0
    efficiency_weight: float = 0.1
    late_window_days: float = 3.0
    functional_retention: float = 0.90
    min_functional_activity: float = 0.0
    efficiency_endpoint_benefit_weight: float = 1.0
    efficiency_auc_benefit_weight: float = 1.0
    efficiency_pseudo_dose: float = 0.10
    efficiency_reference: float = 1.0
    eps: float = 1e-8

    def __post_init__(self) -> None:
        if float(self.efficiency_pseudo_dose) <= 0:
            raise ValueError("efficiency_pseudo_dose must be > 0.")
        if float(self.efficiency_reference) <= 0:
            raise ValueError("efficiency_reference must be > 0.")

    def to_dict(self) -> dict:
        return asdict(self)


def normalized_auc(values: torch.Tensor, times: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    values = values.reshape(-1)
    times = times.reshape(-1).to(device=values.device, dtype=values.dtype)
    if values.numel() <= 1:
        return values.mean()
    dt = times[1:] - times[:-1]
    duration = torch.clamp(times[-1] - times[0], min=float(eps))
    return (((values[:-1] + values[1:]) * 0.5) * dt).sum() / duration


def late_window_average(
    values: torch.Tensor,
    times: torch.Tensor,
    window_days: float,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute the time average over the specified window ending at the final time."""

    values = values.reshape(-1)
    times = times.reshape(-1).to(device=values.device, dtype=values.dtype)
    if values.numel() <= 1 or float(window_days) <= 0:
        return values[-1]
    window_start = times[-1] - float(window_days)
    mask = times >= window_start - float(eps)
    if int(mask.sum().item()) < 2:
        mask = torch.zeros_like(mask, dtype=torch.bool)
        mask[-min(2, values.numel()) :] = True
    return normalized_auc(values[mask], times[mask], eps=eps)


def functional_activity_loss(
    trajectory: dict[str, torch.Tensor],
    config: DirectLossConfig,
) -> dict[str, torch.Tensor]:
    """Penalize low mean per-cell CD8 functional activity in the late treatment window."""

    activity = torch.clamp(trajectory["activity"].reshape(-1), min=0.0)
    times = trajectory["time"].reshape(-1)
    late_activity = late_window_average(
        activity,
        times,
        window_days=config.late_window_days,
        eps=config.eps,
    )
    baseline_activity = activity[0].detach()
    target_activity = torch.maximum(
        activity.new_tensor(float(config.min_functional_activity)),
        baseline_activity * float(config.functional_retention),
    )
    relative_deficit = torch.relu(
        (target_activity - late_activity) / (target_activity + float(config.eps))
    )
    return {
        "functional_activity_loss": relative_deficit.pow(2),
        "late_activity": late_activity,
        "target_activity": target_activity,
        "baseline_activity": baseline_activity,
    }


def drug_efficiency_loss(
    final_tumor: torch.Tensor,
    tumor_auc: torch.Tensor,
    actions: torch.Tensor,
    config: DirectLossConfig,
) -> dict[str, torch.Tensor]:
    """Use a bounded loss to maximize combined tumor benefit per unit of dose administered."""

    total_dose = actions.sum()
    tumor_benefit = (
        float(config.efficiency_endpoint_benefit_weight) * (1.0 - final_tumor)
        + float(config.efficiency_auc_benefit_weight) * (1.0 - tumor_auc)
    )
    efficiency = tumor_benefit / (total_dose + float(config.efficiency_pseudo_dose))
    efficiency_loss = torch.sigmoid(-efficiency / float(config.efficiency_reference))
    return {
        "efficiency_loss": efficiency_loss,
        "drug_efficiency": efficiency,
        "tumor_benefit": tumor_benefit,
        "total_dose": total_dose,
    }


def compute_direct_loss(
    trajectory: dict[str, torch.Tensor],
    config: DirectLossConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Compute the end-to-end loss for one patient.

    The rollout enforces hard dose limits, and the normalized total-dose term encourages lower dosing.
    The efficiency weight defaults to 0.1. The No efficiency ablation sets it to 0.
    """

    C = trajectory["volume"].reshape(-1)
    times = trajectory["time"].reshape(-1)
    C_rel = C / torch.clamp(C[0].detach(), min=float(config.eps))
    final_tumor = C_rel[-1]
    tumor_auc = normalized_auc(C_rel, times, config.eps)
    actions = trajectory["actions"].reshape(-1, 2)
    smoothness = (
        (actions[1:] - actions[:-1]).pow(2).mean()
        if actions.shape[0] > 1
        else C.new_tensor(0.0)
    )
    functional = functional_activity_loss(trajectory, config)
    efficiency = drug_efficiency_loss(final_tumor, tumor_auc, actions, config)
    normalized_total_dose = efficiency["total_dose"] / TOTAL_BUDGET
    total = (
        config.final_tumor_weight * final_tumor
        + config.tumor_auc_weight * tumor_auc
        + config.functional_activity_weight * functional["functional_activity_loss"]
        + config.smoothness_weight * smoothness
        + config.total_dose_weight * normalized_total_dose
        + config.efficiency_weight * efficiency["efficiency_loss"]
    )
    components = {
        "total_loss": total,
        "final_tumor_loss": final_tumor,
        "tumor_auc_loss": tumor_auc,
        **functional,
        "smoothness_loss": smoothness,
        "total_dose_loss": normalized_total_dose,
        "normalized_total_dose": normalized_total_dose,
        **efficiency,
    }
    return total, components


def components_to_float(components: dict[str, torch.Tensor]) -> dict[str, float]:
    return {
        key: float(value.detach().cpu()) if torch.is_tensor(value) else float(value)
        for key, value in components.items()
    }


__all__ = [
    "DirectLossConfig",
    "components_to_float",
    "compute_direct_loss",
    "drug_efficiency_loss",
    "functional_activity_loss",
    "late_window_average",
    "normalized_auc",
]
