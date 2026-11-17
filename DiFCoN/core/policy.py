from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
import torch.nn as nn

from .config import DAILY_TOTAL_DOSE_CAP
from .policy_common import ConvHistoryBlock


DIRECT_DOSE_POLICY_TYPE = "direct_dose"


def _logit(probability: float, eps: float = 1e-6) -> float:
    value = min(max(float(probability), float(eps)), 1.0 - float(eps))
    return math.log(value / (1.0 - value))


@dataclass
class DirectDosePolicyConfig:
    """Configuration for the closed-loop 1D CNN that outputs doses directly."""

    feature_names: tuple[str, ...]
    history_length: int = 5
    hidden_dim: int = 64
    n_conv_blocks: int = 2
    dropout: float = 0.05
    initial_dose: float = 0.01
    output_weight_std: float = 1e-3

    def __post_init__(self) -> None:
        self.feature_names = tuple(self.feature_names)
        if not self.feature_names:
            raise ValueError("feature_names must not be empty.")
        if int(self.history_length) < 1:
            raise ValueError("history_length must be >= 1.")
        if int(self.hidden_dim) < 1 or int(self.n_conv_blocks) < 1:
            raise ValueError("hidden_dim and n_conv_blocks must be >= 1.")
        if not 0.0 < float(self.initial_dose) < DAILY_TOTAL_DOSE_CAP:
            raise ValueError("initial_dose must be between zero and the daily dose cap.")

    @property
    def feature_dim(self) -> int:
        return len(self.feature_names)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["feature_names"] = list(self.feature_names)
        return data


class DirectDoseTreatmentCNN(nn.Module):
    """
    Predict the daily total dose and its anti-PD-1 fraction from the trajectory history.
    """

    policy_type = DIRECT_DOSE_POLICY_TYPE

    def __init__(self, config: DirectDosePolicyConfig):
        super().__init__()
        self.config = config
        self.input_proj = nn.Conv1d(config.feature_dim, config.hidden_dim, kernel_size=1)
        self.blocks = nn.Sequential(
            *[
                ConvHistoryBlock(config.hidden_dim, config.dropout)
                for _ in range(int(config.n_conv_blocks))
            ]
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.current_state = nn.Sequential(
            nn.Linear(config.feature_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Dropout(config.dropout),
        )
        self.head_hidden = nn.Sequential(
            nn.Linear(2 * config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Dropout(config.dropout),
        )
        self.head_out = nn.Linear(config.hidden_dim, 2)
        nn.init.normal_(self.head_out.weight, mean=0.0, std=float(config.output_weight_std))
        self._initialize_action_bias()

    def _initialize_action_bias(self) -> None:
        target_total = min(2.0 * float(self.config.initial_dose), 0.95 * DAILY_TOTAL_DOSE_CAP)
        total_fraction = target_total / DAILY_TOTAL_DOSE_CAP
        bias = torch.tensor([_logit(total_fraction), 0.0], dtype=self.head_out.bias.dtype)
        with torch.no_grad():
            self.head_out.bias.copy_(bias)

    def action_logits(self, history: torch.Tensor) -> torch.Tensor:
        if history.ndim == 2:
            history = history.unsqueeze(0)
        if history.ndim != 3:
            raise ValueError("history must be [batch, length, feature] or [batch, feature, length].")
        if history.shape[-1] == self.config.feature_dim:
            history_lbf = history
            x = history.transpose(1, 2)
        elif history.shape[1] == self.config.feature_dim:
            x = history
            history_lbf = history.transpose(1, 2)
        else:
            raise ValueError(f"Expected feature_dim={self.config.feature_dim}, got {tuple(history.shape)}.")

        encoded = self.blocks(self.input_proj(x))
        pooled = self.pool(encoded).squeeze(-1)
        current = self.current_state(history_lbf[:, -1, :])
        hidden = self.head_hidden(torch.cat([pooled, current], dim=1))
        return self.head_out(hidden)

    def raw_action(
        self,
        history: torch.Tensor,
        remaining_budget: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.action_logits(history).reshape(-1, 2)[0]
        remaining = torch.clamp(remaining_budget.reshape(()), min=0.0)
        available_today = torch.minimum(logits.new_tensor(DAILY_TOTAL_DOSE_CAP), remaining)
        total_dose = available_today * torch.sigmoid(logits[0])

        pd1_low = torch.clamp(total_dose - DAILY_TOTAL_DOSE_CAP, min=0.0)
        pd1_high = torch.minimum(total_dose, total_dose.new_tensor(DAILY_TOTAL_DOSE_CAP))
        pd1_dose = pd1_low + (pd1_high - pd1_low) * torch.sigmoid(logits[1])
        b41bb_dose = total_dose - pd1_dose
        return torch.stack([pd1_dose, b41bb_dose])

    def act(self, history: torch.Tensor, step_index: int, remaining_budget: torch.Tensor) -> torch.Tensor:
        del step_index
        raw = self.raw_action(history, remaining_budget)
        remaining = torch.clamp(remaining_budget.reshape(()), min=0.0)
        scale = torch.clamp(remaining / (raw.sum() + 1e-8), max=1.0)
        return raw * scale


__all__ = [
    "DIRECT_DOSE_POLICY_TYPE",
    "DirectDosePolicyConfig",
    "DirectDoseTreatmentCNN",
    "load_policy_from_checkpoint",
]


def load_policy_from_checkpoint(checkpoint: dict, device):
    config = DirectDosePolicyConfig(**checkpoint["policy_config"])
    policy = DirectDoseTreatmentCNN(config).to(device)
    policy.load_state_dict(checkpoint["policy_state_dict"], strict=True)
    policy.eval()
    return policy
