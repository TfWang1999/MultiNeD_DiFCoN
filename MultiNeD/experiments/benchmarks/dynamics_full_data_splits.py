from __future__ import annotations

from typing import Any

from experiments.volume_loto.splits import HoldoutSplit


FULL_DATA_ROLLOUT_SCOPE = "in_sample_rollout_from_control_day1"
REQUIRED_TRANSITION_COLUMNS = {
    "transition_id",
    "source_group",
    "target_group",
    "day0",
    "day1",
    "P",
    "B",
}


def build_distribution_full_data_joint_split(
    group_data: dict,
    transitions,
    volume_df,
) -> HoldoutSplit:
    """Build the single global joint full-data fit unit.

    Unlike LOTO, no group or transition is withheld.  The returned
    ``HoldoutSplit`` is only a compatibility container for the existing
    benchmark runner. Rollout scoring uses a separate list of target groups.
    """

    missing_columns = sorted(REQUIRED_TRANSITION_COLUMNS - set(transitions.columns))
    if missing_columns:
        raise ValueError(f"Full-data transition manifest is missing columns: {missing_columns}")
    if transitions.empty:
        raise ValueError("Full-data joint benchmark requires at least one transition")
    if transitions["transition_id"].astype(str).duplicated().any():
        raise ValueError("Full-data transition IDs must be unique")

    referenced_groups = set(transitions["source_group"].astype(str)) | set(
        transitions["target_group"].astype(str)
    )
    missing_groups = sorted(referenced_groups - set(map(str, group_data)))
    if missing_groups:
        raise ValueError(f"Transition manifest references unknown groups: {missing_groups}")

    return HoldoutSplit(
        split_id="distribution_full_data_joint",
        kind="full_data_joint",
        heldout_groups=[],
        heldout_treatments=[],
        heldout_days=[],
        train_group_data=dict(group_data),
        train_transitions=transitions.copy().reset_index(drop=True),
        train_volume=volume_df.copy().reset_index(drop=True),
    )


def full_data_rollout_targets(split: HoldoutSplit, source_group: str) -> list[str]:
    source_day = float(split.train_group_data[source_group]["day"])
    return sorted(
        (group_id for group_id, group in split.train_group_data.items()
         if float(group["day"]) > source_day),
        key=lambda group_id: (float(split.train_group_data[group_id]["day"]), group_id),
    )


def distribution_full_data_joint_summary(split: HoldoutSplit) -> dict[str, Any]:
    source_group = "day1__Control"
    treatments = sorted(
        {str(group["treatment_clean"]) for group in split.train_group_data.values()}
    )
    days = sorted({float(group["day"]) for group in split.train_group_data.values()})
    return {
        "split_id": split.split_id,
        "kind": split.kind,
        "prediction_scope": FULL_DATA_ROLLOUT_SCOPE,
        "source_group": source_group,
        "source_day": float(split.train_group_data[source_group]["day"]),
        "global_joint_fit": True,
        "treatment_conditioned": True,
        "conditioning_variables": ["P", "B", "time"],
        "training_treatments": treatments,
        "training_days": days,
        "train_group_count": len(split.train_group_data),
        "train_transition_count": int(split.train_transitions.shape[0]),
        "evaluation_target_count": len(full_data_rollout_targets(split, source_group)),
    }


__all__ = [
    "build_distribution_full_data_joint_split",
    "distribution_full_data_joint_summary",
    "full_data_rollout_targets",
]
