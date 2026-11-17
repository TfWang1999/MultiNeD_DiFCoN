from __future__ import annotations

from typing import Any

import numpy as np

from experiments.volume_loto.splits import (
    HoldoutSplit,
    _split_token as _token,
    assert_no_group_leakage,
    build_volume_timepoint_holdout_split,
)




def build_shared_conditional_loto_splits(
    group_data: dict,
    transitions,
    volume_df,
    *,
    treatments: list[str],
    days: list[float],
    source_group: str = "day1__Control",
) -> tuple[list[HoldoutSplit], list[dict[str, Any]]]:
    """Build Reaction-style treatment-day holdouts with one shared conditional fit.

    Only treatment-day targets with an exactly matching z distribution are
    retained.  Each fold removes that group and every transition that exposes
    it, while all other treatment arms remain available for one joint model.
    """

    source_group = str(source_group)
    if source_group not in group_data:
        raise KeyError(f"Unknown shared-conditional source group: {source_group}")

    splits: list[HoldoutSplit] = []
    exclusions: list[dict[str, Any]] = []
    for treatment in map(str, treatments):
        for day_value in map(float, days):
            matching_groups = sorted(
                group_id
                for group_id, group in group_data.items()
                if str(group["treatment_clean"]) == treatment
                and np.isclose(float(group["day"]), day_value)
            )
            if not matching_groups:
                exclusions.append(
                    {
                        "treatment": treatment,
                        "target_day": day_value,
                        "reason": "no_matching_z_distribution",
                    }
                )
                continue
            if len(matching_groups) != 1:
                raise ValueError(
                    f"Expected one matching z group for treatment={treatment!r}, "
                    f"day={day_value:g}; found {matching_groups}"
                )
            target_group = matching_groups[0]
            if target_group == source_group:
                raise ValueError("The shared-conditional source anchor cannot be held out")

            split = build_volume_timepoint_holdout_split(
                group_data,
                transitions,
                volume_df,
                treatment=treatment,
                day=day_value,
            )
            if split.heldout_groups != [target_group]:
                raise AssertionError(
                    f"Unexpected held-out groups for {treatment} day {day_value:g}: "
                    f"{split.heldout_groups}"
                )
            split.split_id = (
                f"shared_conditional_loto__{_token(treatment)}__{_token(target_group)}"
            )
            split.kind = "shared_conditional_loto"
            assert_no_group_leakage(split)
            if source_group not in split.train_group_data:
                raise AssertionError(f"Source anchor {source_group} is absent from training data")
            splits.append(split)

    return splits, exclusions


def shared_conditional_loto_summary(
    split: HoldoutSplit,
    full_group_data: dict,
    *,
    source_group: str,
) -> dict[str, Any]:
    target_group = split.heldout_groups[0]
    target = full_group_data[target_group]
    source = full_group_data[str(source_group)]
    exposed = split.train_transitions[
        split.train_transitions["source_group"].astype(str).eq(target_group)
        | split.train_transitions["target_group"].astype(str).eq(target_group)
    ]
    return {
        "split_id": split.split_id,
        "kind": split.kind,
        "target_group": target_group,
        "target_treatment": str(target["treatment_clean"]),
        "target_day": float(target["day"]),
        "source_group": str(source_group),
        "source_treatment": str(source["treatment_clean"]),
        "source_day": float(source["day"]),
        "prediction_scope": "heldout_treatment_day_from_control_anchor",
        "global_joint_fit": True,
        "treatment_conditioned": True,
        "conditioning_variables": ["P", "B", "time"],
        "heldout_group_in_training": target_group in split.train_group_data,
        "heldout_group_exposed_in_training_transitions": not exposed.empty,
        "train_group_count": len(split.train_group_data),
        "train_transition_count": int(split.train_transitions.shape[0]),
        "evaluation_target_count": 1,
    }


__all__ = [
    "build_shared_conditional_loto_splits",
    "shared_conditional_loto_summary",
]
