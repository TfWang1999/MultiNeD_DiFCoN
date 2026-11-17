from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from common.cd8_zdist_data import get_z_columns


@dataclass
class HoldoutSplit:
    split_id: str
    kind: str
    heldout_groups: list[str]
    heldout_treatments: list[str]
    heldout_days: list[float]
    train_group_data: dict
    train_transitions: pd.DataFrame
    train_volume: pd.DataFrame


def standardize_holdout_split(split: HoldoutSplit, raw_group_data: dict) -> tuple[HoldoutSplit, dict, dict]:
    """Fit on the fold's raw training cells and transform both sides of the split."""
    train_scores = np.concatenate([group["Z_all"] for group in split.train_group_data.values()])
    scaler = StandardScaler().fit(train_scores)
    groups = {
        group_id: {
            **group,
            "Z_all": scaler.transform(group["Z_all"]).astype(np.float32),
            "by_mouse": {
                mouse_id: {**mouse, "Z": scaler.transform(mouse["Z"]).astype(np.float32)}
                for mouse_id, mouse in group["by_mouse"].items()
            },
        }
        for group_id, group in raw_group_data.items()
    }
    train_groups = {group_id: groups[group_id] for group_id in split.train_group_data}
    params = {
        "z_cols": get_z_columns(train_groups),
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "n_training_cells": len(train_scores),
    }
    return replace(split, train_group_data=train_groups), groups, params


def exclude_groups(group_data: dict, heldout_groups: Iterable[str]) -> dict:
    heldout = set(heldout_groups)
    return {group_id: group for group_id, group in group_data.items() if group_id not in heldout}


def filter_transitions_by_split(
    transitions: pd.DataFrame,
    heldout_groups: Iterable[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove every transition that exposes a held-out group as source or target."""
    heldout = set(heldout_groups)
    exposed = transitions["source_group"].isin(heldout) | transitions["target_group"].isin(heldout)
    return (
        transitions.loc[~exposed].copy().reset_index(drop=True),
        transitions.loc[exposed].copy().reset_index(drop=True),
    )


def _split_token(value: str) -> str:
    return (
        str(value)
        .replace("+", "plus")
        .replace("-", "")
        .replace(" ", "")
        .replace("/", "_")
        .replace("\\", "_")
    )


def build_volume_timepoint_holdout_split(
    group_data: dict,
    transitions: pd.DataFrame,
    volume_df: pd.DataFrame,
    treatment: str,
    day: float,
) -> HoldoutSplit:
    """Hold out one tumor-volume treatment-day and any exactly matching z group.

    Several tumor-volume days do not have a same-day single-cell distribution.
    Those folds therefore remove only the volume rows. When an exact z group is
    available, it and every exposing transition are also removed so that the
    treatment-day cannot leak through the distribution loss.
    """
    treatment = str(treatment)
    day = float(day)
    known_treatments = {
        str(group["treatment_clean"])
        for group in group_data.values()
    } | set(volume_df["treatment_clean"].dropna().astype(str))
    if treatment not in known_treatments:
        raise KeyError(f"Unknown volume-holdout treatment: {treatment}")

    volume_mask = (
        volume_df["treatment_clean"].astype(str).eq(treatment)
        & np.isclose(volume_df["day"].astype(float), day)
    )
    if not volume_mask.any():
        raise KeyError(f"No tumor-volume rows for treatment={treatment!r}, day={day:g}")

    heldout_groups = sorted(
        group_id
        for group_id, group in group_data.items()
        if str(group["treatment_clean"]) == treatment
        and np.isclose(float(group["day"]), day)
    )
    train_transitions, _ = filter_transitions_by_split(transitions, heldout_groups)
    split = HoldoutSplit(
        split_id=f"volume_timepoint__{_split_token(treatment)}__day{day:g}",
        kind="volume_timepoint",
        heldout_groups=heldout_groups,
        heldout_treatments=[treatment],
        heldout_days=[day],
        train_group_data=exclude_groups(group_data, heldout_groups),
        train_transitions=train_transitions,
        train_volume=volume_df.loc[~volume_mask].copy().reset_index(drop=True),
    )
    assert_no_group_leakage(split)
    return split


def assert_no_group_leakage(split: HoldoutSplit) -> None:
    heldout = set(split.heldout_groups)
    leaked_groups = heldout & set(split.train_group_data)
    if leaked_groups:
        raise AssertionError(f"Held-out groups remain in training data: {sorted(leaked_groups)}")
    exposed = split.train_transitions[
        split.train_transitions["source_group"].isin(heldout)
        | split.train_transitions["target_group"].isin(heldout)
    ]
    if not exposed.empty:
        raise AssertionError("Held-out groups remain exposed in training transitions")
    leaked = (
        split.train_volume["treatment_clean"].isin(split.heldout_treatments)
        & split.train_volume["day"].astype(float).isin(split.heldout_days)
    )
    if leaked.any():
        raise AssertionError("Held-out treatment-day remains in training volume data")


def holdout_summary(split: HoldoutSplit) -> dict:
    return {
        "split_id": split.split_id,
        "kind": split.kind,
        "heldout_groups": split.heldout_groups,
        "heldout_treatments": split.heldout_treatments,
        "heldout_days": split.heldout_days,
        "train_group_count": len(split.train_group_data),
        "train_transition_count": int(split.train_transitions.shape[0]),
        "train_volume_row_count": int(split.train_volume.shape[0]),
    }
