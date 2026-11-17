from pathlib import Path
import pickle

import numpy as np
import pandas as pd


DEFAULT_GROUP_PKL = (
    "data/mouse/group_distributions/B16_OT1_tumor_joint_zstd_group_distributions.pkl"
)
DEFAULT_GROUP_INFO_CSV = (
    "data/mouse/group_distributions/B16_OT1_tumor_joint_zstd_group_info.csv"
)
DEFAULT_TRANSITION_CSV = (
    "data/mouse/manifests/B16_OT1_tumor_joint_transition_manifest_temporal.csv"
)
DEFAULT_CELL_SCORE_CSV = "data/mouse/cell_level_z/B16_OT1_tumor_joint_fig1e_cell_z.csv"


def load_training_package(
    pkg_dir,
    group_pkl=DEFAULT_GROUP_PKL,
    group_info_csv=DEFAULT_GROUP_INFO_CSV,
    transition_csv=DEFAULT_TRANSITION_CSV,
):
    """
    Load group-level z distributions, group info, and temporal transitions.

    Parameters
    ----------
    pkg_dir : str or Path
        Repository root used to resolve the input paths.
    group_pkl, group_info_csv, transition_csv : str or Path
        Paths relative to `pkg_dir`, unless absolute paths are supplied.

    Returns
    -------
    group_data : dict
        Group-level cell-level z distributions.
    group_info : pandas.DataFrame
        Group metadata table.
    transitions : pandas.DataFrame
        Temporal-only transition manifest.
    paths : dict
        Resolved important file paths.
    """
    pkg_dir = Path(pkg_dir)

    group_pkl = pkg_dir / group_pkl
    group_info_csv = pkg_dir / group_info_csv
    transition_csv = pkg_dir / transition_csv

    with group_pkl.open("rb") as f:
        group_data = pickle.load(f)

    group_info = pd.read_csv(group_info_csv)
    transitions = pd.read_csv(transition_csv)

    validate_training_package(group_data, group_info, transitions)

    paths = {
        "pkg_dir": pkg_dir,
        "group_pkl": group_pkl,
        "group_info_csv": group_info_csv,
        "transition_csv": transition_csv,
    }

    return group_data, group_info, transitions, paths


def validate_training_package(group_data, group_info, transitions):
    """Run consistency checks for group data and transition manifest."""
    if len(group_data) == 0:
        raise ValueError("group_data is empty.")
    if group_info.shape[0] == 0:
        raise ValueError("group_info is empty.")
    if transitions.shape[0] == 0:
        raise ValueError("transition manifest is empty.")

    required_transition_cols = {
        "transition_id",
        "source_group",
        "target_group",
        "day0",
        "day1",
        "dt",
        "P",
        "B",
        "transition_type",
        "loss_weight",
    }
    missing_cols = required_transition_cols - set(transitions.columns)
    if missing_cols:
        raise ValueError(f"Transition manifest missing columns: {sorted(missing_cols)}")

    if not (transitions["dt"] > 0).all():
        raise ValueError("All transitions must have dt > 0.")

    if "use_for_training" in transitions.columns and not transitions["use_for_training"].astype(bool).all():
        raise ValueError("Temporal-only manifest should contain training transitions only.")

    available_groups = set(group_data.keys())
    missing_source = sorted(set(transitions["source_group"]) - available_groups)
    missing_target = sorted(set(transitions["target_group"]) - available_groups)
    if missing_source:
        raise ValueError(f"Missing source groups in group_data: {missing_source}")
    if missing_target:
        raise ValueError(f"Missing target groups in group_data: {missing_target}")


def get_z_columns(group_data):
    """Return z-space column names stored in the group distribution pkl."""
    return list(next(iter(group_data.values()))["z_cols"])


def load_raw_group_scores(group_data, score_csv):
    """Read raw program scores in the saved treatment, day and mouse groups."""
    score_cols = [f"{col.removesuffix('_zstd')}_wscore" for col in get_z_columns(group_data)]
    group_cols = ["treatment_clean", "day"]
    cells = pd.read_csv(score_csv, usecols=[*group_cols, "Mouse_Number", *score_cols])
    by_group = cells.groupby(group_cols, sort=False)
    raw_groups = {}
    for group_id, group in group_data.items():
        group_cells = by_group.get_group((group["treatment_clean"], group["day"]))
        mouse_cells = group_cells.groupby("Mouse_Number", sort=False)
        raw_z = group_cells[score_cols].to_numpy()
        by_mouse = {
            mouse_id: {**mouse, "Z": mouse_cells.get_group(mouse_id)[score_cols].to_numpy()}
            for mouse_id, mouse in group["by_mouse"].items()
        }
        if raw_z.shape != group["Z_all"].shape or any(
            mouse["Z"].shape != group["by_mouse"][mouse_id]["Z"].shape
            for mouse_id, mouse in by_mouse.items()
        ):
            raise ValueError(f"Raw cell scores do not match the saved group sizes for {group_id}.")
        raw_groups[group_id] = {**group, "Z_all": raw_z, "by_mouse": by_mouse}
    return raw_groups


def mouse_balanced_sample(
    group_data,
    group_id,
    n_max=512,
    random_state=None,
):
    """
    Mouse-balanced sampling with no replacement.

    Sampling rule:
        n_sample = min(n_cells_total, n_max)

    The allocation is approximately balanced across mice and respects each
    mouse's available cell count. Small groups are not upsampled.
    """
    rng = np.random.default_rng(random_state)

    g = group_data[group_id]
    by_mouse = g["by_mouse"]
    mouse_ids = list(rng.permutation(list(by_mouse.keys())))

    capacities = {
        mouse_id: int(by_mouse[mouse_id]["Z"].shape[0])
        for mouse_id in mouse_ids
    }

    n_cells_total = sum(capacities.values())
    n_total_eff = int(min(n_cells_total, n_max))
    quotas = {mouse_id: 0 for mouse_id in mouse_ids}

    for _ in range(n_total_eff):
        eligible = [m for m in mouse_ids if quotas[m] < capacities[m]]
        min_quota = min(quotas[m] for m in eligible)
        candidates = [m for m in eligible if quotas[m] == min_quota]
        chosen = rng.choice(candidates)
        quotas[chosen] += 1

    sampled_Z = []
    sampled_mouse = []

    for mouse_id in mouse_ids:
        Z_m = by_mouse[mouse_id]["Z"]
        n_draw = quotas[mouse_id]
        if n_draw <= 0:
            continue
        idx = rng.choice(Z_m.shape[0], size=n_draw, replace=False)
        sampled_Z.append(Z_m[idx])
        sampled_mouse.extend([mouse_id] * n_draw)

    if not sampled_Z:
        raise ValueError(f"No cells sampled from group {group_id}")

    Z_sample = np.vstack(sampled_Z).astype(np.float32)
    perm = rng.permutation(Z_sample.shape[0])
    Z_sample = Z_sample[perm]
    sampled_mouse = [sampled_mouse[i] for i in perm]

    meta = {
        "group_id": group_id,
        "day": g["day"],
        "treatment_clean": g["treatment_clean"],
        "P": g["P"],
        "B": g["B"],
        "recommended_use": g.get("recommended_use", None),
        "loss_weight": float(g.get("loss_weight", 1.0)),
        "n_cells_total": n_cells_total,
        "n_sample": int(Z_sample.shape[0]),
        "n_mice": len(mouse_ids),
        "mouse_ids": mouse_ids,
        "mouse_capacities": capacities,
        "mouse_quotas": quotas,
        "sampled_mouse": sampled_mouse,
        "z_cols": g["z_cols"],
    }

    return Z_sample, meta


def sample_transition(
    group_data,
    transition_row,
    n_max=512,
    random_state=None,
):
    """Sample source and target distributions for one temporal transition."""
    rng = np.random.default_rng(random_state)
    row = dict(transition_row)

    source_group = row["source_group"]
    target_group = row["target_group"]

    src_seed = int(rng.integers(0, 2**31 - 1))
    tgt_seed = int(rng.integers(0, 2**31 - 1))

    Z0, meta0 = mouse_balanced_sample(
        group_data,
        source_group,
        n_max=n_max,
        random_state=src_seed,
    )
    Z1, meta1 = mouse_balanced_sample(
        group_data,
        target_group,
        n_max=n_max,
        random_state=tgt_seed,
    )

    return {
        "transition_id": row["transition_id"],
        "source_group": source_group,
        "target_group": target_group,
        "Z0": Z0,
        "Z1": Z1,
        "day0": float(row["day0"]),
        "day1": float(row["day1"]),
        "dt": float(row["dt"]),
        "P": float(row["P"]),
        "B": float(row["B"]),
        "transition_type": row["transition_type"],
        "loss_weight": float(row["loss_weight"]),
        "source_meta": meta0,
        "target_meta": meta1,
    }


def summarize_tumor_volume(volume_df):
    """Summarize tumor volume by source, sheet, treatment, and day."""
    group_cols = ["source_file", "sheet", "treatment_clean", "P", "B", "day"]
    if "dataset" in volume_df.columns:
        group_cols = ["dataset"] + group_cols
    return (
        volume_df
        .dropna(subset=["tumor_volume"])
        .groupby(group_cols, dropna=False)["tumor_volume"]
        .agg(n="count", mean="mean", median="median", std="std")
        .reset_index()
    )
