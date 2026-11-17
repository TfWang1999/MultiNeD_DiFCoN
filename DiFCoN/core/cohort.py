from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from scipy.stats import qmc


def _map_unit_interval(values: np.ndarray, low: float, high: float, mapping: str) -> np.ndarray:
    mapping = str(mapping).lower()
    if mapping == "linear":
        return float(low) + values * (float(high) - float(low))
    if mapping == "log":
        if low <= 0 or high <= 0:
            raise ValueError("Log-mapped ranges require low > 0 and high > 0.")
        return np.exp(np.log(float(low)) + values * (np.log(float(high)) - np.log(float(low))))
    raise ValueError(f"Unknown range mapping: {mapping}")


def validate_sampling_ranges(sampling_ranges: Mapping[str, Mapping], capacity_margin: float = 1.2) -> None:
    for name, spec in sampling_ranges.items():
        low, high = float(spec["low"]), float(spec["high"])
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            raise ValueError(f"Invalid sampling range for {name}: ({low}, {high}).")
    if "C0" in sampling_ranges and "tumor_k_abs" in sampling_ranges:
        max_c0 = float(sampling_ranges["C0"]["high"])
        min_k = float(sampling_ranges["tumor_k_abs"]["low"])
        if min_k <= float(capacity_margin) * max_c0:
            raise ValueError(
                "tumor_k_abs.low must exceed capacity_margin * C0.high to keep C0 and K_C independently sampled."
            )
    if "C0_fraction_K" in sampling_ranges:
        low = float(sampling_ranges["C0_fraction_K"]["low"])
        high = float(sampling_ranges["C0_fraction_K"]["high"])
        if low <= 0 or high >= 1:
            raise ValueError("C0_fraction_K must stay strictly between 0 and 1.")


def sample_independent_lhs_cohort(
    n_patients: int,
    cohort: str,
    seed: int,
    sampling_ranges: Mapping[str, Mapping],
    capacity_margin: float = 1.2,
) -> pd.DataFrame:
    """Generate a virtual patient cohort with one independent LHS dimension per parameter."""

    validate_sampling_ranges(sampling_ranges, capacity_margin=capacity_margin)
    names = list(sampling_ranges)
    sampler = qmc.LatinHypercube(d=len(names), seed=int(seed))
    unit = sampler.random(int(n_patients))
    data: dict[str, np.ndarray] = {}
    for idx, name in enumerate(names):
        spec = sampling_ranges[name]
        data[name] = _map_unit_interval(unit[:, idx], spec["low"], spec["high"], spec.get("mapping", "linear"))

    frame = pd.DataFrame(data)
    frame.insert(0, "patient_id", [f"{cohort}_{idx + 1:03d}" for idx in range(int(n_patients))])
    frame.insert(1, "cohort", str(cohort))
    frame.insert(2, "seed", [int(seed) * 1000 + idx + 1 for idx in range(int(n_patients))])
    centered = 2.0 * unit - 1.0
    frame["heterogeneity_level"] = np.sqrt(np.mean(centered**2, axis=1))
    return frame


def materialize_initial_conditions(cohort_df: pd.DataFrame, baseline_n: float) -> pd.DataFrame:
    """Compute ``C0`` and ``N0`` from independent sampling dimensions and check their physical bounds."""

    frame = cohort_df.copy()
    if {"C0_fraction_K", "tumor_k_abs"}.issubset(frame.columns):
        frame["C0"] = frame["C0_fraction_K"] * frame["tumor_k_abs"]
    if "N0_scale" in frame.columns:
        frame["N0"] = float(baseline_n) * frame["N0_scale"]
    if "C0" not in frame:
        raise KeyError("Cohort must contain C0 or both C0_fraction_K and tumor_k_abs.")
    if "N0" not in frame:
        frame["N0"] = float(baseline_n)
    if (frame["C0"] <= 0).any() or (frame["N0"] <= 0).any():
        raise ValueError("C0 and N0 must be positive.")
    if "tumor_k_abs" in frame and (frame["C0"] >= frame["tumor_k_abs"]).any():
        raise ValueError("Every virtual patient must satisfy C0 < K_C.")
    return frame


def build_patient_specs(
    cohort_df: pd.DataFrame,
    group_data,
    reaction_model,
    n_particles: int,
    rollout_seed: int,
) -> list[dict]:
    """Convert the LHS parameter table into virtual patient dictionaries for rollout."""

    from training.coupled_dynamics import _sample_group_tensor, _select_initial_z_anchor

    device = next(reaction_model.parameters()).device
    group_id = _select_initial_z_anchor(group_data)["group_id"]
    specs: list[dict] = []
    for spec in cohort_df.to_dict(orient="records"):
        seed = int(spec["seed"])
        spec["z0"] = _sample_group_tensor(
            group_data, group_id, int(n_particles), device, random_state=seed,
        )
        if "N0_scale" in spec:
            spec["n0"] = float(spec["N0_scale"])
        else:
            spec["n0"] = float(spec["N0"]) / float(reaction_model.abundance_reference)
        spec["rollout_seed"] = int(rollout_seed) + seed
        specs.append(spec)
    return specs


__all__ = [
    "build_patient_specs",
    "materialize_initial_conditions",
    "sample_independent_lhs_cohort",
    "validate_sampling_ranges",
]
