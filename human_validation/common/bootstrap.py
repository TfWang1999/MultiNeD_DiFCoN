"""Paired patient bootstrap, with explicit arguments instead of mutable module state."""
from __future__ import annotations
import numpy as np
import pandas as pd
from common.ridge import harrell_c_index
BOOTSTRAP_REPLICATES = 1000
BASE_RANDOM_SEED = 20260920
TIE_TOLERANCE = 1e-8
def comparable_pair_arrays(
    time: np.ndarray,
    event: np.ndarray,
    risks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Precompute comparable patient pairs and their concordance credits."""

    left_parts: list[np.ndarray] = []
    right_parts: list[np.ndarray] = []
    for event_index in np.flatnonzero(event):
        comparable = np.flatnonzero(
            (time > time[event_index])
            | ((time == time[event_index]) & (event == 0))
        )
        if comparable.size:
            left_parts.append(np.full(comparable.size, event_index, dtype=np.int64))
            right_parts.append(comparable.astype(np.int64, copy=False))
    if not left_parts:
        raise ValueError("No comparable survival pairs")

    left = np.concatenate(left_parts)
    right = np.concatenate(right_parts)
    differences = risks[left] - risks[right]
    credits = np.full(differences.shape, 0.5, dtype=float)
    credits[differences > TIE_TOLERANCE] = 1.0
    credits[differences < -TIE_TOLERANCE] = 0.0
    return left, right, credits


def bootstrap_cindices(
    frame: pd.DataFrame,
    seed: int,
    *, scores: list[str], risk_columns: dict[str, str],
) -> tuple[list[dict[str, float | int]], np.ndarray]:
    """Bootstrap all scores together using patient multiplicities."""

    time = frame["os_time_days"].to_numpy(dtype=float)
    event = frame["os_event"].to_numpy(dtype=int)
    risks = frame[[risk_columns[name] for name in scores]].to_numpy(dtype=float)
    left, right, credits = comparable_pair_arrays(time, event, risks)

    observed = [harrell_c_index(time, event, risks[:, index]) for index in range(len(scores))]

    n = len(frame)
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(
        n,
        np.full(n, 1.0 / n, dtype=float),
        size=BOOTSTRAP_REPLICATES,
    )
    estimates = np.full((BOOTSTRAP_REPLICATES, len(scores)), np.nan, dtype=float)
    batch_size = 32
    for start in range(0, BOOTSTRAP_REPLICATES, batch_size):
        stop = min(start + batch_size, BOOTSTRAP_REPLICATES)
        pair_weights = counts[start:stop, left] * counts[start:stop, right]
        denominators = pair_weights.sum(axis=1)
        valid = denominators > 0
        estimates[start:stop][valid] = (
            pair_weights[valid] @ credits
        ) / denominators[valid, None]

    return observed, estimates


def percentile_summary(values: np.ndarray) -> tuple[float, float, float, float]:
    low, high = np.quantile(values, [0.025, 0.975], method="linear")
    return (
        float(values.mean()),
        float(values.std(ddof=1)),
        float(low),
        float(high),
    )
