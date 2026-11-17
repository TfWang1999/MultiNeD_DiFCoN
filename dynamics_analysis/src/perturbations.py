"""Replay saved weighted trajectories for paired program perturbations."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch
from scipy.integrate import trapezoid

from .model_access import ARMS, PROGRAMS, SCORE_COLS, TREATMENTS, load_final_reaction
from .paths import ALL_PERTURBATION_METRICS, REPO, ROBUSTNESS, new_output


LEGACY = {"Control": "Control", "Anti-PD1": "Anti-PD1", "Anti-41BB": "Anti-4-1BB", "Anti-PD1+Anti-41BB": "Combination"}
INTERVENTION_DAYS = tuple(range(1, 7))
STATE_ATOL, STATE_RTOL = 3e-5, 2e-6
TUMOR_ATOL, TUMOR_RTOL = 0.002, 2e-5


def replay(model, initial, times, P, B, indices):
    return model.simulate(
        initial["z"], P=P, B=B, day0=float(times[0]), time_points=times,
        volume0=initial["volume"], n0=initial["N"],
        weights0=initial["particle_weights"], inflow_indices=indices,
    )


@torch.no_grad()
def run(cfg, output, inputs, magnitudes):
    """Save paired AUCs using the same inflow draws before and after intervention."""
    if not magnitudes or len(set(magnitudes)) != len(magnitudes) or any(not np.isfinite(v) or v <= 0 for v in magnitudes):
        raise ValueError("abs_deltas must contain distinct positive finite magnitudes")
    deltas = [signed for magnitude in magnitudes for signed in (magnitude, -magnitude)]
    output = new_output(output)
    metrics_name = ALL_PERTURBATION_METRICS if output == ROBUSTNESS else "program_perturbation_metrics.csv"
    torch.set_num_threads(int(cfg["cpu_threads"]))
    model, model_info = load_final_reaction(REPO, cfg)
    summary = pd.read_csv(inputs / "paired_timeseries.csv", float_precision="round_trip")
    particles = pd.read_csv(
        inputs / "particle_functional_states.csv.gz", float_precision="round_trip",
        usecols=["treatment", "sim_seed", "time", "particle_id", "particle_weight", "q_in_index", *SCORE_COLS],
    )
    for frame in (summary, particles):
        frame["treatment"] = frame.treatment.replace(LEGACY)
    assert set(summary.treatment) == set(TREATMENTS)

    device = next(model.parameters()).device
    n_inflow = model.config.n_inflow_particles
    metrics = []
    for treatment, (P, B) in zip(TREATMENTS, ARMS):
        groups = summary.loc[summary.treatment == treatment].groupby("sim_seed", sort=True)
        for seed, group in groups:
            group = group.sort_values("time")
            times = group.time.to_numpy(np.float64)
            points = particles.loc[
                (particles.treatment == treatment) & (particles.sim_seed == seed)
            ].sort_values(["time", "particle_id"])
            snapshots = list(points.groupby("time", sort=True))
            np.testing.assert_array_equal([time for time, _ in snapshots], times)

            first, last = snapshots[0][1], snapshots[-1][1]
            n_initial = len(first)
            indices = torch.as_tensor(
                last["q_in_index"].to_numpy(np.int64)[n_initial:], device=device,
            ).reshape(-1, n_inflow)
            initial = {
                "z": torch.as_tensor(first[SCORE_COLS].to_numpy(np.float32), device=device),
                "particle_weights": torch.as_tensor(first["particle_weight"].to_numpy(np.float32), device=device),
                "volume": group.volume.iloc[0],
                "N": group.N.iloc[0],
            }
            baseline = replay(model, initial, times, P, B, indices)
            baseline_volume = np.array([float(record["volume"]) for record in baseline])
            np.testing.assert_allclose(baseline_volume, group.volume, rtol=TUMOR_RTOL, atol=TUMOR_ATOL)
            np.testing.assert_allclose(
                [float(record["N"]) for record in baseline], group.N, rtol=STATE_RTOL, atol=STATE_ATOL,
            )
            for record, (_, snapshot) in zip(baseline, snapshots, strict=True):
                np.testing.assert_allclose(
                    record["z"].cpu().numpy(), snapshot[SCORE_COLS], rtol=STATE_RTOL, atol=STATE_ATOL,
                )
                np.testing.assert_allclose(
                    record["particle_weights"].cpu().numpy(), snapshot["particle_weight"],
                    rtol=STATE_RTOL, atol=STATE_ATOL,
                )
            baseline_auc = float(trapezoid(baseline_volume, times))

            for day in INTERVENTION_DAYS:
                k, = np.flatnonzero(np.isclose(times, day, atol=1e-9, rtol=0))
                state = baseline[k]
                next_inflow = (len(state["z"]) - n_initial) // n_inflow
                for j, program in enumerate(PROGRAMS):
                    for delta in deltas:
                        modified = state["z"].clone()
                        modified[:, j] += delta
                        suffix = replay(model, {**state, "z": modified}, times[k:], P, B, indices[next_inflow:])
                        tumor = np.concatenate((baseline_volume[:k], [float(record["volume"]) for record in suffix]))
                        metrics.append({
                            "treatment": treatment,
                            "sim_seed": seed,
                            "intervention_day": day,
                            "program": program,
                            "delta": delta,
                            "tumor_auc": float(trapezoid(tumor, times)),
                            "baseline_tumor_auc": baseline_auc,
                        })
            print(f"Completed {treatment}, seed={seed}", flush=True)

    metric_frame = pd.DataFrame(metrics)
    if not np.isfinite(metric_frame.select_dtypes(include="number").to_numpy()).all():
        raise FloatingPointError("Non-finite perturbation metrics")
    metric_frame.to_csv(output / metrics_name, index=False, float_format="%.17g")
    settings = {**cfg, "inputs": str(inputs), "intervention_days": INTERVENTION_DAYS, "abs_deltas": magnitudes}
    report = {"status": "passed", "config": settings, "model": model_info, "metric_rows": len(metric_frame)}
    (output / "run_record.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
