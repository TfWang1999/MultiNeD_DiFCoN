"""Paired alpha and paper summaries, calculated only from saved metrics."""
import numpy as np
import pandas as pd
from .paths import ALL_PERTURBATION_METRICS, ROBUSTNESS, new_output
PROGRAMS = ["Precursor", "Activation", "Early effector", "Cytotoxic", "Dysfunction"]
TREATMENTS = ["Control", "Anti-PD1", "Anti-4-1BB", "Combination"]

def finite(name, array):
    if not np.isfinite(np.asarray(array)).all():
        raise AssertionError(f"NaN/Inf in {name}")


def dynamic_importance(all_metrics, output):
    perturbed = all_metrics[all_metrics["delta"] != 0.0].copy()
    perturbed["abs_delta"] = perturbed["delta"].abs()
    records = []
    keys = ["treatment", "sim_seed", "intervention_day", "program", "abs_delta"]
    for key, source in perturbed.groupby(keys, sort=False):
        treatment, seed, day, program, delta = key
        delta = float(delta)
        assert len(source) == 2 and set(np.sign(source.delta)) == {-1.0, 1.0}
        plus_row = source.loc[source.delta > 0].iloc[0]
        minus_row = source.loc[source.delta < 0].iloc[0]
        assert np.isclose(plus_row.baseline_tumor_auc, minus_row.baseline_tumor_auc, rtol=0, atol=1e-12)
        auc0 = float(plus_row.baseline_tumor_auc)
        records.append({
            "treatment": treatment,
            "sim_seed": int(seed),
            "intervention_day": float(day),
            "program": program,
            "delta_sd": delta,
            "baseline_tumor_auc": auc0,
            "tumor_auc_minus": float(minus_row.tumor_auc),
            "tumor_auc_plus": float(plus_row.tumor_auc),
            "alpha": (float(minus_row.tumor_auc) - float(plus_row.tumor_auc)) / (2.0 * delta * auc0),
        })
    per_pair = pd.DataFrame(records)
    treatment_order = {name: index for index, name in enumerate(TREATMENTS)}
    program_order = {name: index for index, name in enumerate(PROGRAMS)}
    per_pair = per_pair.assign(
        _treatment_order=per_pair.treatment.map(treatment_order),
        _program_order=per_pair.program.map(program_order),
    ).sort_values(["_treatment_order", "_program_order", "delta_sd", "intervention_day", "sim_seed"], kind="stable")
    finite("dynamic importance pairs", per_pair.select_dtypes(include="number").to_numpy())

    by_delta = per_pair.groupby(["treatment", "program", "delta_sd"], sort=False)["alpha"].agg(
        alpha_mean="mean", alpha_sd="std", n_day_seed_pairs="size",
    ).reset_index()
    finite("dynamic importance by delta", by_delta.select_dtypes(include="number").to_numpy())
    by_delta.to_csv(output / "dynamic_importance_by_delta.csv", index=False, float_format="%.17g")


def run(output, metrics=None):
    metrics = metrics or ROBUSTNESS / ALL_PERTURBATION_METRICS
    frame = pd.read_csv(metrics, float_precision="round_trip")
    dynamic_importance(frame, new_output(output))
