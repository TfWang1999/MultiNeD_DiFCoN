"""Shared frozen-program scoring and survival-analysis functions.

Arguments select cohort-specific column names and the human gene-ID field
for ortholog mapping, frozen program scores and survival analysis.
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import chi2
from statsmodels.duration.hazard_regression import PHReg


SOURCE_TO_MODEL = {
    "Naive-Like": "Precursor",
    "Memory-Like": "Precursor",
    "Naive-Il7r": "Precursor",
    "Activation": "Activation",
    "Eff-Early": "Early effector",
    "Eff-Cytox": "Cytotoxic",
    "Dysf": "Dysfunction",
}
PROGRAM_ORDER = [
    "Precursor",
    "Activation",
    "Early effector",
    "Cytotoxic",
    "Dysfunction",
]


def zscore_rows(log_expression: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Population-Z-standardize each gene across the supplied cohort."""

    mean = log_expression.mean(axis=1)
    sd = log_expression.std(axis=1, ddof=0)
    variable = sd > 0
    z = log_expression.sub(mean, axis=0)
    z.loc[variable] = z.loc[variable].div(sd.loc[variable], axis=0)
    z.loc[~variable] = 0.0
    return z, variable


def prepare_weights(path: Path) -> pd.DataFrame:
    """Read and normalize the frozen top-50 weights for required source programs."""

    frame = pd.read_csv(path)
    frame = frame[frame["program"].isin(SOURCE_TO_MODEL)].copy()
    counts = frame.groupby("program").size()
    if not (counts == 50).all() or set(counts.index) != set(SOURCE_TO_MODEL):
        raise ValueError(f"Expected 50 fixed genes for each required source program; got {counts.to_dict()}")
    frame["model_program"] = frame["program"].map(SOURCE_TO_MODEL)
    frame["mouse_weight_normalized"] = frame["weight"] / frame.groupby("program")["weight"].transform("sum")
    return frame.rename(columns={"program": "source_program", "gene": "mouse_symbol", "weight": "mouse_weight"})


def build_ortholog_mapping(
    weights: pd.DataFrame,
    hcop_path: Path,
    expression_ids: set[str],
    human_id_column: str = "human_entrez_gene",
) -> pd.DataFrame:
    """Map fixed mouse genes to expressed human orthologs and allocate their weights."""

    valid_id_columns = {"human_entrez_gene", "human_ensembl_gene"}
    if human_id_column not in valid_id_columns:
        raise ValueError(f"human_id_column must be one of {sorted(valid_id_columns)}")

    hcop = pd.read_csv(hcop_path, sep="\t", compression="gzip", dtype=str, low_memory=False)
    hcop = hcop[hcop["mouse_symbol"].isin(weights["mouse_symbol"])].copy()
    hcop = hcop[
        [
            "mouse_entrez_gene",
            "mouse_ensembl_gene",
            "mgi_id",
            "mouse_symbol",
            "human_entrez_gene",
            "human_ensembl_gene",
            "hgnc_id",
            "human_symbol",
            "support",
        ]
    ].drop_duplicates()
    mapping = weights.merge(hcop, how="left", on="mouse_symbol", validate="many_to_many")
    mapping["expression_present"] = mapping[human_id_column].isin(expression_ids)
    mapping["support_count"] = mapping["support"].fillna("").map(
        lambda value: len([item for item in str(value).split(",") if item.strip()])
    )
    key = ["source_program", "mouse_symbol"]
    mapping["hcop_ortholog_count"] = mapping.groupby(key)[human_id_column].transform("nunique")
    mapping["expressed_ortholog_count"] = mapping.groupby(key)["expression_present"].transform("sum")
    mapping["mapping_status"] = np.select(
        [mapping[human_id_column].isna(), mapping["expression_present"]],
        ["no_hcop_ortholog", "used_for_scoring"],
        default="ortholog_not_in_expression",
    )
    mapping["weight_share_before_coverage_normalization"] = np.where(
        mapping["expression_present"],
        mapping["mouse_weight_normalized"] / mapping["expressed_ortholog_count"].replace(0, np.nan),
        0.0,
    )
    coverage_mass = mapping.groupby("source_program")["weight_share_before_coverage_normalization"].transform("sum")
    mapping["weight_used_for_scoring"] = np.where(
        mapping["expression_present"],
        mapping["weight_share_before_coverage_normalization"] / coverage_mass,
        0.0,
    )

    mapping = mapping.sort_values(
        ["model_program", "source_program", "mouse_weight_normalized", "mouse_symbol", "human_symbol"],
        ascending=[True, True, False, True, True],
        na_position="last",
    )
    return mapping


def calculate_scores(
    expression_z: pd.DataFrame,
    mapping: pd.DataFrame,
    sample_titles: list[str],
    human_id_column: str = "human_entrez_gene",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate raw and cohort-standardized frozen program scores."""

    source_scores: dict[str, np.ndarray] = {}
    for source_program in SOURCE_TO_MODEL:
        used = mapping[(mapping["source_program"] == source_program) & mapping["expression_present"]].copy()
        gene_weights = used.groupby(human_id_column)["weight_used_for_scoring"].sum()
        if gene_weights.empty or not np.isclose(gene_weights.sum(), 1.0, atol=1e-10):
            raise ValueError(f"Invalid scoring weights for {source_program}: sum={gene_weights.sum()}")
        values = expression_z.loc[gene_weights.index, sample_titles].to_numpy(dtype=float)
        source_scores[source_program] = gene_weights.to_numpy(dtype=float) @ values
    source_frame = pd.DataFrame(source_scores, index=sample_titles)
    raw = pd.DataFrame(index=sample_titles)
    raw["Precursor"] = source_frame[["Naive-Like", "Memory-Like", "Naive-Il7r"]].mean(axis=1)
    raw["Activation"] = source_frame["Activation"]
    raw["Early effector"] = source_frame["Eff-Early"]
    raw["Cytotoxic"] = source_frame["Eff-Cytox"]
    raw["Dysfunction"] = source_frame["Dysf"]
    standardized = (raw - raw.mean(axis=0)) / raw.std(axis=0, ddof=0)
    if not np.isfinite(standardized.to_numpy()).all():
        raise ValueError("Non-finite standardized program score")
    source_frame.index.name = "sample_title"
    raw.index.name = "sample_title"
    standardized.index.name = "sample_title"
    return raw, standardized


def load_alpha(path: Path, treatment: str) -> pd.Series:
    """Read the five mean sensitivity weights at delta=0.5."""

    frame = pd.read_csv(path)
    selected = frame.loc[frame["treatment"].eq(treatment) & np.isclose(frame["delta_sd"], 0.5)]
    if selected["program"].duplicated().any() or set(selected["program"]) != set(PROGRAM_ORDER):
        raise ValueError(f"{treatment} delta=0.5 alpha rows are incomplete or duplicated")
    return selected.set_index("program").loc[PROGRAM_ORDER, "alpha_mean"].astype(float)


def cox_models(
    scores: pd.DataFrame,
    score_names: Sequence[str],
    time_column: str = "os_time_weeks",
    event_column: str = "os_event",
) -> pd.DataFrame:
    """Fit separate continuous univariable Cox models with Efron ties."""

    rows = []
    for score_name in score_names:
        data = scores[[time_column, event_column, score_name]].dropna().copy()
        data = data[data[time_column] > 0]
        x_raw = data[score_name].to_numpy(dtype=float)
        x = (x_raw - x_raw.mean()) / x_raw.std(ddof=0)
        time = data[time_column].to_numpy(dtype=float)
        event = data[event_column].astype(int).to_numpy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = PHReg(time, x[:, None], status=event, ties="efron").fit(disp=0)
        beta = float(fit.params[0])
        ci_low, ci_high = fit.conf_int()[0]
        rows.append(
            {
                "score": score_name,
                "n": len(data),
                "events": int(event.sum()),
                "coefficient_per_score_sd": beta,
                "standard_error": float(fit.bse[0]),
                "hazard_ratio_per_score_sd": math.exp(beta),
                "hazard_ratio_ci95_low": math.exp(float(ci_low)),
                "hazard_ratio_ci95_high": math.exp(float(ci_high)),
                "wald_p_value": float(fit.pvalues[0]),
                "ties_method": "efron",
            }
        )
    return pd.DataFrame(rows)


def kaplan_meier(time: np.ndarray, event: np.ndarray) -> pd.DataFrame:
    """Calculate an event-time Kaplan-Meier step table."""

    event_times = np.sort(np.unique(time[event == 1]))
    survival = 1.0
    rows = [{"time_days": 0.0, "survival": 1.0, "at_risk": len(time), "events": 0}]
    for value in event_times:
        at_risk = int((time >= value).sum())
        events = int(((time == value) & (event == 1)).sum())
        survival *= 1.0 - events / at_risk
        rows.append(
            {"time_days": float(value), "survival": survival, "at_risk": at_risk, "events": events}
        )
    return pd.DataFrame(rows)


def logrank_test(time: np.ndarray, event: np.ndarray, high: np.ndarray) -> tuple[float, float]:
    """Compute a two-group log-rank test from follow-up times and event indicators."""

    observed_minus_expected = 0.0
    variance = 0.0
    for value in np.sort(np.unique(time[event == 1])):
        risk = time >= value
        n = int(risk.sum())
        n_high = int((risk & high).sum())
        events_here = (time == value) & (event == 1)
        d = int(events_here.sum())
        d_high = int((events_here & high).sum())
        if n <= 1:
            continue
        expected = d * n_high / n
        var = d * (n_high / n) * (1 - n_high / n) * ((n - d) / (n - 1))
        observed_minus_expected += d_high - expected
        variance += var
    statistic = observed_minus_expected**2 / variance if variance > 0 else np.nan
    p_value = float(chi2.sf(statistic, 1)) if np.isfinite(statistic) else np.nan
    return float(statistic), p_value


def km_analysis(scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate the TCGA frozen score's median-split KM curves and log-rank test."""

    score_name = "CD8score_Control"
    data = scores[["os_time_days", "os_event", score_name]].dropna()
    data = data.loc[data["os_time_days"] > 0]
    cutpoint = float(data[score_name].median())
    high = (data[score_name] >= cutpoint).to_numpy()
    time = data["os_time_days"].to_numpy(dtype=float)
    event = data["os_event"].astype(int).to_numpy()
    statistic, p_value = logrank_test(time, event, high)
    summary_rows = []
    curve_rows = []
    for group_label, mask in [("Low", ~high), ("High", high)]:
        curve = kaplan_meier(time[mask], event[mask])
        curve["score"] = score_name
        curve["group"] = group_label
        curve["median_cutpoint"] = cutpoint
        curve_rows.append(curve)
        crossed = curve.loc[curve["survival"] <= 0.5, "time_days"]
        summary_rows.append(
            {
                "score": score_name,
                "group": group_label,
                "median_cutpoint": cutpoint,
                "n": int(mask.sum()),
                "events": int(event[mask].sum()),
                "km_median_os_days": float(crossed.iloc[0]) if len(crossed) else np.nan,
                "logrank_chi_square": statistic,
                "logrank_p_value": p_value,
                "split_rule": "High >= cohort median; Low < cohort median",
            }
        )
    return pd.DataFrame(summary_rows), pd.concat(curve_rows, ignore_index=True)
