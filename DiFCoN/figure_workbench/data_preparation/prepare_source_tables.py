"""Rebuild the six figure-source workbooks and baseline statistics from saved experiments.

No training, simulation or figure export. Run from any working directory.
"""
from __future__ import annotations

import argparse
from io import StringIO
import json
from pathlib import Path
import sys
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon, t as student_t

WORKBENCH_DIR = Path(__file__).resolve().parents[1]
if str(WORKBENCH_DIR) not in sys.path:
    sys.path.insert(0, str(WORKBENCH_DIR))
from plotting._paths import (TRAINING_DIR, PILOT_DIR, OPENLOOP_DIR, INPUT_ATTRIBUTION_DIR,
                             SOURCE_DATA_DIR, BASELINE_DIR, PARETO_DIR, MULTISEED_DIR,
                             LOCKED_BENCHMARK_DIR)
from plotting._source_tables import TABLE_FILES
PROJECT_DIR = TRAINING_DIR


BENCHMARK_WEIGHTS = [0.1, 0.5, 1.0, 1.5, 2.0]


def optional_label(weight: float) -> str:
    return f"Optional\n$\\lambda_d={weight:g}$"


def holm_adjust(values):
    values = np.asarray(values, dtype=float)
    result = np.full(values.shape, np.nan)
    finite = np.isfinite(values)
    if not finite.any():
        return result
    p = values[finite]
    order = np.argsort(p)
    ranked = p[order]
    adjusted = np.maximum.accumulate((len(ranked) - np.arange(len(ranked))) * ranked)
    restored = np.empty_like(adjusted)
    restored[order] = np.minimum(adjusted, 1.0)
    result[finite] = restored
    return result


def significance_symbol(pvalue: float) -> str:
    if not np.isfinite(pvalue):
        return "n/a"
    if pvalue < 1e-4:
        return "****"
    if pvalue < 1e-3:
        return "***"
    if pvalue < 1e-2:
        return "**"
    if pvalue < 5e-2:
        return "*"
    return "ns"


def load_dose_weight_results():
    seed_summary = pd.read_csv(PARETO_DIR / "aggregate" / "seed_summary.csv").sort_values(
        "total_dose_weight"
    ).reset_index(drop=True)
    run_dirs = sorted(path for path in PARETO_DIR.iterdir() if path.is_dir() and path.name.startswith("dose_weight_"))
    summary_frames = []
    reference_cohorts = {}
    for run_index, run_dir in enumerate(run_dirs):
        config = json.loads((run_dir / "resolved_config.json").read_text(encoding="utf-8"))
        weight = float(config["loss"]["total_dose_weight"])
        seed = int(config["seeds"]["model"])
        table_dir = run_dir / "tables"
        for split in ("train", "validation", "test"):
            cohort = pd.read_csv(table_dir / f"{split}_lhs_cohort.csv")
            if run_index == 0:
                reference_cohorts[split] = cohort
            elif not cohort.equals(reference_cohorts[split]):
                raise ValueError(f"Dose-weight runs use different {split} cohorts: {run_dir}")

        summary = load_run_summary(run_dir)
        direct = summary.loc[summary["strategy"].eq("Optimal")].copy()
        direct["weight"] = weight
        direct["model_seed"] = seed
        summary_frames.append(direct)
        if run_index == 0:
            fixed_test = summary.loc[~summary["strategy"].eq("Optimal")].copy()

    direct_summary = pd.concat(summary_frames, ignore_index=True)
    assert len(run_dirs) == 24
    assert seed_summary.shape[0] == 8
    assert direct_summary.groupby(["cohort", "weight", "model_seed"])["patient_id"].nunique().eq(32).all()

    metrics = ["final_C_rel", "tumor_auc_rel", "total_dose"]
    test_patient = direct_summary.groupby(["weight", "patient_id"], as_index=False)[metrics].mean()
    return {
        "direct_summary": direct_summary,
        "fixed_test": fixed_test,
        "seed_summary": seed_summary,
        "test_patient": test_patient,
    }


STRATEGY_ORDER = ("Anti-41BB", "Anti-PD1", "Combine", "Optimal")


def _available_strategy_order(frame: pd.DataFrame) -> list[str]:
    present = set(frame["strategy"].dropna().astype(str))
    ordered = [strategy for strategy in STRATEGY_ORDER if strategy in present]
    return ordered + sorted(present.difference(ordered))


def _paired_test(reference: np.ndarray, comparator: np.ndarray, test: str) -> tuple[float, float]:
    """Test two strategies on matched patients."""

    reference = np.asarray(reference, dtype=float)
    comparator = np.asarray(comparator, dtype=float)
    diff = reference - comparator
    if len(diff) < 2:
        return np.nan, np.nan
    if np.allclose(diff, 0.0):
        return 0.0, 1.0
    if test == "paired_t":
        if np.allclose(np.std(diff, ddof=1), 0.0):
            return float(np.sign(np.mean(diff)) * np.inf), 0.0
        result = ttest_rel(reference, comparator, alternative="two-sided")
    elif test == "wilcoxon":
        result = wilcoxon(reference, comparator, alternative="two-sided")
    else:
        raise ValueError("test must be 'paired_t' or 'wilcoxon'.")
    return float(result.statistic), float(result.pvalue)


def paired_metric_pairwise_statistics(
    summary_df: pd.DataFrame,
    metrics: list[str] | tuple[str, ...],
    test: str = "paired_t",
) -> pd.DataFrame:
    """Compare all pairs of the four strategies within patients for each outcome."""

    from itertools import combinations

    strategy_order = _available_strategy_order(summary_df)
    rows = []
    for metric in metrics:
        pivot = summary_df.pivot(index="patient_id", columns="strategy", values=metric)
        for strategy_a, strategy_b in combinations(strategy_order, 2):
            paired = pivot[[strategy_a, strategy_b]].dropna()
            statistic, pvalue = _paired_test(
                paired[strategy_a].to_numpy(dtype=float),
                paired[strategy_b].to_numpy(dtype=float),
                test=test,
            )
            rows.append(
                {
                    "metric": metric,
                    "strategy_a": strategy_a,
                    "strategy_b": strategy_b,
                    "comparison": f"{strategy_a} vs {strategy_b}",
                    "test": test,
                    "n_pairs": len(paired),
                    "mean_a": float(paired[strategy_a].mean()),
                    "mean_b": float(paired[strategy_b].mean()),
                    "mean_difference_a_minus_b": float((paired[strategy_a] - paired[strategy_b]).mean()),
                    "statistic": statistic,
                    "p_value": pvalue,
                }
            )
    return pd.DataFrame(rows)


LOSS_ABLATION_ORDER = ("full", "wC0", "wCTB0", "wF0", "wE0")


LOSS_METRIC_ORDER = ("C_relative", "CTB", "L_F", "D_total")


OBSERVATION_ORDER = ("Full state", "Practical state", "Schedule only")


OPENLOOP_STRATEGY_ORDER = (
    "Fixed Combine",
    "Validation-derived open-loop",
    "Closed-loop",
)


def load_seed2031_loss_ablation_table() -> pd.DataFrame:
    """Load patient-level test results for Full and the four loss ablations at seed 2031."""
    refs = pd.read_csv(PILOT_DIR / "BASELINE_REFERENCES.tsv", sep="\t")
    full_ref = refs.loc[np.isclose(refs["lambda_d"].astype(float), 1.0), "existing_200epoch_reference"]
    if len(full_ref) != 1: raise ValueError("Cannot uniquely resolve λd=1.0 Full reference")
    run_map = {
        "full": PROJECT_DIR / str(full_ref.iloc[0]),
        "wC0": PILOT_DIR / "no_final_seed_2031",
        "wCTB0": PILOT_DIR / "no_auc_seed_2031",
        "wF0": PILOT_DIR / "no_functional_seed_2031",
        "wE0": PILOT_DIR / "no_efficiency_seed_2031",
    }

    frames, cols = [], ["patient_id", "final_C_rel", "tumor_auc_rel", "loss_functional_activity_loss", "total_dose"]
    for variant, run_dir in run_map.items():
        summary = load_run_summary(run_dir)
        x = summary.loc[summary["strategy"].eq("Optimal"), cols].copy()
        if x["patient_id"].duplicated().any(): raise ValueError(f"Duplicated patient rows in {run_dir}")
        if x["patient_id"].nunique() != 32: raise ValueError(f"Expected 32 test patients in {run_dir}")
        x["variant"], x["run_name"] = variant, run_dir.name
        frames.append(x)

    data = pd.concat(frames, ignore_index=True).rename(columns={
        "final_C_rel": "C_relative", "tumor_auc_rel": "CTB",
        "loss_functional_activity_loss": "L_F", "total_dose": "D_total",
    })

    patient_sets = {v: set(data.loc[data["variant"].eq(v), "patient_id"]) for v in run_map}
    ref = patient_sets["full"]
    if any(ids != ref for ids in patient_sets.values()): raise ValueError("Test-patient sets are not identical across ablations")
    return data


def summarize_seed2031_loss_ablation() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return patient-level values, means and ratios normalized to Full = 1."""
    data = load_seed2031_loss_ablation_table()
    mean_df = data.groupby("variant")[list(LOSS_METRIC_ORDER)].mean().reindex(LOSS_ABLATION_ORDER)
    rel_df = mean_df.div(mean_df.loc["full"], axis=1)
    return data, mean_df, rel_df


def _formal_observation_run(setting: str, seed: int) -> Path:
    """Locate the official run for one of the three O05 observation settings."""
    if setting == "Full state":
        return PARETO_DIR / f"dose_weight_1p0_seed_{seed}"

    prefix = {
        "Practical state": "practical_observation_seed_",
        "Schedule only": "schedule_only_seed_",
    }[setting]

    return MULTISEED_DIR / f"{prefix}{seed}"


def load_formal_observation_patient_results() -> pd.DataFrame:
    """Load the 288 raw O05 test outcomes across patients and seeds."""

    metrics = ["final_C_rel", "tumor_auc_rel", "total_dose"]
    frames = []

    for setting in OBSERVATION_ORDER:
        for seed in (2031, 2032, 2033):
            run_dir = _formal_observation_run(setting, seed)
            summary = load_run_summary(run_dir)

            x = summary.loc[
                summary["strategy"].eq("Optimal"),
                ["patient_id", *metrics],
            ].copy()

            if x["patient_id"].duplicated().any():
                raise ValueError(f"Duplicated patient rows in {run_dir}")
            if x["patient_id"].nunique() != 32:
                raise ValueError(
                    f"Expected 32 test patients in {run_dir}, "
                    f"got {x['patient_id'].nunique()}"
                )

            x["setting"] = setting
            x["model_seed"] = seed
            x["run_name"] = run_dir.name
            frames.append(x)

    data = pd.concat(frames, ignore_index=True)

    # Check that all settings and seeds use the same test patients.
    groups = {
        (setting, seed): set(
            data.loc[
                data["setting"].eq(setting) & data["model_seed"].eq(seed),
                "patient_id",
            ]
        )
        for setting in OBSERVATION_ORDER
        for seed in (2031, 2032, 2033)
    }
    ref = groups[("Full state", 2031)]
    if any(ids != ref for ids in groups.values()):
        raise ValueError("Test-patient sets differ across observation runs")

    return data


def average_formal_observation_over_seeds(
    data: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Average each test patient's outcomes across three training seeds, yielding 32 × 3 = 96 rows."""

    if data is None:
        data = load_formal_observation_patient_results()

    metrics = ["final_C_rel", "tumor_auc_rel", "total_dose"]

    patient_avg = (
        data.groupby(["patient_id", "setting"], as_index=False)[metrics]
        .mean()
    )

    # Verify that each setting has 32 patients.
    counts = patient_avg.groupby("setting")["patient_id"].nunique()
    if not all(counts.reindex(OBSERVATION_ORDER).eq(32)):
        raise ValueError(f"Expected 32 patients per setting, got:\n{counts}")

    return patient_avg


def formal_observation_patient_statistics(
    patient_avg: pd.DataFrame | None = None,
    *,
    paired_test: str = "paired_t",
) -> pd.DataFrame:
    """Compare settings for 32 patients after averaging over seeds, with Holm correction."""

    if patient_avg is None:
        patient_avg = average_formal_observation_over_seeds()

    metrics = ["final_C_rel", "tumor_auc_rel", "total_dose"]
    pairs = (
        ("Full state", "Practical state"),
        ("Full state", "Schedule only"),
        ("Practical state", "Schedule only"),
    )

    rows = []

    for metric in metrics:
        pivot = (
            patient_avg
            .pivot(index="patient_id", columns="setting", values=metric)
            .reindex(columns=OBSERVATION_ORDER)
            .dropna()
        )

        raw = []
        temp = []

        for a, b in pairs:
            if paired_test == "paired_t":
                stat, p = ttest_rel(pivot[a], pivot[b])
            elif paired_test == "wilcoxon":
                stat, p = wilcoxon(pivot[a], pivot[b], alternative="two-sided")
            else:
                raise ValueError("paired_test must be 'paired_t' or 'wilcoxon'")

            raw.append(float(p))
            temp.append({
                "metric": metric,
                "setting_a": a,
                "setting_b": b,
                "n": len(pivot),
                "statistic": float(stat),
                "p_value": float(p),
                "mean_difference": float((pivot[b] - pivot[a]).mean()),
            })

        # Apply Holm correction without an additional dependency.
        order = np.argsort(raw)
        adjusted = np.empty(len(raw), dtype=float)
        running = 0.0

        for rank, idx in enumerate(order):
            value = min(1.0, (len(raw) - rank) * raw[idx])
            running = max(running, value)
            adjusted[idx] = running

        for row, p_adj in zip(temp, adjusted):
            row["p_holm"] = float(p_adj)
            rows.append(row)

    return pd.DataFrame(rows)


def openloop_pairwise_statistics(
    patient_means: pd.DataFrame,
    metrics: Sequence[str],
    *,
    paired_test: str = "paired_t",
    p_adjust: str = "holm",
) -> pd.DataFrame:
    """Compare matched virtual patients after averaging the three model seeds.

    The test unit is a virtual patient (n=32), rather than its three repeated
    model-seed simulations.  Untreated controls are excluded; Holm adjustment
    is applied to the three treatment-strategy pairs within each endpoint,
    matching the loss-weight benchmark convention.
    """

    if paired_test != "paired_t":
        raise ValueError("Only the two-sided paired t-test is currently supported.")
    if p_adjust not in {"holm", "none"}:
        raise ValueError("p_adjust must be 'holm' or 'none'.")
    from itertools import combinations

    available_strategies = set(patient_means["strategy"].astype(str))
    strategy_order = [strategy for strategy in OPENLOOP_STRATEGY_ORDER if strategy in available_strategies]
    if len(strategy_order) != len(OPENLOOP_STRATEGY_ORDER):
        missing = sorted(set(OPENLOOP_STRATEGY_ORDER).difference(available_strategies))
        raise ValueError(f"Missing expected open-loop strategies: {missing}")
    rows: list[dict[str, object]] = []
    for metric in metrics:
        pivot = patient_means.pivot(index="patient_id", columns="strategy", values=metric).reindex(columns=strategy_order)
        metric_rows: list[dict[str, object]] = []
        for strategy_a, strategy_b in combinations(strategy_order, 2):
            paired = pivot[[strategy_a, strategy_b]].dropna()
            values_a = paired[strategy_a].to_numpy(dtype=float)
            values_b = paired[strategy_b].to_numpy(dtype=float)
            difference = values_a - values_b
            if np.allclose(difference, 0.0):
                statistic, pvalue = 0.0, 1.0
            elif np.allclose(np.std(difference, ddof=1), 0.0):
                statistic, pvalue = float(np.sign(np.mean(difference)) * np.inf), 0.0
            else:
                result = ttest_rel(values_a, values_b, alternative="two-sided")
                statistic, pvalue = float(result.statistic), float(result.pvalue)
            metric_rows.append(
                {
                    "metric": metric,
                    "strategy_a": strategy_a,
                    "strategy_b": strategy_b,
                    "comparison": f"{strategy_a} vs {strategy_b}",
                    "test": "two-sided paired t-test",
                    "pairing_unit": "virtual patient; mean across 3 model seeds",
                    "n_pairs": len(paired),
                    "mean_a": float(paired[strategy_a].mean()),
                    "mean_b": float(paired[strategy_b].mean()),
                    "mean_difference_a_minus_b": float((paired[strategy_a] - paired[strategy_b]).mean()),
                    "statistic": statistic,
                    "p_value_raw": pvalue,
                }
            )
        raw = [row["p_value_raw"] for row in metric_rows]
        corrected = holm_adjust(raw) if p_adjust == "holm" else np.asarray(raw, dtype=float)
        for row, value in zip(metric_rows, corrected, strict=True):
            row["p_adjustment"] = p_adjust
            row["p_value_display"] = float(value)
            row["significance"] = significance_symbol(float(value))
        rows.extend(metric_rows)
    return pd.DataFrame(rows)


def load_run_summary(run_dir: Path) -> pd.DataFrame:
    return pd.read_csv(Path(run_dir) / "tables" / "test_strategy_summary.csv")

def baseline_pairwise_statistics(metrics, *, paired_test="paired_t"):
    summary = load_run_summary(BASELINE_DIR)
    return paired_metric_pairwise_statistics(summary, list(metrics), test=paired_test)


def build_fixed_control_benchmark(data):
    fixed_test = data['fixed_test']
    test_patient = data['test_patient']
    comparison_parts = []
    combine_part = fixed_test.loc[fixed_test['strategy'].eq('Combine'), ['patient_id', 'final_C_rel', 'tumor_auc_rel', 'total_dose']].copy()
    combine_part['group'] = 'Combine'
    comparison_parts.append(combine_part)
    for weight in BENCHMARK_WEIGHTS:
        part = test_patient.loc[np.isclose(test_patient['weight'], weight), ['patient_id', 'final_C_rel', 'tumor_auc_rel', 'total_dose']].copy()
        part['group'] = optional_label(weight)
        comparison_parts.append(part)
    comparison = pd.concat(comparison_parts, ignore_index=True)
    return {"fixed_control_comparison": comparison}


def build_dose_weight_tables():
    data = load_dose_weight_results()
    tables = {"cohort_summary": data["seed_summary"]}
    tables.update(build_fixed_control_benchmark(data))
    full = data["direct_summary"].loc[data["direct_summary"]["weight"].eq(1.0)].drop(columns="weight").reset_index(drop=True)
    full = full[[column for column in full if column != "model_seed"] + ["model_seed"]]
    tables["full_policy_patient_by_seed"] = full
    return tables

def _bootstrap_mean_ci(values: np.ndarray, seed: int = 20260901, n_boot: int = 10000) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(int(seed))
    samples = rng.choice(values, size=(int(n_boot), values.size), replace=True).mean(axis=1)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))

def build_attribution_tables() -> dict[str, pd.DataFrame]:
    attribution_dir = INPUT_ATTRIBUTION_DIR / 'full_validation'
    table_dir = attribution_dir / 'tables'
    lag = pd.read_csv(table_dir / 'policy_input_lag_summary.csv')
    feature = pd.read_csv(table_dir / 'policy_input_attribution_summary.csv')
    paired = pd.read_csv(table_dir / 'history_intervention_paired_metrics.csv')
    patient = paired[paired['history_intervention'] != 'observed'].groupby(['history_intervention', 'patient_id'], observed=True).agg(delta_loss=('delta_loss', 'mean'), delta_final_C_rel=('delta_final_C_rel', 'mean'), delta_tumor_auc_rel=('delta_tumor_auc_rel', 'mean'), delta_total_dose=('delta_total_dose', 'mean')).reset_index()
    statistics = []
    for index, (mode, part) in enumerate(patient.groupby('history_intervention', observed=True)):
        values = part['delta_loss'].to_numpy(dtype=float)
        low, high = _bootstrap_mean_ci(values, seed=20260901 + index)
        try:
            test = wilcoxon(values, alternative='two-sided', zero_method='wilcox')
            p_value = float(test.pvalue)
        except ValueError:
            p_value = np.nan
        statistics.append({'history_intervention': mode, 'n_patients': int(values.size), 'delta_loss_mean': float(np.mean(values)), 'delta_loss_median': float(np.median(values)), 'bootstrap_mean_ci95_low': low, 'bootstrap_mean_ci95_high': high, 'wilcoxon_two_sided_p': p_value, 'fraction_delta_loss_positive': float(np.mean(values > 0.0)), 'replication_unit': 'patient after averaging three policy seeds'})
    statistics = pd.DataFrame(statistics)
    feature_past_current = feature.copy()
    feature_past_current['history_role'] = np.where(feature_past_current['lag_from_current'] == 0, 'current', 'past')
    feature_past_current = feature_past_current.groupby(['output', 'history_role', 'feature'], observed=True).agg(absolute_integrated_gradient_mean=('absolute_integrated_gradient_mean', 'sum'), scaled_absolute_gradient_mean=('scaled_absolute_gradient_mean', 'sum')).reset_index()
    tables = {'lag_attribution': lag, 'intervention_patient_means': patient, 'intervention_statistics': statistics, 'feature_attribution': feature_past_current}
    return tables


def build_locked_plot_inputs():
    locked_root = LOCKED_BENCHMARK_DIR
    locked_combined = locked_root / 'combined'
    locked_patient = pd.read_csv(locked_combined / 'patient_seed_mean_metrics.csv')
    locked_official = pd.read_csv(locked_combined / 'paired_primary_comparisons_holm.csv')
    locked_metric_parts = [pd.read_csv(path) for path in sorted(locked_root.glob('seed_*/all_patient_metrics.csv'))]
    locked_compute_parts = [pd.read_csv(path) for path in sorted(locked_root.glob('seed_*/all_compute_accounting.csv'))]
    locked_metrics_all = pd.concat(locked_metric_parts, ignore_index=True)
    locked_compute_raw = pd.concat(locked_compute_parts, ignore_index=True)
    assert locked_patient['patient_id'].nunique() == 32
    assert locked_metrics_all['model_seed'].nunique() == 3 and len(locked_metrics_all) == 960

    def locked_mean_ci(values):
        values = pd.Series(values).dropna().to_numpy(float)
        mean = float(values.mean())
        sd = float(values.std(ddof=1))
        critical = float(student_t.ppf(0.975, len(values) - 1))
        half = critical * sd / np.sqrt(len(values))
        return mean, sd, mean - half, mean + half

    LOCKED_METRICS = ('loss', 'final_C_rel', 'tumor_auc_rel', 'total_dose')
    patient_by_strategy = {
        metric: locked_patient.pivot(index='patient_id', columns='strategy', values=metric)
        for metric in LOCKED_METRICS
    }
    summary_rows = []
    for strategy, part in locked_patient.groupby('strategy'):
        row = {'strategy': strategy, 'patient_count': part['patient_id'].nunique()}
        for metric in LOCKED_METRICS:
            mean, sd, low, high = locked_mean_ci(part[metric])
            row.update({f'mean_{metric}': mean, f'sd_{metric}': sd,
                        f'ci95_low_{metric}': low, f'ci95_high_{metric}': high})
        summary_rows.append(row)
    locked_summary = pd.DataFrame(summary_rows).sort_values('mean_loss')

    comparison_rows = []
    for _, official in locked_official.iterrows():
        metric = official['metric']; comparator = official['comparator']
        wide = patient_by_strategy[metric]
        difference = wide[comparator] - wide['Amortized 1D-CNN']
        mean, sd, low, high = locked_mean_ci(difference)
        comparison_rows.append({
            **official.to_dict(), 'mean_comparator_minus_reference': mean,
            'sd_paired_difference': sd, 'ci95_low': low, 'ci95_high': high,
            'holm_significant_0_05': bool(official['pvalue_holm'] < 0.05),
        })
    locked_comparisons = pd.DataFrame(comparison_rows)
    locked_seed = locked_metrics_all.groupby(['model_seed', 'strategy'], as_index=False).agg(
        mean_loss=('loss', 'mean'), mean_final_C_rel=('final_C_rel', 'mean'),
        mean_tumor_auc_rel=('tumor_auc_rel', 'mean'), mean_total_dose=('total_dose', 'mean'),
        patient_count=('patient_id', 'nunique'))

    compute_rows = []
    for (strategy, method_group), part in locked_compute_raw.groupby(['strategy', 'method_group']):
        if part['patient_id'].notna().any():
            wall = float(part['wall_seconds'].mean())
            forward = float(part['forward_evaluations'].mean())
            unit = 'per-patient online optimization'
        else:
            wall = float(part['marginal_seconds_per_patient'].dropna().mean())
            forward = float((part['forward_evaluations'] / part['patient_count']).mean())
            unit = 'per-patient frozen-policy inference'
        compute_rows.append({'strategy': strategy, 'method_group': method_group,
                             'wall_seconds_per_patient': wall,
                             'forward_evaluations_per_patient': forward,
                             'accounting_unit': unit})
    locked_compute = pd.DataFrame(compute_rows)


    core_comparators = ['Diagonal CMA-ES', 'SAC', 'Per-patient gradient open loop', 'Rolling-horizon MPC']
    reference_strategy = 'Amortized 1D-CNN'
    paired_parts = []
    for metric in ['final_C_rel', 'total_dose']:
        wide = patient_by_strategy[metric]
        for comp in core_comparators:
            diff = (wide[comp] - wide[reference_strategy]).dropna()
            paired_parts.append(pd.DataFrame({'patient_id': diff.index, 'comparator': comp, 'metric': metric, 'difference': diff.values}))
    paired_diff_df = pd.concat(paired_parts, ignore_index=True)

    wide_C = patient_by_strategy['final_C_rel']
    wide_D = patient_by_strategy['total_dose']

    rows = []
    for c in core_comparators:
        rows.append(pd.DataFrame({
            'patient_id': wide_C.index,
            'comparator': c,
            'delta_C': wide_C[c] - wide_C[reference_strategy],
            'delta_D': wide_D[c] - wide_D[reference_strategy],
        }).dropna())

    pareto_df = pd.concat(rows, ignore_index=True)

    methods = ['Amortized 1D-CNN', 'SAC', 'Per-patient gradient open loop', 'Rolling-horizon MPC', 'Diagonal CMA-ES']
    actions_all = pd.concat([pd.read_csv(p) for p in sorted(locked_root.glob('seed_*/all_actions.csv'))], ignore_index=True)
    action_patient = (actions_all[actions_all['strategy'].isin(methods)]
                      .groupby(['patient_id', 'strategy', 'day'], as_index=False)
                      .agg(action_PD1=('action_PD1', 'mean'), action_B41BB=('action_B41BB', 'mean')))
    action_mean = (action_patient.groupby(['strategy', 'day'], as_index=False)
                   .agg(action_PD1=('action_PD1', 'mean'), action_B41BB=('action_B41BB', 'mean')))


    for seed, part in locked_metrics_all.groupby('model_seed'):
        assert set(part['cohort']) == {'test'}
        assert part.groupby('strategy')['patient_id'].nunique().eq(32).all()
        assert not part.duplicated(['strategy','patient_id','model_seed']).any()
    return {'patient_metrics': locked_patient, 'strategy_summary': locked_summary.reset_index(drop=True),
             'paired_comparisons': locked_comparisons, 'seed_metrics': locked_seed,
            'compute_summary': locked_compute, 'patient_pair_differences': paired_diff_df,
            'patient_tradeoff': pareto_df, 'daily_action_means': action_mean}


def build_openloop_plot_inputs(patient_means):
    means = patient_means.loc[patient_means['strategy'].isin(['Validation-derived open-loop','Closed-loop'])]
    pivot = means.pivot(index='patient_id',columns='strategy',values=['final_C_rel','total_dose']).dropna()
    differences = pd.DataFrame({'patient_id':pivot.index,
        'dose_open_minus_closed':pivot['total_dose']['Validation-derived open-loop']-pivot['total_dose']['Closed-loop'],
        'tumour_open_minus_closed':pivot['final_C_rel']['Validation-derived open-loop']-pivot['final_C_rel']['Closed-loop']}).reset_index(drop=True)
    trajectories = pd.read_csv(OPENLOOP_DIR / 'test_trajectories.csv')
    patient_means = trajectories.groupby(['patient_id','strategy','day'],observed=True,as_index=False)['C_rel'].mean()
    grouped = patient_means.groupby(['strategy','day'],observed=True)['C_rel'].agg(['mean','sem']).reset_index()
    return {'feedback_patient_differences':differences,'mean_tumour_trajectories':grouped}

def build_full_practical_tests(patient):
    # Save the raw pair-specific tests alongside the Holm-adjusted figure statistics.
    rows=[]
    for metric in ['final_C_rel','total_dose']:
        pivot=patient.pivot(index='patient_id',columns='setting',values=metric).reindex(columns=['Full state','Practical state']).dropna()
        a,b=pivot['Full state'],pivot['Practical state']
        for test in ['paired_t','wilcoxon']:
            statistic,pvalue=ttest_rel(a,b) if test=='paired_t' else wilcoxon(a,b,alternative='two-sided')
            rows.append({'metric':metric,'test':test,'mean_difference':float((b-a).mean()),'statistic':float(statistic),'p_value':float(pvalue),'n':len(pivot)})
    return pd.DataFrame(rows)


def build_all_tables():
    patients, means, ratios = summarize_seed2031_loss_ablation()
    mean_ratios = means.add_suffix("_mean").join(ratios.add_suffix("_relative")).reset_index()
    observation = load_formal_observation_patient_results()
    observation_means = average_formal_observation_over_seeds(observation)
    metrics = ["final_C_rel", "tumor_auc_rel", "total_dose"]
    openloop = pd.read_csv(OPENLOOP_DIR / "test_patient_level_results_long.csv")
    openloop = openloop.loc[openloop["strategy"].isin([
        "Fixed Combine", "Validation-derived open-loop", "Closed-loop"
    ])].groupby(["patient_id", "strategy"], as_index=False)[metrics].mean()
    openloop["model_seed"] = "mean_across_3_seeds"
    tables = {
        "baseline": {"paired_tests": baseline_pairwise_statistics(
            ["loss", "final_C_rel", "tumor_regression", "tumor_auc_rel", "mean_activity", "total_dose", "loss_late_activity"],
            paired_test="paired_t")},
        "dose_weight": build_dose_weight_tables(),
        "objective": {
            "seed2031_patients": patients,
            "seed2031_means_ratios": mean_ratios,
        },
        "observation": {
            "patient_seed_means": observation_means,
            "paired_tests": formal_observation_patient_statistics(observation_means),
        },
        "openloop": {"patient_seed_means": openloop,
                     "paired_tests": openloop_pairwise_statistics(openloop, metrics, p_adjust="holm")},
        "locked": build_locked_plot_inputs(),
        "attribution": build_attribution_tables(),
    }
    tables['observation']['full_practical_tests'] = build_full_practical_tests(observation_means)
    tables['openloop'].update(build_openloop_plot_inputs(openloop))
    # Match the typed tabular values used by pandas CSV readers; do not round values.
    return {group: {name: pd.read_csv(StringIO(frame.to_csv(index=False)))
                    for name, frame in sheets.items()} for group, sheets in tables.items()}


def write_tables(tables, output_dir: Path) -> None:
    """Export computed tables with the public openpyxl dependency."""
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "generator": "data_preparation/prepare_source_tables.py",
        "inputs": {"baseline": str(BASELINE_DIR), "dose_sweep": str(PARETO_DIR),
                   "objective_ablation": str(PILOT_DIR), "observation_history": str(MULTISEED_DIR),
                   "openloop": str(OPENLOOP_DIR), "benchmark": str(LOCKED_BENCHMARK_DIR),
                   "attribution": str(INPUT_ATTRIBUTION_DIR)},
        "runtime": {"python": sys.version.split()[0], "pandas": pd.__version__, "numpy": np.__version__},
        "files": TABLE_FILES, "tables": {},
    }
    for group, sheets in tables.items():
        manifest["tables"][group] = {}
        for name, frame in sheets.items():
            manifest["tables"][group][name] = {
                "rows": len(frame), "columns": list(frame.columns),
                "dtypes": {column: str(dtype) for column, dtype in frame.dtypes.items()},
            }
        destination = output_dir / TABLE_FILES[group]
        if group == "baseline":
            sheets["paired_tests"].to_csv(destination, index=False)
            continue
        with pd.ExcelWriter(destination, engine="openpyxl") as writer:
            for name, frame in sheets.items():
                frame.to_excel(writer, sheet_name=name, index=False)
                sheet = writer.sheets[name]
                sheet.freeze_panes = "A2"
                sheet.auto_filter.ref = sheet.dimensions
                for cell in sheet[1]:
                    cell.fill = PatternFill("solid", fgColor="24475B")
                    cell.font = Font(bold=True, color="FFFFFF")
                    cell.alignment = Alignment(wrap_text=True)
                sheet.row_dimensions[1].height = 48
                for index, column in enumerate(frame.columns, 1):
                    width = max(len(str(column)), *(len(str(v)) for v in frame[column].head(200)))
                    sheet.column_dimensions[get_column_letter(index)].width = min(44, max(18, width + 2))
    (output_dir / "source_tables_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    global BASELINE_DIR, PARETO_DIR, PILOT_DIR, MULTISEED_DIR
    global OPENLOOP_DIR, LOCKED_BENCHMARK_DIR, INPUT_ATTRIBUTION_DIR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=SOURCE_DATA_DIR)
    parser.add_argument("--baseline-run", type=Path, default=BASELINE_DIR)
    parser.add_argument("--dose-sweep-dir", type=Path, default=PARETO_DIR)
    parser.add_argument("--objective-ablation-dir", type=Path, default=PILOT_DIR)
    parser.add_argument("--observation-history-dir", type=Path, default=MULTISEED_DIR)
    parser.add_argument("--openloop-dir", type=Path, default=OPENLOOP_DIR)
    parser.add_argument("--benchmark-dir", type=Path, default=LOCKED_BENCHMARK_DIR)
    parser.add_argument("--attribution-dir", type=Path, default=INPUT_ATTRIBUTION_DIR)
    args = parser.parse_args()
    BASELINE_DIR = args.baseline_run.resolve()
    PARETO_DIR = args.dose_sweep_dir.resolve()
    PILOT_DIR = args.objective_ablation_dir.resolve()
    MULTISEED_DIR = args.observation_history_dir.resolve()
    OPENLOOP_DIR = args.openloop_dir.resolve()
    LOCKED_BENCHMARK_DIR = args.benchmark_dir.resolve()
    INPUT_ATTRIBUTION_DIR = args.attribution_dir.resolve()
    tables = build_all_tables()
    write_tables(tables, args.output_dir)
    print(f"Source tables: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
