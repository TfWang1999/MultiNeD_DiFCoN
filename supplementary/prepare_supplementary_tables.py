"""Prepare the 25 data tables in manuscript workbooks S1--S5 from saved results.

Run from any directory. This script reads CSV results, never trains or plots, and
exports CSV tables without changing the supplied manuscript workbooks.
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SUPP = ROOT / "supplementary"
TRAINING = ROOT / "DiFCoN"
EXPERIMENTS = TRAINING / "outputs"
MODEL_OUTPUTS = ROOT / "MultiNeD/outputs"
HUMAN_SOURCE = SUPP / "figure/human_validation/source_data"
HUMAN_DISPLAY_TABLES = (
    ("S3_01_ScoreB_Cox", "Human_Score_B_TCGA_score_Cox_comparison_displayed.csv", "TCGA score Cox comparisons", 12),
    ("S3_02_ScoreC_Cindex", "Human_Score_C_TCGA_Cindex_benchmarks_displayed.csv", "TCGA discrimination benchmarks", 12),
    ("S3_03_ScoreD_TopK", "Human_Score_D_TCGA_fixed_TopK_displayed.csv", "Fixed-coefficient CD8score approximations", 8),
    ("S3_04_FunctionA_Enrichment", "Human_Function_A_CD8score_enrichment_matrix_displayed.csv", "CD8score functional enrichment", 39),
    ("S3_05_FunctionB_Top50", "Human_Function_B_Top50_function_annotation_displayed.csv", "Top50 functional memberships", 400),
    ("S3_06_ScoreE_Trajectories", "Human_Score_E_GSE272993_patient_trajectories_displayed.csv", "Patient trajectories and cohort summaries", 20),
)
SEEDS = (2031, 2032, 2033)


def read_csv(path):
    return pd.read_csv(path, float_precision="round_trip")


def human_result_tables(results):
    """Build Table S3 display data from one selected human-validation run."""
    from supplementary.supp_plotting.human import (
        FUNCTION_TERMS, MAIN_THEMES, build_top50_annotations,
        longitudinal_display_table, select_enrichment_terms,
    )
    gene = results / "TCGA_SKCM/gene_level_cd8score"
    ridge = gene / "gene_ridge_cox"
    fixed = gene / "fixed_topk"
    enrichment = gene / "all_cd8score_enrichment"
    longitudinal = results / "GSE272993_longitudinal/results/cd8score_longitudinal_validation"
    cox = ridge / "six_score_cox.csv"
    benchmark = ridge / "benchmark_cindex_results.csv"
    correlation, survival = fixed / "topk_fixed_correlations.csv", fixed / "topk_fixed_cindex_results.csv"
    topk = read_csv(correlation).merge(read_csv(survival), on=["cohort", "score", "score_column"],
                                     suffixes=("_correlation", "_survival"), validate="one_to_one")

    enrichment_paths = {
        "All CD8score": enrichment / "all_cd8score_enrichment_results.csv",
        "Positive weights": enrichment / "positive_weight_enrichment_results.csv",
        "Negative weights": enrichment / "negative_weight_enrichment_results.csv",
    }
    enrichment_tables = {name: read_csv(path) for name, path in enrichment_paths.items()}
    display = select_enrichment_terms(enrichment_tables, FUNCTION_TERMS)
    top50 = gene / "top_gene_enrichment/top50_gene_sets.csv"
    annotations = build_top50_annotations(read_csv(top50), enrichment_tables["All CD8score"], MAIN_THEMES)
    patients = longitudinal / "human_patient_delta_cd8score.csv"
    summary = longitudinal / "human_delta_cd8score_summary.csv"
    trajectories = longitudinal_display_table(read_csv(patients), read_csv(summary))
    return {
        "S3_01_ScoreB_Cox": (read_csv(cox), [cox]),
        "S3_02_ScoreC_Cindex": (read_csv(benchmark), [benchmark]),
        "S3_03_ScoreD_TopK": (topk, [correlation, survival]),
        "S3_04_FunctionA_Enrichment": (display, list(enrichment_paths.values())),
        "S3_05_FunctionB_Top50": (annotations, [top50, enrichment_paths["All CD8score"]]),
        "S3_06_ScoreE_Trajectories": (trajectories, [patients, summary]),
    }


def prepare_tables(args):
    """Return (worksheet, data, provenance) records, without writing files."""
    tables = []

    def add(sheet, frame, sources, description, operation, expected_rows):
        if len(frame) != expected_rows:
            raise ValueError(f"{sheet}: expected {expected_rows} rows, got {len(frame)}")
        metadata = {
            "supplementary_table": sheet[:2],
            "workbook": f"{sheet[:2]}/Supplementary_Data_{sheet[:2]}.xlsx",
            "worksheet": sheet,
            "description": description,
            "source_files": " | ".join(dict.fromkeys(Path(os.path.relpath(p.resolve(), ROOT)).as_posix() for p in sources)),
            "preparation": operation,
            "rows": len(frame),
            "columns": len(frame.columns),
            "csv_file": f"{sheet}.csv",
        }
        tables.append((sheet, frame.reset_index(drop=True), metadata))

    primary = args.cell_loto_dir
    full_data = args.cell_reconstruction_dir
    losses = args.loss_ablation_dir
    for sheet, folder, filename, title, rows in (
        ("S1_01_ST1A", primary, "dynamics_method_summary.csv", "Seven-method held-out summary", 7),
        ("S1_02_ST1B", primary, "paired_method_comparisons_holm.csv", "Fold-paired comparisons with saved Holm adjustment", 18),
    ):
        source = folder / filename
        add(sheet, read_csv(source), [source], title, "Retain saved columns, rows and values", rows)

    source = primary / "paired_target_metrics.csv"
    ids = ["seed", "split_id", "target_group"]
    targets = read_csv(source)
    long = targets.melt(id_vars=ids, var_name="metric_method", value_name="value")
    long[["metric", "method"]] = long.pop("metric_method").str.split("__", n=1, expand=True)
    add("S1_03_ST1C", long[ids + ["metric", "method", "value"]], [source],
        "Fold/seed records for seven methods and four metrics",
        "Wide-to-long reshape; retain source column and row order", 6 * 7 * 4 * targets["seed"].nunique())

    for sheet, folder, filename, title, rows in (
        ("S2_01_ST2A", full_data, "dynamics_method_summary.csv", "Full-data six-method reconstruction", 6),
        ("S2_02_ST2B", full_data, "paired_method_comparisons_holm.csv", "Full-data paired comparisons", 15),
        ("S2_03_ST2C", losses, "transition_distribution_summary.csv", "Transition objective ablations", 10),
        ("S2_04_ST2D", losses, "rollout_distribution_summary.csv", "Rollout objective ablations", 5),
        ("S2_05_ST2E", losses, "volume_summary.csv", "Volume objective ablations", 5),
    ):
        source = folder / filename
        add(sheet, read_csv(source), [source], title, "Retain saved columns, rows and values", rows)
    source = args.model_source_dir / "balanced_loto_predictions.csv"
    predictions = read_csv(source).drop(columns=["source_run", "source_fold"])
    add("S2_07_CV_predictions", predictions, [source], "Tumour held-out predictions",
        "Retain the 14 manuscript data columns; omit source_run and source_fold", 48)

    source = args.pretrained_comparison_dir / "reconstruction_summary.csv"
    frame = read_csv(source)[["treatment", "model", "debiased_transport_cost",
                              "energy_distance", "program_mean_mae", "wasserstein_1"]]
    frame["treatment"] = frame["treatment"].replace({
        "Anti-PD1": "anti-PD-1", "Anti-41BB": "anti-4-1BB", "Anti-PD1+Anti-41BB": "Combination"})
    detail_source = args.pretrained_comparison_dir / "distribution_metrics.csv"
    evaluation_seeds = ", ".join(map(str, sorted(read_csv(detail_source)["seed"].unique())))
    add("S1_04_Reconstruction", frame, [source, detail_source], "Pretrained drift versus complete MultiNeD",
        f"Evaluation seeds: {evaluation_seeds}; equal time weights within treatment and equal treatment weights overall", 10)

    selected_human = human_result_tables(args.human_results_dir) if args.human_results_dir is not None else None
    for sheet, filename, title, rows in HUMAN_DISPLAY_TABLES:
        if selected_human is None:
            sources = [HUMAN_SOURCE / filename]
            frame = read_csv(sources[0])
        else:
            frame, sources = selected_human[sheet]
        add(sheet, frame, sources, title, "Retain saved columns and values", rows)

    human_root = args.human_results_dir or ROOT / "human_validation"
    source = human_root / "Liu2019/outputs/cox_continuous_results.csv"
    add("S3_07_AntiPD1_Cox", read_csv(source), [source], "Anti-PD-1 cohort PFS and OS Cox estimates",
         "Retain the saved CD8score_PD1 estimates", 2)

    prepare_policy_tables(add, args.objective_ablation_dir, args.observation_history_dir, args.dose_sweep_dir)

    locked = args.benchmark_dir / "combined"
    for sheet, filename, title, rows in (
        ("S5_01_ST5A", "strategy_summary.csv", "Ten locked strategies: patient mean and SD", 10),
        ("S5_02_ST5B", "paired_primary_comparisons_holm.csv", "CNN-referenced paired comparisons with saved Holm adjustment", 36),
        ("S5_03_ST5E", "patient_seed_mean_metrics.csv", "Seed-averaged patient records for ten strategies", 320),
    ):
        source = locked / filename
        add(sheet, read_csv(source), [source], title, "Retain saved columns, rows and values", rows)
    return tables


def prepare_policy_tables(add, objective_ablation, observation_history, dose_sweep):
    """Keep the recorded objective reference and use the selected dose sweep for Full/H=5."""
    reference_file = objective_ablation / "BASELINE_REFERENCES.tsv"
    references = pd.read_csv(reference_file, sep="\t")
    reference = references.loc[references["lambda_d"].eq(1.0), "existing_200epoch_reference"]
    if len(reference) != 1:
        raise ValueError("Expected one Full-objective reference at lambda_d=1")
    objective_runs = {
        "full": TRAINING / reference.iloc[0],
        **{name: objective_ablation / f"{name}_seed_2031"
           for name in ("no_final", "no_auc", "no_functional", "no_efficiency")},
    }
    full_state = {seed: dose_sweep / f"dose_weight_1p0_seed_{seed}" for seed in SEEDS}

    cache = {}
    patient_ids = None

    def policy(run):
        nonlocal patient_ids
        run = run.resolve()
        if run not in cache:
            source = run / "tables" / "test_strategy_summary.csv"
            frame = read_csv(source)
            frame = frame.loc[frame["strategy"].eq("Optimal")].copy()
            if len(frame) != 32 or frame["patient_id"].nunique() != 32:
                raise ValueError(f"Expected 32 unique test patients in {run}")
            ids = set(frame["patient_id"])
            if patient_ids is None:
                patient_ids = ids
            elif ids != patient_ids:
                raise ValueError(f"Test-patient cohort differs in {run}")
            cache[run] = (frame, source)
        return cache[run]

    loss_metrics = ["final_C_rel", "tumor_auc_rel", "loss_functional_activity_loss", "total_dose"]
    objective = [
        (setting, 2031, objective_runs[name])
        for setting, name in (("Full objective", "full"), ("No final-tumour term", "no_final"),
                              ("No functional term", "no_functional"))
    ]
    observation = [
        (setting, seed, full_state[seed] if prefix is None else observation_history / f"{prefix}_seed_{seed}")
        for setting, prefix in (("Full state", None), ("Practical state", "practical_observation"),
                                ("Schedule only", "schedule_only"))
        for seed in SEEDS
    ]
    for sheet, family, runs, metrics, sources, description in (
        ("S4_01_ST4A", "loss", objective, loss_metrics, [reference_file], "Seed-2031 objective comparisons"),
        ("S4_02_ST4B", "observation", observation, ["final_C_rel", "tumor_auc_rel", "total_dose"], [],
         "Three-seed observation comparisons"),
    ):
        rows = []
        for setting, seed, run in runs:
            data, source = policy(run)
            row = dict(family=family, setting=setting, model_seed=seed, patient_count=len(data))
            row.update(data[metrics].mean().to_dict())
            rows.append(row)
            sources.append(source)
        add(sheet, pd.DataFrame(rows), sources, description,
            "Optimal test-patient means per setting and seed; retain manuscript setting/seed order", len(runs))

    rows, sources = [], []
    history_metrics = ["loss", "final_C_rel", "tumor_auc_rel", "total_dose"]
    for history in (3, 4, 5):
        for seed in SEEDS:
            run = full_state[seed] if history == 5 else observation_history / f"history_length_{history}_seed_{seed}"
            data, source = policy(run)
            row = dict(history_length=history, model_seed=seed)
            row.update(data[history_metrics].mean().to_dict())
            rows.append(row)
            sources.append(source)
    add("S4_03_ST4C", pd.DataFrame(rows), sources, "Retrained H=3/4/5 comparison",
        "Optimal test-patient means per history length and seed", 9)

    frames, sources = [], [reference_file]
    for variant, name in (("full", "full"), ("wC0", "no_final"), ("wCTB0", "no_auc"),
                            ("wF0", "no_functional"), ("wE0", "no_efficiency")):
        data, source = policy(objective_runs[name])
        frame = data[["patient_id"] + loss_metrics].copy()
        frame["variant"] = variant
        frames.append(frame)
        sources.append(source)
    patient = pd.concat(frames, ignore_index=True).rename(columns=dict(zip(loss_metrics, ["C_relative", "CTB", "L_F", "D_total"])))
    add("S4_04_ST4D", patient, sources, "Seed-2031 patient-level objective ablations",
        "Full plus four objective ablations; retain saved patient order and values", 160)
    means = patient.groupby("variant", sort=False)[["C_relative", "CTB", "L_F", "D_total"]].mean()
    ratios = means.div(means.loc["full"], axis=1).add_suffix("_ratio_to_full")
    add("S4_05_ST4E", means.join(ratios).reset_index(), sources, "Seed-2031 means and Full-normalized ratios",
        "Mean of 32 patients per setting; divide each mean by the corresponding Full mean", 5)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=SUPP / "supplementary_tables")
    parser.add_argument("--cell-loto-dir", type=Path, help="Combined cellular LOTO results",
                        default=MODEL_OUTPUTS / "cell_loto_benchmark/b1_shared_loto_all7_combined")
    parser.add_argument("--cell-reconstruction-dir", type=Path, help="Combined reconstruction results",
                        default=MODEL_OUTPUTS / "cell_reconstruction_benchmark/b1_full_data_combined")
    parser.add_argument("--loss-ablation-dir", type=Path, help="Combined model loss-ablation results",
                        default=MODEL_OUTPUTS / "training_loss_ablation/full_data_loss_all_seeds_combined")
    parser.add_argument("--model-source-dir", type=Path, default=ROOT / "MultiNeD/figure_work/source_data")
    parser.add_argument("--benchmark-dir", type=Path, help="Policy benchmark study containing combined/",
                        default=EXPERIMENTS / "locked_test_benchmark")
    parser.add_argument("--objective-ablation-dir", type=Path, default=EXPERIMENTS / "objective_ablation")
    parser.add_argument("--observation-history-dir", type=Path, default=EXPERIMENTS / "observation_history_ablation")
    parser.add_argument("--dose-sweep-dir", type=Path, default=EXPERIMENTS / "dose_weight_sweep")
    parser.add_argument("--pretrained-comparison-dir", type=Path, default=MODEL_OUTPUTS / "pretrained_vs_MultiNeD")
    parser.add_argument("--human-results-dir", type=Path, help="One complete run from human_validation/run_validation.py")
    parser.add_argument("--human-display-dir", type=Path, help="Also save the six displayed S3 source tables here")
    args = parser.parse_args()
    if args.human_display_dir is not None and args.human_results_dir is None:
        parser.error("--human-display-dir requires --human-results-dir")
    return args


def main():
    args = parse_args()
    tables = prepare_tables(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for sheet, frame, _ in tables:
        frame.to_csv(args.output_dir / f"{sheet}.csv", index=False)
    pd.DataFrame([metadata for _, _, metadata in tables]).to_csv(
        args.output_dir / "supplementary_tables_manifest.csv", index=False)
    if args.human_display_dir is not None:
        args.human_display_dir.mkdir(parents=True, exist_ok=True)
        by_sheet = {sheet: frame for sheet, frame, _ in tables}
        for sheet, filename, _, _ in HUMAN_DISPLAY_TABLES:
            by_sheet[sheet].to_csv(args.human_display_dir / filename, index=False)
    print(f"Wrote {len(tables)} data CSVs and one source manifest to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
