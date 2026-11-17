"""Nested-CV supervised Ridge Cox benchmarks for frozen CD8score genes.

The Cox fitting, fold-only standardization, inner-CV lambda selection, and
Harrell C-index calculations are imported from the existing five-program
benchmark. Feature sets are fixed before survival analysis using only the
absolute frozen CD8score gene weight.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from common.ridge import OUTER_FOLDS, harrell_c_index, ridge_oof
from common.cohort_inputs import cohort_frame
COHORTS = ['All tumors', 'Metastatic only']
FEATURE_SET_SIZES: dict[str, int | None] = {'All230': None, 'Top10': 10, 'Top20': 20, 'Top50': 50}

def load_inputs(inputs) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    EXPRESSION_PATH = inputs.results / 'TCGA_SKCM/outputs/tcga_skcm_program_gene_expression_z.csv.gz'
    PATIENT_SCORES_PATH = inputs.results / 'TCGA_SKCM/outputs/tcga_skcm_patient_scores.csv'
    REFERENCE_BENCHMARK_PATH = inputs.results / 'TCGA_SKCM/five_program_cox/benchmark_cindex_results.csv'
    REFERENCE_OOF_PATH = inputs.results / 'TCGA_SKCM/five_program_cox/ridge_oof_scores.csv'
    WEIGHTS_PATH = inputs.results / 'TCGA_SKCM/gene_level_cd8score/cd8score_gene_weights.csv'
    weights = pd.read_csv(WEIGHTS_PATH)
    expression = pd.read_csv(EXPRESSION_PATH, index_col=0)
    patients = pd.read_csv(PATIENT_SCORES_PATH)
    reference_oof = pd.read_csv(REFERENCE_OOF_PATH)
    reference_benchmark = pd.read_csv(REFERENCE_BENCHMARK_PATH)
    required_weight_columns = {'human_ensembl_gene', 'human_symbol', 'cd8score_gene_weight', 'abs_model_weight_rank', 'variable_expression_all_tumors'}
    missing = sorted(required_weight_columns - set(weights.columns))
    if missing:
        raise ValueError(f'Gene-weight table lacks required columns: {missing}')
    if len(weights) != 230 or weights['human_ensembl_gene'].nunique() != 230:
        raise ValueError('Expected exactly 230 unique CD8score decomposition genes')
    genes = weights['human_ensembl_gene'].astype(str).tolist()
    if set(genes) != set(expression.index.astype(str)):
        raise ValueError('Frozen expression genes do not exactly match the 230-gene decomposition')
    if patients['sample_id'].duplicated().any():
        raise ValueError('Patient-score input contains duplicate sample_id values')
    if set(patients['sample_id']) != set(expression.columns.astype(str)):
        raise ValueError('Frozen expression and patient-score sample sets differ')
    return (weights, expression, patients, reference_oof, reference_benchmark)


def define_feature_sets(weights: pd.DataFrame, expression: pd.DataFrame) -> tuple[dict[str, list[str]], pd.DataFrame]:
    ranked = weights.sort_values(['abs_model_weight_rank', 'human_ensembl_gene'], kind='mergesort').reset_index(drop=True)
    expected_ranks = np.arange(1, len(ranked) + 1)
    if not np.array_equal(ranked['abs_model_weight_rank'].to_numpy(dtype=int), expected_ranks):
        raise ValueError('abs_model_weight_rank is not a unique 1..230 ordering')
    computed_order = weights.assign(_abs=weights['cd8score_gene_weight'].abs()).sort_values(['_abs', 'human_ensembl_gene'], ascending=[False, True], kind='mergesort')['human_ensembl_gene'].tolist()
    if computed_order != ranked['human_ensembl_gene'].tolist():
        raise ValueError('Saved abs_model_weight_rank does not match abs(cd8score_gene_weight)')
    expression_sd = expression.std(axis=1, ddof=0)
    variable_from_matrix = expression_sd.gt(0) & np.isfinite(expression_sd)
    saved_variable = ranked.set_index('human_ensembl_gene')['variable_expression_all_tumors'].astype(bool)
    matrix_variable = variable_from_matrix.reindex(saved_variable.index)
    if not matrix_variable.equals(saved_variable):
        disagreements = saved_variable.index[saved_variable.ne(matrix_variable)].tolist()
        raise ValueError(f'Saved and measured frozen-expression variance disagree: {disagreements}')
    predictor_sets: dict[str, list[str]] = {}
    rows: list[dict[str, object]] = []
    for feature_set, size in FEATURE_SET_SIZES.items():
        selected = ranked if size is None else ranked.iloc[:size]
        selected_genes = selected['human_ensembl_gene'].tolist()
        predictors = [gene for gene in selected_genes if bool(variable_from_matrix.loc[gene])]
        predictor_sets[feature_set] = predictors
        for feature_rank, row in enumerate(selected.itertuples(index=False), start=1):
            gene = row.human_ensembl_gene
            included = gene in predictors
            rows.append({'feature_set': feature_set, 'feature_set_rank': feature_rank, 'requested_gene_count': len(selected_genes), 'actual_predictor_count': len(predictors), 'human_ensembl_gene': gene, 'human_symbol': row.human_symbol, 'cd8score_gene_weight': row.cd8score_gene_weight, 'abs_cd8score_gene_weight': abs(row.cd8score_gene_weight), 'abs_model_weight_rank': row.abs_model_weight_rank, 'frozen_all_tumor_expression_sd_ddof0': float(expression_sd.loc[gene]), 'frozen_zero_variance': not bool(variable_from_matrix.loc[gene]), 'included_as_cox_predictor': included, 'exclusion_reason': '' if included else 'frozen all-tumor zero variance', 'feature_set_definition': 'all 230 CD8score decomposition genes' if feature_set == 'All230' else f'top {size} by abs(cd8score_gene_weight), fixed before survival analysis'})
    if len(predictor_sets['All230']) != 226:
        raise ValueError('All230 must contain exactly 226 estimable predictors')
    for feature_set, expected in [('Top10', 10), ('Top20', 20), ('Top50', 50)]:
        if len(predictor_sets[feature_set]) != expected:
            raise ValueError(f'{feature_set} unexpectedly contains a zero-variance predictor')
    return (predictor_sets, pd.DataFrame(rows))


def build_analysis_frame(expression: pd.DataFrame, patients: pd.DataFrame) -> pd.DataFrame:
    expression_t = expression.T.copy()
    expression_t.index = expression_t.index.astype(str)
    expression_t.index.name = 'sample_id'
    expression_t = expression_t.reset_index()
    frame = patients.merge(expression_t, on='sample_id', how='inner', validate='one_to_one')
    if len(frame) != len(patients):
        raise ValueError('Failed to join every TCGA patient to frozen gene-Z expression')
    gene_values = frame[expression.index.astype(str).tolist()].to_numpy(dtype=float)
    if not np.isfinite(gene_values).all():
        raise ValueError('Frozen gene-Z matrix contains non-finite values')
    return frame



def reference_outer_splits(frame: pd.DataFrame, cohort: str, reference_oof: pd.DataFrame) -> tuple[list[tuple[np.ndarray, np.ndarray]], pd.DataFrame]:
    eligible = frame[['sample_id', 'patient_id', 'os_time_days', 'os_event']].dropna().copy()
    eligible = eligible[eligible['os_time_days'] > 0].reset_index(drop=True)
    reference = reference_oof[reference_oof['cohort'].eq(cohort)][['sample_id', 'outer_fold']].copy()
    if reference['sample_id'].duplicated().any():
        raise ValueError(f'Reference OOF contains duplicate samples in {cohort}')
    mapped = eligible.merge(reference, on='sample_id', how='left', validate='one_to_one')
    if mapped['outer_fold'].isna().any() or len(mapped) != len(reference):
        raise ValueError(f'Gene Ridge and five-program OOF patient sets differ in {cohort}')
    folds = mapped['outer_fold'].astype(int).to_numpy()
    splits = []
    all_indices = np.arange(len(mapped))
    for fold in range(1, OUTER_FOLDS + 1):
        heldout = all_indices[folds == fold]
        training = all_indices[folds != fold]
        if not len(heldout):
            raise ValueError(f'Empty reference outer fold {fold} in {cohort}')
        splits.append((training, heldout))
    return (splits, mapped)


def annotate_coefficients(coefficients: pd.DataFrame, feature_set: str, weights: pd.DataFrame) -> pd.DataFrame:
    table = coefficients.rename(columns={'program': 'human_ensembl_gene', 'coefficient_for_training_standardized_program': 'coefficient_for_training_standardized_gene', 'coefficient_per_training_risk_sd': 'coefficient_per_training_risk_sd', 'training_program_mean': 'training_gene_mean', 'training_program_sd_ddof0': 'training_gene_sd_ddof0'}).copy()
    table.insert(1, 'feature_set', feature_set)
    provenance = weights[['human_ensembl_gene', 'human_symbol', 'cd8score_gene_weight', 'abs_model_weight_rank']]
    table = table.merge(provenance, on='human_ensembl_gene', how='left', validate='many_to_one')
    beta = table['coefficient_for_training_standardized_gene'].to_numpy(dtype=float)
    table['coefficient_sign'] = np.where(beta > 0, 'positive', np.where(beta < 0, 'negative', 'zero'))
    ordered = ['cohort', 'feature_set', 'outer_fold', 'human_ensembl_gene', 'human_symbol', 'cd8score_gene_weight', 'abs_model_weight_rank', 'ridge_penalty_lambda', 'coefficient_for_training_standardized_gene', 'coefficient_per_training_risk_sd', 'coefficient_sign']
    return table[ordered + [column for column in table.columns if column not in ordered]]


def make_benchmark(reference_benchmark: pd.DataFrame, gene_scores: dict[tuple[str, str], pd.DataFrame], fold_summaries: dict[tuple[str, str], dict[str, float | int]]) -> pd.DataFrame:
    baseline = reference_benchmark.copy()
    baseline['feature_set'] = ''
    baseline['model_category'] = np.where(baseline['predictor'].eq('Five-program Ridge Cox OOF'), 'TCGA-outcome-trained supervised benchmark', 'fixed prespecified score')
    baseline['uses_tcga_survival_for_fitting'] = baseline['predictor'].eq('Five-program Ridge Cox OOF')
    rows = [baseline]
    for cohort in COHORTS:
        for feature_set in FEATURE_SET_SIZES:
            scores = gene_scores[cohort, feature_set]
            time = scores['os_time_days'].to_numpy(dtype=float)
            event = scores['os_event'].astype(int).to_numpy()
            risk = scores['ridge_oof_risk_score'].to_numpy(dtype=float)
            stats = harrell_c_index(time, event, risk)
            fold_summary = fold_summaries[cohort, feature_set]
            rows.append(pd.DataFrame([{'cohort': cohort, 'predictor': f'{feature_set} gene Ridge Cox OOF', 'n': len(scores), 'events': int(event.sum()), 'risk_score_definition': 'nested-CV held-out supervised gene Ridge Cox linear predictor; higher value is higher risk', **stats, 'outer_fold_weighted_c_index': fold_summary['outer_fold_weighted_c_index'], 'outer_fold_comparable_pairs': fold_summary['outer_fold_comparable_pairs'], 'feature_set': feature_set, 'model_category': 'TCGA-outcome-trained supervised benchmark', 'uses_tcga_survival_for_fitting': True}]))
    benchmark = pd.concat(rows, ignore_index=True)
    cd8score = benchmark[benchmark['predictor'].eq('CD8score_Control')].set_index('cohort')['harrell_c_index']
    benchmark['delta_c_index_vs_frozen_cd8score_control'] = benchmark.apply(lambda row: row['harrell_c_index'] - cd8score.loc[row['cohort']], axis=1)
    predictor_order = {'CD8score_Control': 1, 'Five-program Ridge Cox OOF': 2, 'All230 gene Ridge Cox OOF': 3, 'Top10 gene Ridge Cox OOF': 4, 'Top20 gene Ridge Cox OOF': 5, 'Top50 gene Ridge Cox OOF': 6}
    cohort_order = {cohort: index for index, cohort in enumerate(COHORTS)}
    benchmark['_cohort_order'] = benchmark['cohort'].map(cohort_order)
    benchmark['_predictor_order'] = benchmark['predictor'].map(predictor_order)
    return benchmark.sort_values(['_cohort_order', '_predictor_order']).drop(columns=['_cohort_order', '_predictor_order']).reset_index(drop=True)


def make_oof_table(patients: pd.DataFrame, reference_oof: pd.DataFrame, gene_scores: dict[tuple[str, str], pd.DataFrame]) -> pd.DataFrame:
    fixed = patients[['sample_id', 'CD8score_Control']]
    rows = []
    for cohort in COHORTS:
        reference = reference_oof[reference_oof['cohort'].eq(cohort)][['cohort', 'sample_id', 'patient_id', 'os_time_days', 'os_event', 'outer_fold', 'ridge_oof_linear_predictor_raw', 'ridge_oof_risk_score']].rename(columns={'ridge_oof_linear_predictor_raw': 'five_program_ridge_oof_linear_predictor_raw', 'ridge_oof_risk_score': 'five_program_ridge_oof_risk_score'})
        combined = reference.merge(fixed, on='sample_id', how='left', validate='one_to_one')
        for feature_set in FEATURE_SET_SIZES:
            scores = gene_scores[cohort, feature_set][['sample_id', 'outer_fold', 'ridge_oof_linear_predictor_raw', 'ridge_oof_risk_score']].rename(columns={'outer_fold': f'{feature_set}_outer_fold', 'ridge_oof_linear_predictor_raw': f'{feature_set}_ridge_oof_linear_predictor_raw', 'ridge_oof_risk_score': f'{feature_set}_ridge_oof_risk_score'})
            combined = combined.merge(scores, on='sample_id', how='inner', validate='one_to_one')
            if not np.array_equal(combined['outer_fold'].to_numpy(dtype=int), combined[f'{feature_set}_outer_fold'].to_numpy(dtype=int)):
                raise ValueError(f'Outer-fold assignments differ for {cohort}, {feature_set}')
            combined = combined.drop(columns=f'{feature_set}_outer_fold')
        rows.append(combined)
    output = pd.concat(rows, ignore_index=True)
    if output['CD8score_Control'].isna().any():
        raise ValueError('The fixed CD8score_Control score is missing from the OOF table')
    return output


def run(inputs, output) -> None:
    HERE = output
    weights, expression, patients, reference_oof, reference_benchmark = load_inputs(inputs)
    predictor_sets, feature_sets = define_feature_sets(weights, expression)
    analysis = build_analysis_frame(expression, patients)
    gene_scores: dict[tuple[str, str], pd.DataFrame] = {}
    fold_summaries: dict[tuple[str, str], dict[str, float | int]] = {}
    coefficient_tables = []
    tuning_tables = []
    for cohort in COHORTS:
        frame = cohort_frame(analysis, cohort)
        splits, mapped = reference_outer_splits(frame, cohort, reference_oof)
        for feature_set, predictors in predictor_sets.items():
            print(f'Running {cohort} / {feature_set}: {len(predictors)} predictors, N={len(mapped)}', flush=True)
            scores, coefficients, tuning, fold_summary = ridge_oof(frame, cohort, feature_columns=predictors, outer_splits=splits)
            if scores['sample_id'].tolist() != mapped['sample_id'].tolist():
                raise ValueError(f'OOF sample order differs from reference in {cohort}, {feature_set}')
            if not np.array_equal(scores['outer_fold'].to_numpy(dtype=int), mapped['outer_fold'].to_numpy(dtype=int)):
                raise ValueError(f'OOF fold assignment differs in {cohort}, {feature_set}')
            gene_scores[cohort, feature_set] = scores
            fold_summaries[cohort, feature_set] = fold_summary
            coefficient_tables.append(annotate_coefficients(coefficients, feature_set, weights))
            tuning.insert(1, 'feature_set', feature_set)
            tuning.insert(3, 'actual_predictor_count', len(predictors))
            tuning_tables.append(tuning)
    fold_coefficients = pd.concat(coefficient_tables, ignore_index=True)
    penalty_tuning = pd.concat(tuning_tables, ignore_index=True)
    benchmark = make_benchmark(reference_benchmark, gene_scores, fold_summaries)
    oof_predictions = make_oof_table(patients, reference_oof, gene_scores)
    for name, table in [('feature_sets', feature_sets), ('fold_coefficients', fold_coefficients), ('penalty_tuning', penalty_tuning), ('benchmark_cindex_results', benchmark), ('oof_predictions', oof_predictions)]:
        table.to_csv(HERE / f'{name}.csv', index=False)

if __name__ == '__main__':
    raise SystemExit('Use human_validation/run_validation.py with --output under runs/')
