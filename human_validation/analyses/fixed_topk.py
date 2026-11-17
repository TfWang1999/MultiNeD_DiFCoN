"""Evaluate frozen-weight TopK CD8score approximations in TCGA-SKCM.

Top10, Top20, and Top50 are selected only by the existing absolute frozen
CD8score gene weight. No coefficient is trained or changed in this script.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from common.cohort_inputs import cohort_frame
import common.bootstrap as fixed_score_bootstrap
COHORTS = ['All tumors', 'Metastatic only']
TOP_K = [10, 20, 50]
SCORE_ORDER = ['Full CD8score', 'Top10 fixed', 'Top20 fixed', 'Top50 fixed']
SCORE_COLUMNS = {'Full CD8score': 'Full_CD8score', 'Top10 fixed': 'Top10_fixed', 'Top20 fixed': 'Top20_fixed', 'Top50 fixed': 'Top50_fixed'}
RISK_COLUMNS = {score: f'risk_{SCORE_COLUMNS[score]}' for score in SCORE_ORDER}
RISK_DEFINITION = 'negative of the fixed score; higher CD8score is prespecified as protective; direction never flipped from observed survival'

def load_inputs(inputs) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tcga = inputs.results / 'TCGA_SKCM'
    weights = pd.read_csv(tcga / 'gene_level_cd8score/cd8score_gene_weights.csv')
    expression = pd.read_csv(tcga / 'outputs/tcga_skcm_program_gene_expression_z.csv.gz', index_col=0)
    patients = pd.read_csv(tcga / 'outputs/tcga_skcm_patient_scores.csv')
    reference_oof = pd.read_csv(tcga / 'five_program_cox/ridge_oof_scores.csv')
    return weights, expression, patients, reference_oof


def select_genes(weights: pd.DataFrame) -> dict[int, list[str]]:
    ranked = weights.sort_values(['abs_model_weight_rank', 'human_ensembl_gene'], kind='mergesort').reset_index(drop=True)
    gene_sets: dict[int, list[str]] = {}
    for k in TOP_K:
        selected = ranked.iloc[:k].copy()
        gene_sets[k] = selected['human_ensembl_gene'].astype(str).tolist()
    return gene_sets


def calculate_fixed_scores(weights: pd.DataFrame, expression: pd.DataFrame, patients: pd.DataFrame, gene_sets: dict[int, list[str]]) -> pd.DataFrame:
    expression.columns = expression.columns.astype(str)
    weight_by_gene = weights.set_index('human_ensembl_gene')['cd8score_gene_weight'].astype(float)
    topk = pd.DataFrame(index=expression.columns)
    for k, genes in gene_sets.items():
        topk[f'Top{k}_fixed'] = weight_by_gene.loc[genes].to_numpy(dtype=float) @ expression.loc[genes, expression.columns].to_numpy(dtype=float)
    topk.index.name = 'sample_id'
    topk = topk.reset_index()
    output = patients[['sample_id', 'patient_id', 'sample_type_code', 'sample_type', 'os_time_days', 'os_event', 'CD8score_Control']].merge(topk, on='sample_id', how='left', validate='one_to_one')
    output = output.rename(columns={'CD8score_Control': 'Full_CD8score'})
    return output


def correlation_results(scores: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for cohort in COHORTS:
        data = cohort_frame(scores, cohort)
        full = data['Full_CD8score'].to_numpy(dtype=float)
        for score in SCORE_ORDER:
            values = data[SCORE_COLUMNS[score]].to_numpy(dtype=float)
            pearson = pearsonr(full, values)
            spearman = spearmanr(full, values)
            rows.append({'cohort': cohort, 'score': score, 'score_column': SCORE_COLUMNS[score], 'n': len(data), 'pearson_r_vs_full_cd8score': float(pearson.statistic), 'pearson_p_value': float(pearson.pvalue), 'spearman_rho_vs_full_cd8score': float(spearman.statistic), 'spearman_p_value': float(spearman.pvalue), 'r_squared_linear_vs_full_cd8score': float(pearson.statistic ** 2), 'r_squared_definition': 'squared Pearson correlation; variance in Full CD8score explained after an intercept-and-slope linear calibration'})
    return pd.DataFrame(rows)


def cindex_and_bootstrap(scores: pd.DataFrame, reference_oof: pd.DataFrame) -> pd.DataFrame:
    score_columns = ['sample_id', *SCORE_COLUMNS.values()]
    fixed = scores[score_columns].copy()
    analysis = reference_oof.merge(fixed, on='sample_id', how='left', validate='many_to_one')
    for score in SCORE_ORDER:
        analysis[RISK_COLUMNS[score]] = -analysis[SCORE_COLUMNS[score]]
    summary_rows: list[dict[str, object]] = []
    for cohort_index, cohort in enumerate(COHORTS):
        data = analysis[analysis['cohort'].eq(cohort)].reset_index(drop=True)
        seed = fixed_score_bootstrap.BASE_RANDOM_SEED + cohort_index
        observed, bootstrap = fixed_score_bootstrap.bootstrap_cindices(data, seed, scores=SCORE_ORDER, risk_columns=RISK_COLUMNS)
        event = data['os_event'].astype(int).to_numpy()
        for index, score in enumerate(SCORE_ORDER):
            stats = observed[index]
            mean, standard_error, low, high = fixed_score_bootstrap.percentile_summary(bootstrap[:, index])
            summary_rows.append({'cohort': cohort, 'score': score, 'score_column': SCORE_COLUMNS[score], 'n': len(data), 'events': int(event.sum()), **stats, 'bootstrap_mean': mean, 'bootstrap_standard_error': standard_error, 'bootstrap_ci95_low': low, 'bootstrap_ci95_high': high, 'bootstrap_replicates': fixed_score_bootstrap.BOOTSTRAP_REPLICATES, 'valid_bootstrap_replicates': int(np.isfinite(bootstrap[:, index]).sum()), 'ci_method': 'patient-level percentile bootstrap', 'random_seed': seed, 'risk_score_definition': RISK_DEFINITION, 'weights_frozen': True, 'uses_tcga_survival_for_weight_fitting': False})
    summary = pd.DataFrame(summary_rows)
    score_order = {score: index for index, score in enumerate(SCORE_ORDER)}
    summary['_score_order'] = summary['score'].map(score_order)
    summary = summary.sort_values(['cohort', '_score_order'], kind='mergesort').drop(columns='_score_order')
    return summary.reset_index(drop=True)


def run(inputs, output) -> None:
    CINDEX_OUTPUT = output / 'topk_fixed_cindex_results.csv'
    CORRELATION_OUTPUT = output / 'topk_fixed_correlations.csv'
    HERE = output
    weights, expression, patients, reference_oof = load_inputs(inputs)
    gene_sets = select_genes(weights)
    scores = calculate_fixed_scores(weights, expression, patients, gene_sets)
    correlations = correlation_results(scores)
    cindex = cindex_and_bootstrap(scores, reference_oof)
    correlations.to_csv(CORRELATION_OUTPUT, index=False)
    cindex.to_csv(CINDEX_OUTPUT, index=False)
    print(f'Completed frozen-weight TopK CD8score analysis for {len(scores)} patients; wrote results to {HERE}', flush=True)

if __name__ == '__main__':
    raise SystemExit('Use human_validation/run_validation.py with --output under runs/')
