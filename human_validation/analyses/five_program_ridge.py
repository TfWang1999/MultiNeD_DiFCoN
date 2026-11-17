"""Five-program nested-CV Ridge benchmark using frozen TCGA program scores.
Exports OOF predictions, fold coefficients, penalty tuning and C-index benchmarks."""
from __future__ import annotations
import numpy as np
import pandas as pd
from common.ridge import harrell_c_index, ridge_oof
COHORTS = ['All tumors', 'Metastatic only']

def benchmark_cindex(cohort_frame: pd.DataFrame, ridge_scores: pd.DataFrame, cohort: str, fold_summary: dict[str, float | int]) -> pd.DataFrame:
    fixed = cohort_frame[['sample_id', 'os_time_days', 'os_event', 'CD8score_Control']].dropna().copy()
    fixed = fixed[fixed['os_time_days'] > 0]
    comparison = fixed.merge(ridge_scores[['sample_id', 'ridge_oof_risk_score']], how='inner', on='sample_id', validate='one_to_one')
    if len(comparison) != len(fixed) or len(comparison) != len(ridge_scores):
        raise AssertionError(f'Benchmark predictors do not share the same patient set in {cohort}')
    time = comparison['os_time_days'].to_numpy(dtype=float)
    event = comparison['os_event'].astype(int).to_numpy()
    definitions = [
        ('CD8score_Control', -comparison['CD8score_Control'].to_numpy(), '-CD8score_Control; higher fixed marker is prespecified as protective'),
        ('Five-program Ridge Cox OOF', comparison['ridge_oof_risk_score'].to_numpy(), 'nested-CV held-out linear predictor; higher value is higher risk'),
    ]
    rows = []
    for predictor, risk, definition in definitions:
        stats = harrell_c_index(time, event, risk)
        rows.append({'cohort': cohort, 'predictor': predictor, 'n': len(comparison), 'events': int(event.sum()), 'risk_score_definition': definition, **stats, 'outer_fold_weighted_c_index': fold_summary['outer_fold_weighted_c_index'] if predictor == 'Five-program Ridge Cox OOF' else np.nan, 'outer_fold_comparable_pairs': fold_summary['outer_fold_comparable_pairs'] if predictor == 'Five-program Ridge Cox OOF' else np.nan})
    return pd.DataFrame(rows)


def run(inputs, output) -> None:
    HERE = output
    PATIENT_SCORES_PATH = inputs.results / 'TCGA_SKCM/outputs/tcga_skcm_patient_scores.csv'
    scores = pd.read_csv(PATIENT_SCORES_PATH)
    saved = {name: [] for name in ['ridge_oof_scores', 'ridge_cox_fold_weights', 'ridge_penalty_tuning', 'benchmark_cindex_results']}
    for cohort in COHORTS:
        frame = scores if cohort == 'All tumors' else scores[scores['sample_type'].eq('Metastatic')]
        oof, weights, tuning, summary = ridge_oof(frame, cohort)
        for name, table in zip(saved, [oof, weights, tuning, benchmark_cindex(frame, oof, cohort, summary)]):
            saved[name].append(table)
    for name, tables in saved.items():
        pd.concat(tables, ignore_index=True).to_csv(HERE / f'{name}.csv', index=False)

if __name__ == '__main__':
    raise SystemExit('Use human_validation/run_validation.py with --output under runs/')
