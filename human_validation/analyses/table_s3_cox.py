"""Reconstruct Table S3's six-score Cox estimates from saved predictions.

Only 12 univariable Efron Cox models are fitted. The five supervised OOF
predictions, their folds, CD8score coefficients and all paper tables stay frozen.
The supervised Wald intervals condition on saved predictions and do not
include uncertainty from model training or fold assignment.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from common.validation_utils import cox_models
SCORES = {'CD8score_Control': 'CD8score_Control', 'Five-program Ridge Cox OOF': 'five_program_ridge_oof_risk_score', 'All230 gene Ridge Cox OOF': 'All230_ridge_oof_risk_score', 'Top10 gene Ridge Cox OOF': 'Top10_ridge_oof_risk_score', 'Top20 gene Ridge Cox OOF': 'Top20_ridge_oof_risk_score', 'Top50 gene Ridge Cox OOF': 'Top50_ridge_oof_risk_score'}

def calculate(inputs) -> pd.DataFrame:
    OOF_PATH = inputs.results / 'TCGA_SKCM/gene_level_cd8score/gene_ridge_cox/oof_predictions.csv'
    PATIENT_PATH = inputs.results / 'TCGA_SKCM/outputs/tcga_skcm_patient_scores.csv'
    oof = pd.read_csv(OOF_PATH)
    patients = pd.read_csv(PATIENT_PATH)
    if oof.duplicated(['cohort', 'patient_id']).any() or patients['sample_id'].duplicated().any():
        raise ValueError('Expected one sample per patient within each cohort')
    required = ['sample_id', 'os_time_days', 'os_event', 'CD8score_Control']
    check = oof.merge(patients[required], on='sample_id', validate='many_to_one', how='left', suffixes=('', '_patient'))
    for col in required[1:]:
        if not np.allclose(check[col], check[col + '_patient'], atol=1e-12, rtol=1e-12):
            raise ValueError(f'Saved OOF and patient source disagree: {col}')
    results = []
    for cohort, (n, events) in {'All tumors': (455, 220), 'Metastatic only': (359, 194)}.items():
        frame = oof.loc[oof['cohort'].eq(cohort)].copy()
        if len(frame) != n or int(frame['os_event'].sum()) != events:
            raise ValueError(f'Unexpected survival population: {cohort}')
        values = frame[['os_time_days', 'os_event', *SCORES.values()]].to_numpy(float)
        if not np.isfinite(values).all() or not frame['os_time_days'].gt(0).all():
            raise ValueError(f'Invalid saved Cox input: {cohort}')
        for score, source in SCORES.items():
            frozen = score == 'CD8score_Control'
            data = frame[['os_time_days', 'os_event']].copy()
            data[score] = frame[source] if frozen else -frame[source]
            if data[score].std(ddof=0) <= 0:
                raise ValueError(f'Constant predictor: {score}')
            fit = cox_models(data, [score], time_column='os_time_days', event_column='os_event')
            fit.insert(0, 'cohort', cohort)
            fit['score_source_column'] = source
            fit['score_direction'] = 'unchanged frozen CD8score; higher is favourable' if frozen else 'negative saved OOF risk; higher is favourable'
            fit['calculation'] = 'univariable Efron Cox on saved scores; no Ridge refit'
            fit['inference_scope'] = 'fixed-score Wald inference' if frozen else 'nominal Wald inference conditional on saved predictions; no training/split uncertainty'
            results.append(fit)
    return pd.concat(results, ignore_index=True)


def run(inputs, output):
    calculate(inputs).to_csv(output / 'six_score_cox.csv', index=False)
