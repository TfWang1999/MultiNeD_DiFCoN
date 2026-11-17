"""Frozen-program survival validation in TCGA-SKCM tumors.

The analysis reads the saved GDC-derived STAR-FPKM matrix and survival table from
the UCSC Xena GDC hub, applies the existing frozen mouse-program weights and
Control delta=0.5 alpha vector, and runs prespecified survival analyses. No
gene weight, alpha, gene set, or survival cutoff is fitted or optimized.
"""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from common.paths import HUMAN_ROOT, REPO_ROOT, HCOP_PATH
from common.cohort_inputs import candidate_ensembl_ids, load_program_expression
from common import PROGRAM_ORDER, build_ortholog_mapping, calculate_scores, km_analysis, load_alpha, prepare_weights, zscore_rows
RAW_DIR = HUMAN_ROOT / 'TCGA_SKCM/inputs/raw'
WEIGHTS_PATH = REPO_ROOT / 'data/mouse/program_weights/Fig1e_program_top_genes.csv'
RAW_FILENAMES = {'expression': 'TCGA-SKCM.star_fpkm.tsv.gz', 'expression_metadata': 'TCGA-SKCM.star_fpkm.tsv.json', 'survival': 'TCGA-SKCM.survival.tsv.gz', 'survival_metadata': 'TCGA-SKCM.survival.tsv.json'}
TUMOR_TYPE_CODES = {'01': 'Primary Tumor', '06': 'Metastatic', '07': 'Metastatic'}
SAMPLE_PRIORITY = {'06': 0, '07': 1, '01': 2}

def input_paths() -> dict[str, Path]:
    paths = {key: RAW_DIR / filename for key, filename in RAW_FILENAMES.items()}
    paths['hcop_mouse'] = HCOP_PATH
    missing = [str(p) for p in paths.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError(f'Restore the recorded input files: {missing}')
    return paths


def select_samples(expression_path: Path) -> pd.DataFrame:
    columns = pd.read_csv(expression_path, sep='\t', compression='gzip', nrows=0).columns.tolist()
    if not columns or columns[0] != 'Ensembl_ID':
        raise ValueError(f'Unexpected expression-matrix first column: {columns[:1]}')
    samples = pd.DataFrame({'sample_id': columns[1:]})
    samples['patient_id'] = samples['sample_id'].str.split('-').str[:3].str.join('-')
    samples['sample_type_code'] = samples['sample_id'].str.split('-').str[3].str[:2]
    tumors = samples.loc[samples['sample_type_code'].isin(TUMOR_TYPE_CODES)].copy()
    tumors['sample_type'] = tumors['sample_type_code'].map(TUMOR_TYPE_CODES)
    tumors['selection_priority'] = tumors['sample_type_code'].map(SAMPLE_PRIORITY)
    tumors = tumors.sort_values(['patient_id', 'selection_priority', 'sample_id'])
    return tumors.drop_duplicates('patient_id')


def load_patient_survival(path: Path) -> pd.DataFrame:
    survival = pd.read_csv(path, sep='\t', compression='gzip')
    expected = {'sample', 'OS.time', 'OS', '_PATIENT'}
    if not expected.issubset(survival.columns):
        raise ValueError(f'Survival table lacks columns: {sorted(expected - set(survival.columns))}')
    inconsistent = survival.groupby('_PATIENT').agg(time_values=('OS.time', 'nunique'), event_values=('OS', 'nunique'))
    if (inconsistent[['time_values', 'event_values']] > 1).any().any():
        raise ValueError('A patient has inconsistent survival values across sample rows')
    patient = survival.groupby('_PATIENT', as_index=False).agg(os_time_days=('OS.time', 'first'), os_event=('OS', 'first'))
    patient = patient.rename(columns={'_PATIENT': 'patient_id'})
    patient['os_time_days'] = pd.to_numeric(patient['os_time_days'], errors='coerce')
    patient['os_event'] = pd.to_numeric(patient['os_event'], errors='coerce')
    return patient


def check_input(name: str, passed: bool, detail: object) -> None:
    if not passed:
        raise AssertionError(f'{name}: {detail}')


def run(inputs, output) -> None:
    ALPHA_PATH = inputs.dynamics / 'robustness/dynamic_importance_by_delta.csv'
    HERE = output
    OUTPUT_DIR = output / 'outputs'
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = input_paths()
    expression_metadata = json.loads(paths['expression_metadata'].read_text(encoding='utf-8'))
    check_input('expression_unit_is_log2_fpkm_plus_1', expression_metadata.get('unit', '').lower() == 'log2(fpkm+1)', expression_metadata.get('unit'))
    selected = select_samples(paths['expression'])
    selected_samples = selected['sample_id'].tolist()
    check_input('metastatic_subset_present', selected['sample_type'].eq('Metastatic').any(), int(selected['sample_type'].eq('Metastatic').sum()))
    weights = prepare_weights(WEIGHTS_PATH)
    candidates = candidate_ensembl_ids(weights, paths['hcop_mouse'])
    expression_log = load_program_expression(paths['expression'], selected_samples, candidates)
    expression_z, _ = zscore_rows(expression_log)
    mapping = build_ortholog_mapping(weights, paths['hcop_mouse'], set(expression_z.index), human_id_column='human_ensembl_gene')
    mapping.to_csv(OUTPUT_DIR / 'ortholog_mapping.csv', index=False)
    raw_scores, standardized_scores = calculate_scores(expression_z, mapping, selected_samples, human_id_column='human_ensembl_gene')
    alpha = load_alpha(ALPHA_PATH, treatment='Control')
    survival = load_patient_survival(paths['survival'])
    patient_scores = selected[['sample_id', 'patient_id', 'sample_type_code', 'sample_type']].merge(survival, how='left', on='patient_id', validate='one_to_one')
    patient_scores = patient_scores.set_index('sample_id')
    for program in PROGRAM_ORDER:
        patient_scores[f'{program}_raw_weighted_gene_z'] = raw_scores[program]
        patient_scores[program] = standardized_scores[program]
    patient_scores['CD8score_Control'] = standardized_scores[PROGRAM_ORDER].mul(alpha, axis=1).sum(axis=1)
    patient_scores = patient_scores.reset_index()
    patient_scores.to_csv(OUTPUT_DIR / 'tcga_skcm_patient_scores.csv', index=False)
    program_expression = expression_z.loc[:, selected_samples].copy()
    program_expression.insert(0, 'human_ensembl_gene', program_expression.index)
    program_expression.to_csv(OUTPUT_DIR / 'tcga_skcm_program_gene_expression_z.csv.gz', index=False, compression='gzip', float_format='%.7g')
    cohort_frames = {'All tumors': patient_scores.copy(), 'Metastatic only': patient_scores[patient_scores['sample_type'] == 'Metastatic'].copy()}
    km_tables = []
    curve_tables = []
    for cohort, frame in cohort_frames.items():
        km_summary, km_curves = km_analysis(frame)
        km_summary.insert(0, 'cohort', cohort)
        km_curves.insert(0, 'cohort', cohort)
        km_tables.append(km_summary)
        curve_tables.append(km_curves)
    km_results = pd.concat(km_tables, ignore_index=True)
    km_curves = pd.concat(curve_tables, ignore_index=True)
    km_results.to_csv(OUTPUT_DIR / 'km_median_split_results.csv', index=False)
    km_curves.to_csv(OUTPUT_DIR / 'km_curve_source_data.csv', index=False)
    print(f'Wrote TCGA-SKCM validation to {HERE}')

if __name__ == '__main__':
    raise SystemExit('Use human_validation/run_validation.py with --output under runs/')
