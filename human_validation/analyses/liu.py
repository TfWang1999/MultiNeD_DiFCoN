"""Frozen-program validation in pretreatment Liu et al. 2019 melanoma tumors.

Reads fixed official TPM/clinical inputs and computes ortholog mapping,
program scores and continuous CD8score PFS/OS Cox results.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from common.paths import HUMAN_ROOT, REPO_ROOT, HCOP_PATH
from common import PROGRAM_ORDER, build_ortholog_mapping, calculate_scores, cox_models, load_alpha, prepare_weights, zscore_rows
RAW_DIR = HUMAN_ROOT / 'Liu2019/inputs/raw'
WEIGHTS_PATH = REPO_ROOT / 'data/mouse/program_weights/Fig1e_program_top_genes.csv'
RAW_FILENAMES = {'official_rnaseq_tpm': '41591_2019_654_MOESM3_ESM.txt', 'official_supplementary_tables': '41591_2019_654_MOESM4_ESM.xlsx'}
BIOPSY_CONTEXT_COLUMN = 'biopsyContext (1=Pre-Ipi; 2=On-Ipi; 3=Pre-PD1; 4=On-PD1)'
PRETREATMENT_CONTEXT_CODE = 3

def input_paths() -> dict[str, Path]:
    paths = {key: RAW_DIR / filename for key, filename in RAW_FILENAMES.items()}
    paths['hcop_mouse'] = HCOP_PATH
    missing = [str(p) for p in paths.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError(f'Restore the recorded input files: {missing}')
    return paths


def load_clinical(path: Path) -> tuple[pd.DataFrame, int]:
    frame = pd.read_excel(path, sheet_name='Supplemental Table 1', header=2, index_col=0)
    workbook_rows = len(frame)
    frame.index = frame.index.astype(str).str.strip()
    column_names = {
        'PFS': 'pfs_time_days',
        'OS': 'os_time_days',
        'progressed': 'pfs_event',
        'dead': 'os_event',
        BIOPSY_CONTEXT_COLUMN: 'biopsy_context_code',
        'daysBiopsyToPD1': 'days_biopsy_to_pd1',
    }
    frame = frame.loc[frame.index.str.fullmatch('Patient\\d+'), list(column_names)]
    frame.index.name = 'patient_id'
    frame = frame.rename(columns=column_names).apply(pd.to_numeric, errors='coerce')
    return frame.reset_index(), workbook_rows


def load_expression(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep='\t', index_col=0)
    frame.index = frame.index.astype(str).str.strip()
    frame.columns = frame.columns.astype(str).str.strip()
    if frame.index.duplicated().any() or frame.columns.duplicated().any():
        raise ValueError('Official TPM matrix contains duplicate sample or gene identifiers')
    frame = frame.apply(pd.to_numeric, errors='coerce')
    return frame


def build_entrez_expression(expression_samples_by_symbol: pd.DataFrame, sample_ids: list[str], weights: pd.DataFrame) -> pd.DataFrame:
    """Translate official gene symbols to HCOP Entrez IDs before shared mapping."""
    hcop = pd.read_csv(HCOP_PATH, sep='\t', compression='gzip', dtype=str, low_memory=False)
    candidates = hcop[hcop['mouse_symbol'].isin(weights['mouse_symbol']) & hcop['human_symbol'].isin(expression_samples_by_symbol.columns) & hcop['human_entrez_gene'].notna() & hcop['human_entrez_gene'].ne('-')][['human_symbol', 'human_entrez_gene']].drop_duplicates()
    symbol_ambiguity = candidates.groupby('human_symbol')['human_entrez_gene'].nunique()
    entrez_ambiguity = candidates.groupby('human_entrez_gene')['human_symbol'].nunique()
    if (symbol_ambiguity > 1).any() or (entrez_ambiguity > 1).any():
        raise ValueError('Target HCOP symbol-to-Entrez bridge is not one-to-one')
    bridge = candidates.sort_values(['human_symbol', 'human_entrez_gene']).reset_index(drop=True)
    expression = expression_samples_by_symbol.loc[sample_ids, bridge['human_symbol']].T
    symbol_to_entrez = bridge.set_index('human_symbol')['human_entrez_gene']
    expression.index = symbol_to_entrez.loc[expression.index].to_numpy()
    expression.index.name = 'human_entrez_gene'
    if expression.index.duplicated().any():
        raise ValueError('Entrez translation created duplicate expression rows')
    return expression


def check_input(check: str, passed: bool, detail: object) -> None:
    if not passed:
        raise AssertionError(f'{check}: {detail}')


def run(inputs, output) -> None:
    ALPHA_PATH = inputs.dynamics / 'robustness/dynamic_importance_by_delta.csv'
    OUTPUT_DIR = output / 'outputs'
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = input_paths()
    clinical, workbook_rows = load_clinical(paths['official_supplementary_tables'])
    expression_all = load_expression(paths['official_rnaseq_tpm'])
    check_input('supplementary_table_patient_rows', len(clinical) == 144, f'workbook data rows={workbook_rows}')
    check_input('official_expression_dimensions', expression_all.shape == (121, 20848), 'samples x genes')
    check_input('expression_values_finite', np.isfinite(expression_all.to_numpy()).all(), 'official TPM matrix')
    check_input('expression_values_nonnegative', bool((expression_all.to_numpy() >= 0).all()), 'TPM')
    check_input('all_expression_samples_match_clinical', set(expression_all.index).issubset(set(clinical['patient_id'])), 'patient ID exact match')
    expression_metadata = expression_all.index.to_frame(index=False, name='patient_id').merge(clinical, on='patient_id', how='left', validate='one_to_one')
    context_counts = expression_metadata['biopsy_context_code'].value_counts(dropna=False).sort_index().to_dict()
    metadata = expression_metadata[expression_metadata['biopsy_context_code'].eq(PRETREATMENT_CONTEXT_CODE)].copy()
    check_input('pretreatment_pre_pd1_sample_count', len(metadata) == 104, context_counts)
    check_input('one_pretreatment_sample_per_patient', metadata['patient_id'].is_unique, 'one expression row per patient')
    check_input('only_pre_pd1_context_retained', metadata['biopsy_context_code'].eq(3).all(), 'official code 3=Pre-PD1')
    check_input('pretreatment_days_nonpositive', metadata['days_biopsy_to_pd1'].le(0).all(), 'biopsy day relative to PD1 start')
    check_input('pfs_complete_and_binary', metadata[['pfs_time_days', 'pfs_event']].notna().all().all() and set(metadata['pfs_event'].unique()).issubset({0, 1}), 'PFS')
    check_input('os_complete_and_binary', metadata[['os_time_days', 'os_event']].notna().all().all() and set(metadata['os_event'].unique()).issubset({0, 1}), 'OS')
    sample_ids = metadata['patient_id'].tolist()
    weights = prepare_weights(WEIGHTS_PATH)
    expression = build_entrez_expression(expression_all, sample_ids, weights)
    log_expression = np.log2(expression + 1.0)
    expression_z, _ = zscore_rows(log_expression)
    mapping = build_ortholog_mapping(weights, HCOP_PATH, set(expression_z.index), human_id_column='human_entrez_gene')
    _, standardized_scores = calculate_scores(expression_z, mapping, sample_ids, human_id_column='human_entrez_gene')
    alpha = load_alpha(ALPHA_PATH, treatment='Anti-PD1')
    patient_scores = metadata.set_index('patient_id')[['pfs_time_days', 'pfs_event', 'os_time_days', 'os_event']].copy()
    patient_scores['CD8score_PD1'] = standardized_scores[PROGRAM_ORDER].mul(alpha, axis=1).sum(axis=1)
    cox_tables = []
    for endpoint, time_column, event_column in [('PFS', 'pfs_time_days', 'pfs_event'), ('OS', 'os_time_days', 'os_event')]:
        cox = cox_models(patient_scores, ['CD8score_PD1'], time_column=time_column, event_column=event_column)
        cox.insert(0, 'endpoint', endpoint)
        cox_tables.append(cox)
    pd.concat(cox_tables, ignore_index=True).to_csv(OUTPUT_DIR / 'cox_continuous_results.csv', index=False)
    print(f'Saved PFS/OS Cox to {OUTPUT_DIR}')

if __name__ == '__main__':
    raise SystemExit('Use human_validation/run_validation.py with --output under runs/')
