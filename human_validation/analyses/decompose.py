"""Expand frozen TCGA-SKCM CD8score-Control scores to the gene level.

This analysis reuses the existing TCGA-SKCM tumor selection, ortholog mapping,
standardized expression, five-program scoring rules, and frozen Control
delta=0.5 alpha vector.  It does not refit alpha, select genes using survival,
or alter any existing TCGA-SKCM result.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from common import PROGRAM_ORDER, SOURCE_TO_MODEL, load_alpha

def slug(value: str) -> str:
    return value.lower().replace('-', '_').replace(' ', '_')


def absolute_weight_rank(values: pd.Series) -> pd.Series:
    """Rank absolute weights with Ensembl ID as the tie-break."""
    ordered = pd.DataFrame({'value': values.abs(), 'gene': values.index.astype(str)})
    ordered = ordered.sort_values(['value', 'gene'], ascending=[False, True], kind='mergesort')
    rank = pd.Series(np.arange(1, len(ordered) + 1), index=ordered.index, dtype=int)
    return rank.reindex(values.index)


def build_gene_weights(mapping: pd.DataFrame, alpha: pd.Series, raw_program_sd: pd.Series, variable_genes: pd.Series) -> pd.DataFrame:
    """Expand the five-program coefficients into frozen gene weights."""
    used = mapping[mapping['expression_present']].copy()
    source_programs = list(SOURCE_TO_MODEL)
    source_coefficients = used.groupby(['human_ensembl_gene', 'source_program'], sort=True)['weight_used_for_scoring'].sum().unstack(fill_value=0.0).reindex(columns=source_programs, fill_value=0.0)
    program_coefficients = pd.DataFrame(index=source_coefficients.index)
    program_coefficients['Precursor'] = source_coefficients[['Naive-Like', 'Memory-Like', 'Naive-Il7r']].mean(axis=1)
    program_coefficients['Activation'] = source_coefficients['Activation']
    program_coefficients['Early effector'] = source_coefficients['Eff-Early']
    program_coefficients['Cytotoxic'] = source_coefficients['Eff-Cytox']
    program_coefficients['Dysfunction'] = source_coefficients['Dysf']
    program_coefficients = program_coefficients[PROGRAM_ORDER]
    cd8score_contributions = program_coefficients.mul(alpha / raw_program_sd, axis=1)
    gene_weights = pd.DataFrame(index=program_coefficients.index)
    symbols = used.groupby('human_ensembl_gene')['human_symbol'].agg(lambda values: ';'.join(sorted({str(value) for value in values.dropna()})))
    sources = used.groupby('human_ensembl_gene')['source_program'].agg(lambda values: ';'.join(sorted(set(values))))
    models = used.groupby('human_ensembl_gene')['model_program'].agg(lambda values: ';'.join(sorted(set(values))))
    gene_weights['human_symbol'] = symbols
    gene_weights['source_programs'] = sources
    gene_weights['model_programs'] = models
    gene_weights['n_source_programs'] = sources.str.split(';').map(len)
    gene_weights['n_model_programs'] = models.str.split(';').map(len)
    gene_weights['variable_expression_all_tumors'] = variable_genes.reindex(gene_weights.index).astype(bool)
    for source_program in source_programs:
        gene_weights[f'source_coefficient_{slug(source_program)}'] = source_coefficients[source_program]
    for program in PROGRAM_ORDER:
        gene_weights[f'program_raw_coefficient_{slug(program)}'] = program_coefficients[program]
    gene_weights['cd8score_gene_weight'] = cd8score_contributions.sum(axis=1)
    gene_weights['abs_model_weight_rank'] = absolute_weight_rank(gene_weights['cd8score_gene_weight'])
    for program in PROGRAM_ORDER:
        gene_weights[f"weight_{program.replace(' ', '_')}"] = cd8score_contributions[program]
    gene_weights.index.name = 'human_ensembl_gene'
    gene_weights = gene_weights.reset_index().sort_values('abs_model_weight_rank')
    return gene_weights


def run(inputs, output) -> None:
    ALPHA_PATH = inputs.dynamics / 'robustness/dynamic_importance_by_delta.csv'
    tcga = inputs.results / 'TCGA_SKCM/outputs'
    patient_scores = pd.read_csv(tcga / 'tcga_skcm_patient_scores.csv')
    mapping = pd.read_csv(tcga / 'ortholog_mapping.csv')
    mapping['expression_present'] = mapping['expression_present'].astype(bool)
    expression_z = pd.read_csv(tcga / 'tcga_skcm_program_gene_expression_z.csv.gz', index_col=0)
    # zscore_rows stores zero for all samples of a non-variable gene.
    variable_genes = expression_z.loc[:, patient_scores['sample_id']].ne(0).any(axis=1)
    alpha = load_alpha(ALPHA_PATH, treatment='Control')
    raw_column_by_program = {program: f'{program}_raw_weighted_gene_z' for program in PROGRAM_ORDER}
    raw_saved = patient_scores.set_index('sample_id')[list(raw_column_by_program.values())].rename(columns={column: program for program, column in raw_column_by_program.items()})
    raw_program_sd = raw_saved.std(axis=0, ddof=0)
    gene_weights = build_gene_weights(mapping, alpha, raw_program_sd, variable_genes)
    gene_weights.to_csv(output / 'cd8score_gene_weights.csv', index=False, float_format='%.17g')
    print(f'Frozen gene decomposition: {len(gene_weights)} genes')


if __name__ == '__main__':
    raise SystemExit('Use human_validation/run_validation.py with --output under runs/')
