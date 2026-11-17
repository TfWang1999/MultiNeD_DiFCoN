"""Frozen Top50 selection, GMT parsing and within-family BH adjustment."""
from __future__ import annotations
import numpy as np
import pandas as pd
from common.paths import HUMAN_ROOT
GENE_SET_DIR = HUMAN_ROOT / 'resources/gene_set_libraries'
TOP_N = 50
LIBRARIES = {'GO Biological Process': 'GO_Biological_Process_2025', 'Reactome': 'Reactome_Pathways_2024', 'MSigDB Hallmark': 'MSigDB_Hallmark_2020'}

def bh_fdr(p_values: pd.Series) -> pd.Series:
    """Benjamini-Hochberg adjustment over finite P values only."""
    result = pd.Series(np.nan, index=p_values.index, dtype=float)
    finite = np.isfinite(p_values.to_numpy(dtype=float))
    if not finite.any():
        return result
    values = p_values.loc[finite].to_numpy(dtype=float)
    order = np.argsort(values, kind='mergesort')
    ranked = values[order] * len(values) / np.arange(1, len(values) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted = np.empty_like(ranked)
    adjusted[order] = np.clip(ranked, 0.0, 1.0)
    result.loc[p_values.loc[finite].index] = adjusted
    return result

def validate_weights(weights_path) -> tuple[pd.DataFrame, pd.DataFrame]:
    weights = pd.read_csv(weights_path)
    required = {'human_ensembl_gene', 'human_symbol', 'cd8score_gene_weight', 'abs_model_weight_rank', 'variable_expression_all_tumors'}
    missing = sorted(required - set(weights.columns))
    if missing:
        raise ValueError(f'Frozen gene-weight table lacks columns: {missing}')
    if len(weights) != 230 or weights['human_ensembl_gene'].nunique() != 230:
        raise ValueError('Expected exactly 230 unique genes in the existing CD8score decomposition')
    ranked = weights.sort_values(['abs_model_weight_rank', 'human_ensembl_gene'], kind='mergesort').reset_index(drop=True)
    computed = weights.assign(_abs_weight=weights['cd8score_gene_weight'].abs()).sort_values(['_abs_weight', 'human_ensembl_gene'], ascending=[False, True], kind='mergesort')['human_ensembl_gene'].tolist()
    if computed != ranked['human_ensembl_gene'].tolist():
        raise ValueError('Saved abs_model_weight_rank does not match abs(cd8score_gene_weight)')
    if not np.array_equal(ranked['abs_model_weight_rank'].to_numpy(dtype=int), np.arange(1, len(ranked) + 1)):
        raise ValueError('abs_model_weight_rank must be a unique 1..230 ordering')
    background = ranked[ranked['variable_expression_all_tumors'].astype(bool)].copy()
    if len(background) != 226:
        raise ValueError(f'Expected 226 available mapped genes; found {len(background)}')
    if background['human_symbol'].isna().any() or background['human_symbol'].duplicated().any():
        raise ValueError('The 226-gene enrichment background must have unique non-missing symbols')
    top50 = ranked.iloc[:TOP_N].copy()
    if len(top50) != TOP_N or not top50['variable_expression_all_tumors'].astype(bool).all():
        raise ValueError('Top50 must contain 50 variable genes from the enrichment background')
    top50['weight_direction'] = np.where(top50['cd8score_gene_weight'] > 0, 'positive', 'negative')
    if (top50['cd8score_gene_weight'] == 0).any():
        raise ValueError('A Top50 frozen CD8score weight is zero')
    return (background, top50)

def load_library(library_name: str) -> dict[str, set[str]]:
    path = GENE_SET_DIR / f'{library_name}.gmt'
    if not path.is_file():
        raise FileNotFoundError(f'Missing frozen GMT library: {path}; restore the recorded version')
    gene_sets: dict[str, set[str]] = {}
    for line_number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), start=1):
        parts = line.rstrip('\n').split('\t')
        if len(parts) < 3 or not parts[0].strip():
            raise ValueError(f'Malformed {library_name} line {line_number}')
        term = parts[0].strip()
        genes = {gene.strip().upper() for gene in parts[2:] if gene.strip()}
        if not genes:
            raise ValueError(f'Empty gene set in {library_name}: {term}')
        if term in gene_sets:
            gene_sets[term].update(genes)
        else:
            gene_sets[term] = genes
    return gene_sets
