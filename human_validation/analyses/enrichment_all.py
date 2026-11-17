"""Functional enrichment of all variable-expression mapped CD8score genes.

The query contains the variable-expression CD8score genes, while the
background is the full set of unique, GENCODE-mapped, variable-expression gene
symbols in the existing 469-patient TCGA-SKCM tumor expression matrix.
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from common.paths import HUMAN_ROOT
from common.enrichment import fisher_ora
from common.annotation import LIBRARIES, load_library
EXPRESSION_PATH = HUMAN_ROOT / 'TCGA_SKCM/inputs/raw/TCGA-SKCM.star_fpkm.tsv.gz'
EXPRESSION_METADATA_PATH = HUMAN_ROOT / 'TCGA_SKCM/inputs/raw/TCGA-SKCM.star_fpkm.tsv.json'
PROBEMAP_NAME = 'gencode.v36.annotation.gtf.gene.probemap'
PROBEMAP_PATH = HUMAN_ROOT / 'TCGA_SKCM/gene_level_cd8score/all_cd8score_enrichment/reference/gencode.v36.annotation.gtf.gene.probemap'
EXPECTED_SELECTED_SAMPLES = 469
EXPECTED_QUERY_GENES = 226
EXPRESSION_CHUNK_SIZE = 2048

def parse_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    normalized = series.astype(str).str.strip().str.lower()
    unexpected = sorted(set(normalized) - {'true', 'false'})
    if unexpected:
        raise ValueError(f'Unexpected Boolean values: {unexpected}')
    return normalized.eq('true')


def load_probemap() -> pd.DataFrame:
    probemap = pd.read_csv(PROBEMAP_PATH, sep='\t', dtype=str, usecols=['id', 'gene'])
    probemap['human_ensembl_gene'] = probemap['id'].str.replace('\\.\\d+$', '', regex=True)
    probemap['gene_symbol'] = probemap['gene'].str.strip()
    probemap = probemap[probemap['human_ensembl_gene'].ne('') & probemap['gene_symbol'].ne('')].copy()
    ambiguity = probemap.groupby('human_ensembl_gene')['gene_symbol'].nunique()
    if (ambiguity > 1).any():
        examples = ambiguity[ambiguity > 1].head().to_dict()
        raise ValueError(f'GENCODE probemap has ambiguous versionless Ensembl IDs: {examples}')
    probemap = probemap.drop_duplicates('human_ensembl_gene', keep='first')
    return probemap[['human_ensembl_gene', 'gene_symbol']]


def selected_sample_ids(inputs) -> list[str]:
    patient_scores = inputs.results / 'TCGA_SKCM/outputs/tcga_skcm_patient_scores.csv'
    selected = pd.read_csv(patient_scores, usecols=['sample_id'])['sample_id'].astype(str).tolist()
    if len(selected) != EXPECTED_SELECTED_SAMPLES or len(set(selected)) != len(selected):
        raise ValueError(f'Expected {EXPECTED_SELECTED_SAMPLES} unique selected tumor samples; found {len(selected)}')
    return selected


def build_detectable_background(samples: list[str], probemap: pd.DataFrame) -> pd.DataFrame:
    map_by_ensembl = probemap.set_index('human_ensembl_gene')['gene_symbol']
    records: list[pd.DataFrame] = []
    usecols = ['Ensembl_ID', *samples]
    for chunk in pd.read_csv(EXPRESSION_PATH, sep='\t', compression='gzip', usecols=usecols, chunksize=EXPRESSION_CHUNK_SIZE, low_memory=False):
        ensembl = chunk['Ensembl_ID'].astype(str).str.replace('\\.\\d+$', '', regex=True)
        # The CSV reader retains the original Xena column order.
        values = chunk.drop(columns='Ensembl_ID').apply(pd.to_numeric, errors='coerce').to_numpy(dtype=float)
        finite = np.isfinite(values).all(axis=1)
        variable = finite & (np.ptp(values, axis=1) > 0)
        symbols = ensembl.map(map_by_ensembl)
        keep = variable & symbols.notna().to_numpy() & symbols.astype(str).str.strip().ne('').to_numpy()
        if keep.any():
            records.append(pd.DataFrame({
                'human_ensembl_gene': ensembl.loc[keep].to_numpy(),
                'gene_symbol': symbols.loc[keep].astype(str).str.strip().to_numpy(),
            }))
    if not records:
        raise ValueError('No variable-expression genes were recovered from TCGA-SKCM')
    per_ensembl = pd.concat(records, ignore_index=True)
    if per_ensembl['human_ensembl_gene'].duplicated().any():
        raise ValueError('Versionless Ensembl IDs are duplicated in the TCGA expression matrix')
    per_ensembl['gene_symbol_upper'] = per_ensembl['gene_symbol'].str.upper()
    rows: list[dict[str, object]] = []
    for symbol_upper, group in per_ensembl.groupby('gene_symbol_upper', sort=True):
        ordered = group.sort_values('human_ensembl_gene', kind='mergesort')
        rows.append({
            'gene_symbol': ordered.iloc[0]['gene_symbol'],
            'gene_symbol_upper': symbol_upper,
            'human_ensembl_genes': ';'.join(ordered['human_ensembl_gene']),
            'n_ensembl_ids_for_symbol': len(ordered),
            'background_inclusion_rule': 'GENCODE-v36-mapped unique symbol with finite, nonzero-variance log2(FPKM+1) expression across the 469 selected tumors',
        })
    background = pd.DataFrame(rows).sort_values('gene_symbol_upper', kind='mergesort')
    background = background.reset_index(drop=True)
    return background


def build_query(inputs, background: pd.DataFrame, probemap: pd.DataFrame) -> pd.DataFrame:
    WEIGHTS_PATH = inputs.results / 'TCGA_SKCM/gene_level_cd8score/cd8score_gene_weights.csv'
    weights = pd.read_csv(WEIGHTS_PATH)
    required = {'human_ensembl_gene', 'human_symbol', 'cd8score_gene_weight', 'variable_expression_all_tumors', 'abs_model_weight_rank'}
    missing = sorted(required - set(weights.columns))
    if missing:
        raise ValueError(f'Frozen CD8score weight table lacks columns: {missing}')
    query = weights.loc[parse_bool(weights['variable_expression_all_tumors'])].copy()
    if len(query) != EXPECTED_QUERY_GENES or query['human_ensembl_gene'].nunique() != EXPECTED_QUERY_GENES:
        raise ValueError(f'Expected {EXPECTED_QUERY_GENES} variable-expression CD8score genes')
    query = query.merge(probemap, how='left', on='human_ensembl_gene', validate='one_to_one')
    if query['gene_symbol'].isna().any():
        missing_ids = query.loc[query['gene_symbol'].isna(), 'human_ensembl_gene'].tolist()
        raise ValueError(f'CD8score query genes missing from GENCODE v36 probemap: {missing_ids}')
    query = query.rename(columns={'human_symbol': 'source_human_symbol'})
    query['gene_symbol_upper'] = query['gene_symbol'].str.upper()
    if query['gene_symbol_upper'].duplicated().any():
        duplicated = query.loc[query['gene_symbol_upper'].duplicated(False), ['human_ensembl_gene', 'gene_symbol']]
        raise ValueError(f"CD8score query symbols are not unique: {duplicated.to_dict('records')}")
    background_symbols = set(background['gene_symbol_upper'])
    outside = sorted(set(query['gene_symbol_upper']) - background_symbols)
    if outside:
        raise ValueError(f'CD8score query genes outside the detectable TCGA background: {outside}')
    source_symbols = query['source_human_symbol'].fillna('').astype(str).str.upper()
    remapped = source_symbols.ne(query['gene_symbol_upper'])
    query['symbol_resolved_with_gencode_v36'] = remapped
    query['selected_without_tcga_outcome'] = True
    query['query_definition'] = 'all variable-expression genes in frozen CD8score decomposition'
    query = query.sort_values('abs_model_weight_rank', kind='mergesort').reset_index(drop=True)
    return query


def run(inputs, output) -> None:
    BACKGROUND_PATH = output / 'tcga_detectable_gene_background.csv'
    HERE = output
    QUERY_PATH = output / 'all_cd8score_query_genes.csv'
    RESULTS_PATH = output / 'all_cd8score_enrichment_results.csv'
    PATIENT_SCORES_PATH = inputs.results / 'TCGA_SKCM/outputs/tcga_skcm_patient_scores.csv'
    WEIGHTS_PATH = inputs.results / 'TCGA_SKCM/gene_level_cd8score/cd8score_gene_weights.csv'
    required_inputs = [WEIGHTS_PATH, EXPRESSION_PATH, EXPRESSION_METADATA_PATH, PATIENT_SCORES_PATH]
    missing_inputs = [str(path) for path in required_inputs if not path.is_file()]
    if missing_inputs:
        raise FileNotFoundError(f'Missing required existing inputs: {missing_inputs}')
    expression_metadata = json.loads(EXPRESSION_METADATA_PATH.read_text(encoding='utf-8'))
    if expression_metadata.get('unit', '').lower() != 'log2(fpkm+1)':
        raise ValueError(f"Unexpected TCGA expression unit: {expression_metadata.get('unit')}")
    if expression_metadata.get(':probeMap') != PROBEMAP_NAME:
        raise ValueError(f"Unexpected TCGA expression probe map: {expression_metadata.get(':probeMap')}")
    samples = selected_sample_ids(inputs)
    probemap = load_probemap()
    background = build_detectable_background(samples, probemap)
    query = build_query(inputs, background, probemap)
    background_symbols = set(background['gene_symbol_upper'])
    query_symbols = set(query['gene_symbol_upper'])
    libraries: dict[str, dict[str, set[str]]] = {}
    for display_name, library_name in LIBRARIES.items():
        gene_sets = load_library(library_name)
        libraries[display_name] = gene_sets
    results = fisher_ora(background_symbols, query_symbols, libraries)
    query_output = query[['abs_model_weight_rank', 'human_ensembl_gene', 'source_human_symbol', 'gene_symbol', 'gene_symbol_upper', 'cd8score_gene_weight', 'symbol_resolved_with_gencode_v36', 'selected_without_tcga_outcome', 'query_definition']].copy()
    background.to_csv(BACKGROUND_PATH, index=False)
    query_output.to_csv(QUERY_PATH, index=False)
    results.to_csv(RESULTS_PATH, index=False)
    print(f'Completed all-CD8score enrichment: query={len(query_symbols)}, background={len(background_symbols)}, tests={len(results)}; outputs={HERE}', flush=True)

if __name__ == '__main__':
    raise SystemExit('Use human_validation/run_validation.py with --output under runs/')
