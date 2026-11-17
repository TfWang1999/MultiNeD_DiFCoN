"""Fisher ORA for frozen positive- and negative-weight CD8score genes.

The script consumes the already-saved 226-gene all-CD8score query and the already-
saved 56,296-symbol TCGA-SKCM detectable background.  It does not reconstruct
the background or rerun the existing all-226 enrichment.
"""
from __future__ import annotations
import pandas as pd
from common.enrichment import fisher_ora
from common.annotation import LIBRARIES, load_library

def load_fixed_inputs(inputs) -> tuple[pd.DataFrame, pd.DataFrame]:
    folder = inputs.results / 'TCGA_SKCM/gene_level_cd8score/all_cd8score_enrichment'
    query = pd.read_csv(folder / 'all_cd8score_query_genes.csv')
    background = pd.read_csv(folder / 'tcga_detectable_gene_background.csv')
    return query, background


def run(inputs, output) -> None:
    query, background = load_fixed_inputs(inputs)
    background_symbols = set(background['gene_symbol_upper'])
    weights = query['cd8score_gene_weight']
    libraries = {name: load_library(filename) for name, filename in LIBRARIES.items()}
    tables = {}
    for direction, mask in (('positive', weights.gt(0)), ('negative', weights.lt(0))):
        selected_symbols = set(query.loc[mask, 'gene_symbol_upper'])
        if not selected_symbols:
            raise ValueError(f'{direction} query is empty')
        results = fisher_ora(background_symbols, selected_symbols, libraries)
        results['selection'] = f'Full {direction}-weight CD8score genes'
        results.insert(1, 'weight_direction', direction)
        tables[direction] = results
    for direction, results in tables.items():
        results.to_csv(output / f'{direction}_weight_enrichment_results.csv', index=False)
        print(f'Completed {direction}-weight ORA: query={results["query_gene_count"].iloc[0]}, background={len(background_symbols)}', flush=True)

if __name__ == '__main__':
    raise SystemExit('Use human_validation/run_validation.py with --output under runs/')
