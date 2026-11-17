from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from common.annotation import bh_fdr, LIBRARIES
def fisher_ora(
    background_symbols: set[str],
    query_symbols: set[str],
    libraries: dict[str, dict[str, set[str]]],
) -> pd.DataFrame:
    if not query_symbols.issubset(background_symbols):
        raise ValueError("All CD8score query genes must be contained in the detectable background")
    background_size = len(background_symbols)
    query_size = len(query_symbols)
    rows: list[dict[str, object]] = []
    for library, gene_sets in libraries.items():
        for term, term_genes in gene_sets.items():
            term_background = term_genes & background_symbols
            term_size = len(term_background)
            if term_size == 0:
                continue
            overlap = sorted(query_symbols & term_background)
            overlap_count = len(overlap)
            b = query_size - overlap_count
            c = term_size - overlap_count
            d = background_size - overlap_count - b - c
            if min(overlap_count, b, c, d) < 0:
                raise ValueError(f"Invalid Fisher table for {library}: {term}")
            odds_ratio, p_value = fisher_exact(
                [[overlap_count, b], [c, d]], alternative="greater"
            )
            expected_overlap = query_size * term_size / background_size
            fold_enrichment = overlap_count / expected_overlap if expected_overlap > 0 else np.nan
            rows.append(
                {
                    "selection": "All variable-expression mapped CD8score genes",
                    "library": library,
                    "enrichr_library_name": LIBRARIES[library],
                    "term": term,
                    "background_definition": (
                        "unique GENCODE-v36 symbols with finite, nonzero-variance "
                        "TCGA-SKCM log2(FPKM+1) expression across 469 selected tumors"
                    ),
                    "background_size": background_size,
                    "query_gene_count": query_size,
                    "term_background_gene_count": term_size,
                    "overlap_count": overlap_count,
                    "gene_ratio": overlap_count / query_size,
                    "background_ratio": term_size / background_size,
                    "expected_overlap": expected_overlap,
                    "fold_enrichment": fold_enrichment,
                    "fisher_odds_ratio": float(odds_ratio),
                    "p_value": float(p_value),
                    "overlap_genes": ";".join(overlap),
                    "test": "one-sided Fisher exact over-representation test",
                }
            )
    results = pd.DataFrame(rows)
    if results.empty:
        raise ValueError("No pathway terms overlap the detectable TCGA background")
    results["bh_fdr"] = np.nan
    for _, indices in results.groupby("library", sort=False).groups.items():
        results.loc[indices, "bh_fdr"] = bh_fdr(results.loc[indices, "p_value"])
    results["significant_bh_fdr_0_05"] = results["bh_fdr"] <= 0.05
    library_order = {name: index for index, name in enumerate(LIBRARIES)}
    results["_library_order"] = results["library"].map(library_order)
    results = results.sort_values(
        ["_library_order", "bh_fdr", "p_value", "term"], kind="mergesort"
    ).drop(columns="_library_order")
    return results.reset_index(drop=True)
