"""TCGA expression inputs and survival cohort selection."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

def candidate_ensembl_ids(weights: pd.DataFrame, hcop_path: Path) -> set[str]:
    columns = ["mouse_symbol", "human_ensembl_gene"]
    hcop = pd.read_csv(hcop_path, sep="\t", compression="gzip", dtype=str, usecols=columns, low_memory=False)
    candidates = hcop.loc[hcop["mouse_symbol"].isin(weights["mouse_symbol"]), "human_ensembl_gene"]
    return set(candidates.dropna()) - {"", "-"}


def load_program_expression(
    path: Path, sample_ids: list[str], candidate_ids: set[str]
) -> pd.DataFrame:
    selected_chunks = []
    usecols = ["Ensembl_ID", *sample_ids]
    for chunk in pd.read_csv(
        path,
        sep="\t",
        compression="gzip",
        usecols=usecols,
        chunksize=2048,
        low_memory=False,
    ):
        chunk["Ensembl_ID"] = chunk["Ensembl_ID"].astype(str).str.replace(r"\.\d+$", "", regex=True)
        chunk = chunk[chunk["Ensembl_ID"].isin(candidate_ids)]
        if not chunk.empty:
            selected_chunks.append(chunk)
    if not selected_chunks:
        raise ValueError("No frozen-program human orthologs were found in the expression matrix")
    expression = pd.concat(selected_chunks, ignore_index=True).set_index("Ensembl_ID")
    expression = expression.apply(pd.to_numeric, errors="coerce")
    if expression.index.duplicated().any():
        expression = expression.groupby(level=0, sort=False).mean()
    if not np.isfinite(expression.to_numpy()).all():
        raise ValueError("Program-gene expression contains missing or non-finite values")
    return expression.loc[:, sample_ids]

def cohort_frame(frame: pd.DataFrame, cohort: str) -> pd.DataFrame:
    if cohort == 'All tumors':
        return frame.copy()
    if cohort == 'Metastatic only':
        return frame[frame['sample_type'].eq('Metastatic')].copy()
    raise ValueError(f'Unknown cohort: {cohort}')
