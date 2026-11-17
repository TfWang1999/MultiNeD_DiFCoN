"""Fetch public inputs for human validation, or check local copies."""
from __future__ import annotations
import argparse
from pathlib import Path
import shutil
import urllib.request

# Input paths are relative to the repository root, independent of the working directory.
ROOT = Path(__file__).resolve().parents[1]
INPUTS = [
    {
        "path": "human_validation/TCGA_SKCM/inputs/raw/TCGA-SKCM.star_fpkm.tsv.gz",
        "url": "https://gdc-hub.s3.us-east-1.amazonaws.com/download/TCGA-SKCM.star_fpkm.tsv.gz",
        "group": "tcga",
        "bytes": 117599955,
    },
    {
        "path": "human_validation/TCGA_SKCM/inputs/raw/TCGA-SKCM.star_fpkm.tsv.json",
        "url": "https://gdc-hub.s3.us-east-1.amazonaws.com/download/TCGA-SKCM.star_fpkm.tsv.json",
        "group": "tcga",
        "bytes": 815,
    },
    {
        "path": "human_validation/TCGA_SKCM/inputs/raw/TCGA-SKCM.survival.tsv.gz",
        "url": "https://gdc-hub.s3.us-east-1.amazonaws.com/download/TCGA-SKCM.survival.tsv.gz",
        "group": "tcga",
        "bytes": 4281,
    },
    {
        "path": "human_validation/TCGA_SKCM/inputs/raw/TCGA-SKCM.survival.tsv.json",
        "url": "https://gdc-hub.s3.us-east-1.amazonaws.com/download/TCGA-SKCM.survival.tsv.json",
        "group": "tcga",
        "bytes": 346,
    },
    {
        "path": "human_validation/resources/human_mouse_hcop_fifteen_column.txt.gz",
        "url": "https://storage.googleapis.com/public-download-files/hcop/human_mouse_hcop_fifteen_column.txt.gz",
        "group": "resources",
        "bytes": 3454647,
    },
    {
        "path": "human_validation/resources/gene_set_libraries/GO_Biological_Process_2025.gmt",
        "url": "https://maayanlab.cloud/Enrichr/geneSetLibrary?mode=text&libraryName=GO_Biological_Process_2025",
        "group": "resources",
        "bytes": 1355235,
    },
    {
        "path": "human_validation/resources/gene_set_libraries/Reactome_Pathways_2024.gmt",
        "url": "https://maayanlab.cloud/Enrichr/geneSetLibrary?mode=text&libraryName=Reactome_Pathways_2024",
        "group": "resources",
        "bytes": 887130,
    },
    {
        "path": "human_validation/resources/gene_set_libraries/MSigDB_Hallmark_2020.gmt",
        "url": "https://maayanlab.cloud/Enrichr/geneSetLibrary?mode=text&libraryName=MSigDB_Hallmark_2020",
        "group": "resources",
        "bytes": 44235,
    },
    {
        "path": "human_validation/Liu2019/inputs/raw/41591_2019_654_MOESM3_ESM.txt",
        "url": "https://static-content.springer.com/esm/art%3A10.1038%2Fs41591-019-0654-5/MediaObjects/41591_2019_654_MOESM3_ESM.txt",
        "group": "liu",
        "bytes": 35614392,
    },
    {
        "path": "human_validation/Liu2019/inputs/raw/41591_2019_654_MOESM4_ESM.xlsx",
        "url": "https://static-content.springer.com/esm/art%3A10.1038%2Fs41591-019-0654-5/MediaObjects/41591_2019_654_MOESM4_ESM.xlsx",
        "group": "liu",
        "bytes": 5382313,
    },
    {
        "path": "human_validation/GSE272993_longitudinal/data/raw/GSE272993_cd8_nn_labeled_FINAL.RDS.gz",
        "url": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE272nnn/GSE272993/suppl/GSE272993_cd8_nn_labeled_FINAL.RDS.gz",
        "group": "longitudinal",
        "bytes": 2097240284,
    },
    {
        "path": "human_validation/TCGA_SKCM/gene_level_cd8score/all_cd8score_enrichment/reference/gencode.v36.annotation.gtf.gene.probemap",
        "url": "https://gdc-hub.s3.us-east-1.amazonaws.com/download/gencode.v36.annotation.gtf.gene.probemap",
        "group": "tcga",
        "bytes": 3206945,
    },
]


def check_exists(path):
    if not path.is_file():
        raise FileNotFoundError(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group', choices=['resources', 'tcga', 'liu', 'longitudinal', 'all'], default='resources')
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    if args.list:
        for item in INPUTS:
            print(f"{item['group']:12} {item['bytes']/2**20:8.2f} MiB  {item['path']}\n  {item['url']}")
        return
    for item in INPUTS:
        if args.group != 'all' and item['group'] not in {args.group, 'resources'}:
            continue
        path = (ROOT / item['path']).resolve()
        if not path.is_relative_to(ROOT):
            raise ValueError('Input path escapes repository')
        if not args.check and not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            partial = path.with_name(path.name + '.part')
            print(f"Downloading {item['path']} ({item['bytes']/2**20:.1f} MiB)", flush=True)
            request = urllib.request.Request(item['url'], headers={'User-Agent':'MultiNeD-DiFCoN/0.1'})
            with urllib.request.urlopen(request, timeout=120) as response, partial.open('xb') as out:
                shutil.copyfileobj(response, out, length=1024*1024)
            check_exists(partial)
            partial.rename(path)
        check_exists(path)
        print('AVAILABLE', item['path'], flush=True)


if __name__ == '__main__':
    main()
