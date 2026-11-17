# Human validation

These analyses apply mouse-derived CD8score weights to three human cohorts.
The frozen program weights use Control for TCGA and anti-PD-1 for Liu2019 and
GSE272993. The tables retain the fields `CD8score_Control`, `CD8score_PD1` and
`delta_CD8score`.

| Cohort or analysis | Implementation | Manuscript use |
| --- | --- | --- |
| [TCGA-SKCM](TCGA_SKCM/) | [Frozen scoring and survival](analyses/tcga.py), with 469 patients scored and 455 with usable overall survival | Fig. 3f, Fig. S4c, Table S3 |
| [Liu2019](Liu2019/) | [Pretreatment anti-PD-1 cohort](analyses/liu.py), with PFS and OS Cox estimates for 104 patients | Fig. 3g, Table S3 |
| [GSE272993](GSE272993_longitudinal/) | [R matrix export](analyses/export_longitudinal.R) and [longitudinal scoring](analyses/longitudinal.py) over five visits. The four treatment-naive anti-PD-1 patients (13-689, 16-2006, 17-3017 and 18-3282) come from the GSE272734 cohort within the integrated RDS. | Fig. 3h, Fig. S4d, Table S3 |
| TCGA gene contributions and enrichment | [Gene decomposition](analyses/decompose.py), [overall ORA](analyses/enrichment_all.py), [signed ORA](analyses/enrichment_sign.py) and [Top50 annotations](analyses/top50.py) | Fig. 3d/e, Fig. S5a/b, Table S3 |
| TCGA supervised and fixed gene-score comparisons | [Five-program Ridge](analyses/five_program_ridge.py), [gene Ridge](analyses/gene_ridge.py), [fixed TopK](analyses/fixed_topk.py) and [six-score Cox](analyses/table_s3_cox.py). Outputs include out-of-fold predictions, fitted coefficients and patient-bootstrap summaries. | Table S3 |

Shared scoring, survival and enrichment code is in [common/](common/).
The download manifest covers raw cohort inputs, the HCOP orthology map and
fixed GO Biological Process, Reactome and Hallmark libraries. [Data sources](../data/README.md#external-human-inputs).

Saved results feed the [Fig. 3 notebook](figure_workbench/fig03_dynamics_human_validation.ipynb)
and [human supplement](../supplementary/supp_human.ipynb). The Table S3 export
uses the selected Cox, C-index, TopK, gene-annotation and longitudinal tables.
[Analysis order and commands](../docs/REPRODUCIBILITY.md#human-validation).
