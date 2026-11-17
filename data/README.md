# Data inputs

The repository includes processed mouse states, distribution and transition
files, fitted models and policies, virtual-patient cohorts, saved human
validation results and figure tables. The reproduction starts from the
processed mouse states and distributions.

## Included mouse inputs

The B16-OVA tumour OT1 CD8 inputs are under [`data/mouse/`](mouse/).

| Directory | Files and fields |
| --- | --- |
| [`source_metadata/`](mouse/source_metadata/) | `paper_tumor_volume_long.csv` supplies joint calibration. `B16_OT1_tumor_joint_clean_metadata.csv` links samples, mice, days and treatments to source files. The `Fig1e_left` sheet of `Fig_1.xlsx` supplies original program weights. |
| [`program_weights/`](mouse/program_weights/) | `Fig1e_program_gene_weights_long.csv` and `Fig1e_program_top_genes.csv` contain `program`, `gene` and `weight`. Mouse scores use the top 50 genes per source program with normalized weights. Human validation also uses the top-gene table. |
| [`cell_level_z/`](mouse/cell_level_z/) | `B16_OT1_tumor_joint_fig1e_cell_z.csv` holds raw weighted scores and cell metadata. The `_cell_zstd.csv` counterpart stores the standardized state. Both retain sample, mouse, day and treatment identifiers. |
| [`group_distributions/`](mouse/group_distributions/) | `B16_OT1_tumor_joint_zstd_group_distributions.pkl` stores cell-level states by treatment and day. `Z_all` has five columns per cell, and `by_mouse` supports balanced sampling. The matching `_group_info.csv` adds training-use labels and `loss_weight`. Replicate and cell-count tables support QC. |
| [`manifests/`](mouse/manifests/) | `B16_OT1_tumor_joint_transition_manifest_temporal.csv` is the training manifest. It records source and target groups, days, `dt`, treatment flags and `loss_weight`. The other manifests provide endpoint references, group usage and sampling QC. |
| [`scalers/`](mouse/scalers/) | `B16_OT1_fig1e_z_scaler.pkl` stores the full-data means and standard deviations. Use its `transform()` to place additional cells in the same coordinates. |

The five standardized columns are `P_precursor_zstd`, `Activation_zstd`,
`Eff_Early_zstd`, `Eff_Cytox_zstd` and `Dysf_zstd`. Their raw counterparts
use the suffix `_wscore`. The precursor score averages the Naive-Like,
Memory-Like and Naive-Il7r scores. Treatment flags `(P, B)` indicate anti-PD-1
and anti-4-1BB. Control has `(0, 0)`, monotherapies have `(1, 0)` or
`(0, 1)`, and combination has `(1, 1)`. The weighted group files include
the weak day-3 anti-4-1BB and combination groups. Unweighted counterparts
retain their original exploratory-use labels.

Cellular and tumour-volume holdout experiments fit their own scaling and
activity quantiles from the training cells in each fold. The full-data scaler
is used for the full-data model. Mouse data originate from GEO accession
[GSE249630](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE249630) and
the accompanying Barboy et al. source materials.

## External human inputs

The [download manifest](../human_validation/fetch_data.py) records URLs, sizes
and paths for TCGA-SKCM (UCSC Xena and GENCODE), [Liu et al. (2019)](https://doi.org/10.1038/s41591-019-0654-5), and the
longitudinal single-cell analysis.
They share the HCOP orthology map and fixed GO, Reactome and Hallmark gene sets.

The longitudinal analysis selects the treatment-naive anti-PD-1 cohort
[GSE272734](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE272734) from
the integrated [GSE272993](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE272993)
RDS (about 2 GB). Exporting this matrix requires R and its Matrix package.
Raw downloads and matrix exports are Git-ignored. The saved results can be
plotted directly. See [human validation](../docs/REPRODUCIBILITY.md#human-validation)
for download, export and analysis commands. Third-party data retain their
original terms and attribution.
