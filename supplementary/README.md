# Supplementary figures and tables

Each notebook loads saved results for a group of panels. Run its setup and
data-loading cells, then the cells for the desired panels.

| Notebook | Panels | Figure output |
| --- | --- | --- |
| [supp_model.ipynb](supp_model.ipynb) | S1–S3 and S4a/b | `figure/model/` |
| [supp_human.ipynb](supp_human.ipynb) | S4c/d and S5a/b | `figure/human_validation/` |
| [supp_policy.ipynb](supp_policy.ipynb) | S6 and S7 | `figure/policy/` |

The model panels cover cellular and tumour-volume holdouts, training losses,
drift, state densities and perturbations. The human panels cover metastatic
survival, longitudinal scores and enrichment. Fig. S4c calculates the KM risk
counts in memory. The policy panels cover training, final state, across-seed
outcomes, input attribution and open-loop schedules.
[Model panel manifest](../MultiNeD/figure_work/panel_manifest.csv).

The five workbooks contain 25 data sheets. The paper lists fitted model
parameters in Supplementary Table S6. [Workbooks](../paper/source_data/).

| Workbook | Data sheets | Contents |
| --- | ---: | --- |
| S1 | 4 | Cell-LOTO means, paired tests and fold metrics for seed 20260816, plus pretrained drift versus complete MultiNeD |
| S2 | 6 | Full-data reconstruction, training-loss ablations and 48 held-out tumour-volume predictions |
| S3 | 7 | TCGA Cox, C-index, fixed TopK and gene-function annotations, longitudinal scores and Liu2019 PFS/OS Cox |
| S4 | 5 | Objective, observation and history ablations and patient-level outcomes |
| S5 | 3 | Ten-policy benchmark means, paired tests and patient-level results averaged over seeds |

The [table exporter](prepare_supplementary_tables.py) assembles CSVs and a
[source manifest](supplementary_tables/supplementary_tables_manifest.csv)
from saved results. It can select new model, policy and human
analysis outputs. [Figure and table commands](../docs/REPRODUCIBILITY.md#figures-and-source-tables).
