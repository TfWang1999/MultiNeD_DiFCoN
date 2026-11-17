# DiFCoN

DiFCoN trains feedback dosing policies in the fitted MultiNeD environment. The
[reproduction entry](reproduce_DiFCoN.py) runs training, ablations and evaluation.
The [reproduction guide](../docs/REPRODUCIBILITY.md#difcon) lists commands and
dependencies for each study.

| Location | Contents |
| --- | --- |
| [`core/`](core/) | Policy network, simulator, cohort loading, loss and training |
| [`train_DiFCoN.py`](train_DiFCoN.py) | Train one policy using the [base configuration](configs/direct_dose/base.json) and its overrides |
| [`cohorts/`](cohorts/) | Fixed LHS train, validation and test cohorts of 128, 32 and 32 virtual patients |
| [`scripts/`](scripts/) | Cohort generation, policy evaluation, benchmark and analysis |
| [`outputs/`](outputs/) | Policy checkpoints, training histories, evaluation metrics and run settings |
| [`figure_workbench/`](figure_workbench/) | Notebooks and source tables for Figs. 4–6 |

Each training run saves its actual settings in `resolved_config.json`, the
policy in `checkpoints/policy.pt`, and cohort-level metrics and patient-level
test outputs. The locked-test benchmark compares ten strategies on the same
32 patients, using three model seeds and SAC actors trained in the matching
environment.

Fig. 4 and Fig. S6a–c use the dose-weight-1 policy with seed 2032.
Objective ablations use seed 2031. Input attribution and history perturbations
use all three dose-weight-1 policies on the same validation cohort.
See [figures and source tables](../docs/REPRODUCIBILITY.md#figures-and-source-tables)
for the notebooks and their outputs.
