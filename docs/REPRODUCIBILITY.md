# Reproducing the paper

Set up the [Python environment](../README.md#install) and run the commands
from the repository root in Bash.

- [Use the supplied results](#figures-and-source-tables) to draw the figures and export the supplementary tables.
- [Rerun the study](#multined) to train both models, run the evaluations and prepare new figures and tables.

The examples use the workspace below. The formal MultiNeD and DiFCoN entries
accept `--plan` to preview the tasks and paths. `--output-dir` selects the
destination. The default is the task directory under `MultiNeD/outputs/` or
`DiFCoN/outputs/`.

```bash
run=reproduction/paper
device=cuda:0
drift="$run/notebook_drift_run"
joint="$run/reaction_coupled_run"
```

## MultiNeD

Train the drift and joint models using the [processed mouse inputs](../data/README.md#included-mouse-inputs).
The [training settings](../MultiNeD/scripts/reproduce/paper_settings.py)
use 800 epochs for drift and 600 for joint calibration. Keep the checkpoint and
its accompanying files together for validation, ablations and the DiFCoN cohort.

```bash
python -B MultiNeD/reproduce_MultiNeD.py drift --device "$device" --output-dir "$drift"
python -B MultiNeD/reproduce_MultiNeD.py joint --device "$device" \
  --drift-source "$drift" --output-dir "$joint"
python -B MultiNeD/reproduce_MultiNeD.py volume-loto --device "$device" \
  --drift-source "$drift" --reaction-source "$joint" --output-dir "$run/volume_loto"
python -B MultiNeD/reproduce_MultiNeD.py compare-pretrained --device "$device" \
  --drift-source "$drift" --reaction-source "$joint" --output-dir "$run/pretrained_vs_MultiNeD"
```

Install the [external benchmark methods](#external-benchmark-methods) before
running the two cell benchmarks. Loss ablations use the matching drift and
joint checkpoints, parameter settings and empirical `q_in` saved by joint
calibration.

```bash
python -B MultiNeD/reproduce_MultiNeD.py cell-loto --device "$device" \
  --drift-source "$drift" --output-dir "$run/cell_loto_benchmark"
python -B MultiNeD/reproduce_MultiNeD.py cell-reconstruction --device "$device" \
  --output-dir "$run/cell_reconstruction_benchmark"
python -B MultiNeD/reproduce_MultiNeD.py loss-ablation --device "$device" \
  --drift-source "$drift" --reaction-source "$joint" --output-dir "$run/training_loss_ablation"
```

| Task | What it runs | Key output within the selected directory |
| --- | --- | --- |
| `drift` | Pretrain the CD8 drift model. | `drift_model.pt` and `drift_training_history.csv` |
| `joint` | Fit the cellular and tumour-volume data starting from the pretrained drift. | `reaction_coupled_model.pt`, `reaction_coupled_history.csv`, `reaction_parameter_settings.json` and `q_in_precursor_empirical.npy` |
| `volume-loto` | Hold out one treatment–day tumour-volume group per fold across four treatments and three seeds. [Configuration](../MultiNeD/configs/validation/volume_loto_common_days.json) | 48 predictions in `combined/all_heldout_volume_metrics.csv` |
| `compare-pretrained` | Compare the two fitted checkpoints on the same samples at seed 20260816. | `distribution_metrics.csv` and `reconstruction_summary.csv` for Table S1 |
| `cell-loto` | Compare seven methods on six cellular holdouts at seed 20260816. [Configuration](../MultiNeD/configs/benchmarking/cell_loto_benchmark.json) | 42 metric rows in the seed run, with summaries in `b1_shared_loto_all7_combined/` |
| `cell-reconstruction` | Compare six methods on full-data reconstruction from Control day 1 over three seeds. [Configuration](../MultiNeD/configs/benchmarking/cell_reconstruction_benchmark.json) | `b1_full_data_combined/` for Table S2 |
| `loss-ablation` | Fit five loss and pretraining variants over three seeds. [Configuration](../MultiNeD/configs/benchmarking/training_loss_ablation.json) | Transition, rollout and volume summaries in `full_data_loss_all_seeds_combined/` |

Cell and volume holdouts fit their scaling on the training fold. Full-data
reconstruction and loss ablations report in-sample fit.

### External benchmark methods

Run these scripts on Linux with Bash and Conda. They set up the pinned MIOFlow,
TIGON and OT-CFM sources and environments. OT-CFM is used by Cell-LOTO. The
default source directory is `external_baselines/` next to this repository.

```bash
bash MultiNeD/experiments/benchmarks/setup_external_baselines.sh
bash MultiNeD/experiments/benchmarks/setup_otcfm.sh
```

To use existing installations, set `MIOFLOW_REPO`, `TIGON_REPO` and
`OTCFM_REPO` to the source directories and `MIOFLOW_PYTHON`, `TIGON_PYTHON` and
`OTCFM_PYTHON` to the corresponding Python executables.

## DiFCoN

### Generate virtual patients

Use the [supplied cohort](../DiFCoN/cohorts/) with the supplied MultiNeD weights.
For a new fit, run the following after joint calibration in a separate training
checkout. This writes the 128 training, 32 validation and 32 test patients to
`DiFCoN/cohorts/` using the new model and the
[LHS configuration](../DiFCoN/configs/direct_dose/lhs_cohort.json).
The fitted baseline is recorded in `cohort_generation_config.json`.

```bash
python -B - "$run" <<'PY'
import json
import sys
from pathlib import Path

run = Path(sys.argv[1])
config = json.loads(Path("DiFCoN/configs/direct_dose/lhs_cohort.json").read_text())
config["model"]["drift_checkpoint"] = (run / "notebook_drift_run/drift_model.pt").as_posix()
config["model"]["reaction_checkpoint"] = (run / "reaction_coupled_run/reaction_coupled_model.pt").as_posix()
(run / "lhs_cohort.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
PY

python -B DiFCoN/scripts/cohorts/generate_direct_dose_cohorts.py \
  --config "$PWD/$run/lhs_cohort.json" --from-checkpoints
```

### Train and evaluate policies

Train with the new checkpoints and matching cohort. For the supplied model and
cohort, omit `--drift-source` and `--reaction-source`. The
[base configuration](../DiFCoN/configs/direct_dose/base.json) specifies the
200-epoch budget and policy settings.

```bash
policy="$run/dose_weight_sweep"
openloop="$run/openloop_comparison"
sac="$run/sac_training"
locked="$run/locked_test_benchmark"

# Standalone baseline, if needed separately from the dose sweep.
python -B DiFCoN/reproduce_DiFCoN.py baseline --device "$device" \
  --drift-source "$drift" --reaction-source "$joint" --output-dir "$run/baseline"

python -B DiFCoN/reproduce_DiFCoN.py dose-sweep --device "$device" \
  --drift-source "$drift" --reaction-source "$joint" --output-dir "$policy"
python -B DiFCoN/reproduce_DiFCoN.py objective-ablation --device "$device" \
  --drift-source "$drift" --reaction-source "$joint" --policy-root "$policy" \
  --output-dir "$run/objective_ablation"
python -B DiFCoN/reproduce_DiFCoN.py observation-history --device "$device" \
  --drift-source "$drift" --reaction-source "$joint" --output-dir "$run/observation_history_ablation"
python -B DiFCoN/reproduce_DiFCoN.py openloop --device "$device" \
  --policy-root "$policy" --output-dir "$openloop"
python -B DiFCoN/reproduce_DiFCoN.py attribution --device "$device" \
  --policy-root "$policy" --output-dir "$run/input_attribution"
```

| Task | What it runs and where to find the results |
| --- | --- |
| `baseline` | Fit the standalone weight-1 policy at seed 2031. Each fit saves the policy in `checkpoints/policy.pt` and test results in `tables/`. |
| `dose-sweep` | Fit eight dose weights at seeds 2031–2033. The summary is in `aggregate/seed_summary.csv`. |
| `objective-ablation` | Fit four loss variants at seed 2031. `BASELINE_REFERENCES.tsv` selects the full-objective reference from the sweep. |
| `observation-history` | Fit the practical and schedule-only observations and history lengths 3 and 4 over three seeds. Results are in the 12 policy run directories. |
| `openloop` | Derive schedules on validation and evaluate them with frozen feedback policies on test. The schedules are in `validation_derived_openloop_schedule.csv`. |
| `attribution` | Run input attribution and history perturbations on validation using all three weight-1 policies. Results are in `full_validation/`. |

Fig. 4 and Figs. S6a–c use the dose-weight-1 policy at seed 2032.

### Train SAC and run the locked-test benchmark

For a new fit, train SAC in the matching environment before running the
locked test. SAC trains on the training cohort and evaluates on validation.
It saves actors in `seed_<seed>/model_seed_<seed>/sac_actor.pt`.
With the supplied model and cohort, skip SAC training and omit `--sac-root`
to load the supplied actors.

```bash
for seed in 2031 2032 2033; do
  python -B DiFCoN/scripts/benchmarking/run_policy_benchmark.py \
    --config DiFCoN/configs/benchmarking/sac_training.json --cohort validation \
    --policy-root "$PWD/$policy" --openloop-source "$PWD/$openloop" \
    --model-seed "$seed" --device "$device" --output-dir "$PWD/$sac/seed_$seed" || break
done

python -B DiFCoN/reproduce_DiFCoN.py benchmark --device "$device" \
  --policy-root "$policy" --openloop-source "$openloop" --sac-root "$sac" \
  --output-dir "$locked"
```

The [locked test](../DiFCoN/configs/benchmarking/locked_test_benchmark.json)
compares ten strategies on the 32 test patients. It writes patient metrics,
actions and compute accounting per seed and the summaries in `combined/`.
These results feed Fig. 6, Fig. S7 and Table S5.

## Dynamics analysis

Start with matched MultiNeD checkpoints and use the same device for baseline
generation and perturbation replay. Omit the two model-source arguments on
the three model-evaluation commands to use the supplied weights.

```bash
dyn="$run/dynamics_analysis"
python -B dynamics_analysis/run_analysis.py trajectories --device "$device" \
  --drift-source "$drift" --reaction-source "$joint" --output "$dyn/inputs"
python -B dynamics_analysis/run_analysis.py pseudocells --inputs "$dyn/inputs" \
  --output "$dyn/pseudocell/tables"
python -B dynamics_analysis/run_analysis.py drift-source --device "$device" \
  --drift-source "$drift" --reaction-source "$joint" --inputs "$dyn/inputs" \
  --pseudocells "$dyn/pseudocell/tables" --output "$dyn/vtheta/source_data"
python -B dynamics_analysis/run_analysis.py drift-field --source "$dyn/vtheta/source_data" \
  --output "$dyn/vtheta/plot_data"
python -B dynamics_analysis/run_analysis.py perturb --device "$device" \
  --drift-source "$drift" --reaction-source "$joint" --inputs "$dyn/inputs" \
  --deltas 0.5 0.25 1.0 --output "$dyn/perturb"
python -B dynamics_analysis/run_analysis.py importance \
  --metrics "$dyn/perturb/program_perturbation_metrics.csv" --output "$dyn/robustness"
```

The workflow writes baseline trajectories and reference data in `inputs/`,
pseudocells in `pseudocell/tables/`, and the drift sources and display grid in
`vtheta/`. The perturbation metrics in `perturb/` feed the CTB-based program
sensitivity summary in `robustness/dynamic_importance_by_delta.csv`.
[Input and output map](../dynamics_analysis/README.md).

## Human validation

Download the [external inputs](../data/README.md#external-human-inputs).
To select one cohort, use `--group tcga`, `--group liu` or `--group longitudinal`.
`--group resources` selects the shared inputs. Add `--check` to check
availability.

```bash
python -B human_validation/fetch_data.py --list
python -B human_validation/fetch_data.py --group all
```

R with the Matrix package exports the longitudinal expression matrix. The
Python analyses use one result workspace under `human_validation/runs/`.
Use the new dynamics workspace for the program weights and paired mouse
trajectories. Omit `--dynamics-dir` and `--mouse-trajectories` to use the
supplied results.

```bash
human=human_validation/runs/paper_reproduction
Rscript human_validation/analyses/export_longitudinal.R \
  human_validation/GSE272993_longitudinal/data/raw/GSE272993_cd8_nn_labeled_FINAL.RDS.gz \
  "$human/GSE272993_longitudinal/data/exported" data/required_human_genes.txt

for task in tcga decompose top50 enrichment-all enrichment-sign five-program-ridge gene-ridge fixed-topk table-s3-cox liu; do
  python -B human_validation/run_validation.py "$task" --input-root "$human" \
    --output "$human" --dynamics-dir "$dyn" || break
done
python -B human_validation/run_validation.py longitudinal --input-root "$human" \
  --output "$human" --dynamics-dir "$dyn" --mouse-trajectories "$dyn/inputs/paired_timeseries.csv"
```

TCGA scoring feeds gene decomposition and enrichment. Five-program Ridge
supplies the folds for gene Ridge and fixed TopK. The six-score Cox analysis
uses the saved gene Ridge predictions. Liu and GSE272993 run independently
of TCGA. The results under `$human` are organized into `TCGA_SKCM/`,
`Liu2019/` and `GSE272993_longitudinal/` for Fig. 3, the human supplement
and Table S3. [Cohort and output map](../human_validation/README.md).

## Figures and source tables

To plot the supplied results, select the installed Python kernel and run the
setup, data-loading and relevant panel cells. In Fig. 2, `SAVE_FIGURES`
controls export.

| Panels | Notebook | Default figure directory |
| --- | --- | --- |
| Fig. 2a–f | [fig02_MultiNeD.ipynb](../MultiNeD/figure_work/notebooks/fig02_MultiNeD.ipynb) | `MultiNeD/figure_work/figures/` |
| Fig. 3a–h | [fig03_dynamics_human_validation.ipynb](../human_validation/figure_workbench/fig03_dynamics_human_validation.ipynb) | `human_validation/figure_workbench/outputs/figure3_main/` |
| Fig. 4 | [fig04_DiFCoN_dynamics.ipynb](../DiFCoN/figure_workbench/fig04_DiFCoN_dynamics.ipynb) | `DiFCoN/figure_workbench/figure/` |
| Fig. 5 | [fig05_tradeoffs_ablations.ipynb](../DiFCoN/figure_workbench/fig05_tradeoffs_ablations.ipynb) | `DiFCoN/figure_workbench/figure/` |
| Fig. 6 | [fig06_policy_benchmark.ipynb](../DiFCoN/figure_workbench/fig06_policy_benchmark.ipynb) | `DiFCoN/figure_workbench/figure/` |
| Figs. S1–S3 and S4a/b | [supp_model.ipynb](../supplementary/supp_model.ipynb) | `supplementary/figure/model/` |
| Figs. S4c/d and S5a/b | [supp_human.ipynb](../supplementary/supp_human.ipynb) | `supplementary/figure/human_validation/` |
| Figs. S6–S7 | [supp_policy.ipynb](../supplementary/supp_policy.ipynb) | `supplementary/figure/policy/` |

Use these commands to export the policy plotting tables and the 25
supplementary CSVs from supplied results. The supplied Excel versions are in
[paper/source_data/](../paper/source_data/).

```bash
python -B DiFCoN/figure_workbench/data_preparation/prepare_source_tables.py \
  --output-dir "$run/saved_policy_source_data"
python -B supplementary/prepare_supplementary_tables.py \
  --output-dir "$run/saved_supplementary_tables"
```

### Connect new results to figures

Export Fig. 2 trajectories and the supplementary state trajectories to the
same `source_data/` directory. Figure preparation collects the volume
holdouts, cell holdouts, loss ablations and dynamics data into the MultiNeD
workspace and builds its `tables/`.

```bash
loto="$run/cell_loto_benchmark/b1_shared_loto_all7_combined"
reconstruction="$run/cell_reconstruction_benchmark/b1_full_data_combined"
ablation="$run/training_loss_ablation/full_data_loss_all_seeds_combined"

python -B MultiNeD/scripts/export_figure_work_rollouts.py --device "$device" \
  --drift-source "$drift" --reaction-source "$joint" --output-dir "$run/MultiNeD/source_data"
python -B MultiNeD/scripts/export_downstream_trajectories.py --device "$device" \
  --drift-source "$drift" --reaction-source "$joint" --output-dir "$run/MultiNeD/source_data"
python -B MultiNeD/figure_work/scripts/prepare_figure_data.py \
  --volume-loto-dir "$run/volume_loto" --loss-ablation-dir "$ablation" \
  --cell-loto-dir "$loto" --dynamics-dir "$dyn" --output-dir "$run/MultiNeD"
```

Export the policy inputs from the selected studies. Use the same baseline run
for the exporter and the policy notebooks.

```bash
python -B DiFCoN/figure_workbench/data_preparation/prepare_source_tables.py \
  --baseline-run "$policy/dose_weight_1p0_seed_2032" --dose-sweep-dir "$policy" \
  --objective-ablation-dir "$run/objective_ablation" \
  --observation-history-dir "$run/observation_history_ablation" \
  --openloop-dir "$openloop" --benchmark-dir "$locked" \
  --attribution-dir "$run/input_attribution" --output-dir "$run/DiFCoN/source_data"
```

Set the applicable paths in a fresh Notebook kernel before the setup cell.

```python
import os
from pathlib import Path

repo = Path("/absolute/path/to/MultiNeD_DiFCoN")
run = repo / "reproduction/paper"
os.environ["MULTINED_FIGURE_DATA"] = str(run / "MultiNeD")
os.environ["DIFCON_SOURCE_DATA"] = str(run / "DiFCoN/source_data")
os.environ["DIFCON_BASELINE_RUN"] = str(run / "dose_weight_sweep/dose_weight_1p0_seed_2032")
os.environ["DIFCON_OPENLOOP_DIR"] = str(run / "openloop_comparison")
os.environ["HUMAN_VALIDATION_RESULTS"] = str(repo / "human_validation/runs/paper_reproduction")
os.environ["DYNAMICS_RESULTS"] = str(run / "dynamics_analysis")
```

Fig. 2 writes to `figures/` under `MULTINED_FIGURE_DATA`. Figs. 4–6 write to
`figure/` under the selected `DIFCON_SOURCE_DATA` when using new sources.
With nondefault inputs, supplementary outputs go to
`reproduction/supplementary/figure/`.

### Export new supplementary tables

The exporter writes S1–S5 CSVs and a source manifest. For a partial rerun,
pass only the relevant source options to load the other inputs from the
supplied results. `--human-display-dir` can also export the six displayed
human source tables.

```bash
python -B supplementary/prepare_supplementary_tables.py \
  --cell-loto-dir "$loto" --cell-reconstruction-dir "$reconstruction" \
  --loss-ablation-dir "$ablation" --model-source-dir "$run/MultiNeD/source_data" \
  --pretrained-comparison-dir "$run/pretrained_vs_MultiNeD" \
  --dose-sweep-dir "$policy" --objective-ablation-dir "$run/objective_ablation" \
  --observation-history-dir "$run/observation_history_ablation" --benchmark-dir "$locked" \
  --human-results-dir "$human" --output-dir "$run/supplementary_tables"
```

## Refit the Cell-LOTO MultiNeD method

This optional rerun uses a completed Cell-LOTO workspace. It replaces the
MultiNeD metrics and retains the six baseline methods. Use the saved seed
configuration and reaggregate the merged results.

```bash
loto="$run/cell_loto_benchmark"
python -B MultiNeD/experiments/benchmarks/run_cell_state_benchmark.py \
  --config "$loto/configs/b1_shared_loto_all7_seed20260816.json" \
  --seed 20260816 --device "$device" --methods project_balanced_sinkhorn_drift \
  --keep-other-methods --output-dir "$loto/b1_shared_loto_all7_seed20260816"

python -B MultiNeD/experiments/benchmarks/aggregate_dynamics_benchmark.py \
  --run-dir "$loto/b1_shared_loto_all7_seed20260816" \
  --output-dir "$loto/b1_shared_loto_all7_combined"
```
