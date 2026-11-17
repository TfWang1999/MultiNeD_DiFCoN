# MultiNeD

MultiNeD fits CD8 T-cell functional-state dynamics and couples the fitted drift
to CD8 abundance and tumour volume. The [reproduction entry](reproduce_MultiNeD.py)
uses the processed B16-OVA OT1 CD8 inputs described in the [data guide](../data/README.md#included-mouse-inputs).

| Location | Contents |
| --- | --- |
| [`common/`](common/) | Model and PK components, mouse-balanced sampling, Sinkhorn loss and display transforms |
| [`training/`](training/) | Drift training, reaction-coupled dynamics and checkpoint I/O |
| [`scripts/reproduce/`](scripts/reproduce/) | Training entry helpers and [paper model settings](scripts/reproduce/paper_settings.py) |
| [`configs/`](configs/) | JSON settings for [volume holdout](configs/validation/volume_loto_common_days.json), [cellular holdout](configs/benchmarking/cell_loto_benchmark.json), [full-data reconstruction](configs/benchmarking/cell_reconstruction_benchmark.json) and [loss ablations](configs/benchmarking/training_loss_ablation.json) |
| [`experiments/`](experiments/) | Volume splits and refits, cellular baselines and ablation runners |
| [`outputs/`](outputs/) | Fitted checkpoints, histories, per-seed metrics and combined results |
| [`figure_work/`](figure_work/) | Fig. 2 notebook, saved source tables and plot exports |

The [drift](notebooks/01_pretrain_drift.ipynb) and [joint calibration](notebooks/02_calibrate_MultiNeD.ipynb)
notebooks use the same settings as the command-line entry. They expose
`DEVICE` and `OUTPUT_DIR`. Joint calibration also accepts `DRIFT_SOURCE`.

Use the [reproduction guide](../docs/REPRODUCIBILITY.md#multined) for the
training and benchmark commands, budgets and result paths. The [figure guide](../docs/REPRODUCIBILITY.md#figures-and-source-tables)
links each panel to its notebook and inputs.
