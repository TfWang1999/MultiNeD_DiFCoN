# MultiNeD_DiFCoN

MultiNeD fits tumour growth and CD8 T-cell state dynamics from mouse data.
DiFCoN uses the fitted model to train and evaluate feedback dosing policies for
anti-PD-1 and anti-4-1BB.

## Repository layout

| Directory | Contents |
| --- | --- |
| [MultiNeD](MultiNeD/README.md) | Drift pretraining, joint calibration, benchmarks and Fig. 2 |
| [DiFCoN](DiFCoN/README.md) | Policy training, evaluation and Figs. 4–6 |
| [dynamics_analysis](dynamics_analysis/README.md) | Simulated trajectories, pseudocells, drift fields and program perturbations |
| [human_validation](human_validation/README.md) | Human CD8score analyses and Fig. 3 |
| [data](data/README.md) | Processed mouse inputs and sources for external human data |
| [supplementary](supplementary/README.md) | Supplementary figures and [source workbooks](paper/source_data/) |

## Install

Use Python 3.11 (tested with 3.11.15). The pinned PyTorch build supports CPU and
CUDA on Windows and Linux. Use a CUDA GPU for full training.

```text
git clone -c core.longpaths=true https://github.com/TfWang1999/MultiNeD_DiFCoN.git
cd MultiNeD_DiFCoN
python -m venv .venv
```

Activate the environment with `.\.venv\Scripts\Activate.ps1` in PowerShell
or `source .venv/bin/activate` in Bash, then install the dependencies.

```text
python -m pip install -r requirements.txt
```

## Quick reproduction

Preview the configured training runs from the repository root.

```text
python MultiNeD/reproduce_MultiNeD.py joint --device cpu --plan
python DiFCoN/reproduce_DiFCoN.py baseline --device cpu --plan
```

To plot the included results, open the [Fig. 2 notebook](MultiNeD/figure_work/notebooks/fig02_MultiNeD.ipynb)
in Jupyter or in VS Code with the Jupyter extension. Select the installed Python
kernel and run the setup and data-loading cells before the desired panels.

## Further information

- [Reproduction guide](docs/REPRODUCIBILITY.md) for training, external baselines, human validation and all figure and table commands.
- [Data sources and downloads](data/README.md).
- [Citation](CITATION.cff) and [MIT license](LICENSE). Third-party data and code retain their original terms.
