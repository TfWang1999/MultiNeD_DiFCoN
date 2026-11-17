# Dynamics analysis

The workflow starts with a fitted MultiNeD model and the processed mouse data.
It simulates the four treatments, groups simulated cells in a common diffusion
map, evaluates the frozen drift and measures the effect of perturbing each of
the five programs.

| Stage | Implementation | Saved inputs or results |
| --- | --- | --- |
| Baseline simulations and projection reference | [trajectories.py](src/trajectories.py) | [Baseline inputs](mechanism_aligned_dynamics/inputs/) contain balanced observed cells, weighted simulated particles and their inflow indices, paired time series, state centers and the frozen Nyström basis. |
| Pseudocells | [pseudocells.py](src/pseudocells.py) | [Pseudocell tables](outputs/pseudocell/tables/) contain projected particles, fixed membership, averaged coordinates and velocity anchors for Fig. 3a/b and Figs. S2a–d. |
| Direct drift field | [drift_field.py](src/drift_field.py) | [Source tables](outputs/vtheta/source_data/) hold projected neural drift and display parameters. The [display grid](outputs/vtheta/plot_data/) serves Fig. 3b. |
| Paired perturbations and program weights | [perturbations.py](src/perturbations.py), [importance.py](src/importance.py) | [Robustness results](outputs/robustness/) contain paired perturbation metrics at all amplitudes, `dynamic_importance_by_delta.csv` and the run record. Fig. 3c and Figs. S4a/b use these summaries. |

Paired perturbations quantify program sensitivity to cumulative tumor burden
(CTB). The main figure and human validation use δ = 0.5. TCGA uses Control
weights. Liu2019 and GSE272993 use anti-PD-1 weights, and the longitudinal
analysis also reads the paired mouse time series.

The plotting notebooks can read the saved results. To generate matched inputs
from new checkpoints, run the six stages in order. [Reproduction commands](../docs/REPRODUCIBILITY.md#dynamics-analysis).
