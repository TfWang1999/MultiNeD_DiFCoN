"""Path constants for the manuscript figure notebooks."""

import os
from pathlib import Path


WORKBENCH_DIR = Path(__file__).resolve().parents[1]
TRAINING_DIR = WORKBENCH_DIR.parent
EXPERIMENT_DIR = TRAINING_DIR / "outputs"
DEFAULT_SOURCE_DATA_DIR = WORKBENCH_DIR / "source_data"
SOURCE_DATA_DIR = Path(os.environ.get("DIFCON_SOURCE_DATA", DEFAULT_SOURCE_DATA_DIR)).resolve()
FIGURE_DIR = (WORKBENCH_DIR if SOURCE_DATA_DIR == DEFAULT_SOURCE_DATA_DIR else SOURCE_DATA_DIR) / "figure"

PARETO_DIR = EXPERIMENT_DIR / "dose_weight_sweep"
DEFAULT_BASELINE_DIR = PARETO_DIR / "dose_weight_1p0_seed_2032"
BASELINE_DIR = Path(os.environ.get("DIFCON_BASELINE_RUN", DEFAULT_BASELINE_DIR))
PILOT_DIR = EXPERIMENT_DIR / "objective_ablation"
MULTISEED_DIR = EXPERIMENT_DIR / "observation_history_ablation"
OPENLOOP_DIR = Path(os.environ.get("DIFCON_OPENLOOP_DIR", EXPERIMENT_DIR / "openloop_comparison"))
LOCKED_BENCHMARK_DIR = EXPERIMENT_DIR / "locked_test_benchmark"
INPUT_ATTRIBUTION_DIR = EXPERIMENT_DIR / "input_attribution"
