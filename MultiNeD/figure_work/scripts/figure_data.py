"""Read the prepared Fig. 2 tables."""
from pathlib import Path
import json
import os
import pandas as pd


SOURCE_TABLES = {
    'program_importance_by_delta.csv', 'tumour_doses.csv', 'drift_dense_field.csv.gz',
    'state_summary.csv', 'reaction_volume.csv', 'reaction_training_history.csv',
    'drift_background.csv.gz', 'dynamics_loto_targets.csv', 'drift_display_parameters.json',
    'evaluation_metadata.json', 'drift_training_history.csv', 'tumour_volume.csv',
    'balanced_loto_predictions.csv',
    'downstream_trajectory_particles.csv',
}


def find_analysis_dir(start=None):
    return Path(os.environ.get("MULTINED_FIGURE_DATA", start or Path(__file__).resolve().parents[1]))


def load_table(name, analysis_dir=None, **kwargs):
    folder = "source_data" if name in SOURCE_TABLES else "tables"
    path = find_analysis_dir(analysis_dir) / folder / name
    return pd.read_csv(path, float_precision="round_trip", **kwargs)


def load_main_data(analysis_dir=None):
    return {key: load_table(name, analysis_dir) for key, name in {
        "loto": "balanced_loto_predictions.csv",
        "dynamics_benchmark_folds": "dynamics_loto_fold_metrics.csv",
        "full_data_ablation": "full_data_ablation_by_seed.csv",
        "full_data_volume_r2": "full_data_volume_r2_by_treatment_seed.csv",
    }.items()}


def load_main_rollouts(analysis_dir=None):
    directory = find_analysis_dir(analysis_dir)
    result = {key: load_table(key + ".csv", directory)
              for key in ("reaction_volume", "tumour_volume", "tumour_doses", "state_summary")}
    metadata = json.loads((directory / "source_data/evaluation_metadata.json").read_text(encoding="utf-8"))
    result["z_cols"] = metadata["z_cols"]
    return result
