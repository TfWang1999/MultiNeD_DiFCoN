"""Collect experiment results and build the Fig. 2 / model-supplement tables."""
from pathlib import Path
import argparse
import shutil
import numpy as np
import pandas as pd
from constants import (DYNAMICS_METRICS, PRIMARY_DYNAMICS_METHOD_ORDER,
                       FULL_DATA_ABLATION_ORDER, HELDOUT_DAY_ORDER, VARIANT_LABELS)

WORK = Path(__file__).resolve().parents[1]
REPO = WORK.parents[1]
OUTPUTS = WORK.parent / "outputs"

def _prepare_shared_loto_fold_metrics(targets):
    """Average shared-LOTO metrics across seeds within each held-out fold."""
    id_columns = ["seed", "split_id", "target_group"]
    if missing := set(id_columns).difference(targets.columns):
        raise KeyError(f"Missing shared-LOTO target columns: {sorted(missing)}")
    if targets.duplicated(id_columns).any():
        raise ValueError("Shared-LOTO targets are not unique by seed, split, and target group.")

    long_parts = []
    for metric in DYNAMICS_METRICS:
        value_columns = {f"{metric}__{method}": method for method in PRIMARY_DYNAMICS_METHOD_ORDER}
        if missing := set(value_columns).difference(targets.columns):
            raise KeyError(f"Missing shared-LOTO metric columns: {sorted(missing)}")
        part = targets[id_columns + list(value_columns)].rename(columns=value_columns)
        part = part.melt(id_vars=id_columns, var_name="method", value_name="value")
        part["metric"] = metric
        long_parts.append(part)

    long_metrics = pd.concat(long_parts, ignore_index=True)
    if not np.isfinite(long_metrics["value"].to_numpy(float)).all():
        raise ValueError("Shared-LOTO fold metrics contain non-finite values.")
    fold_metrics = long_metrics.groupby(
        ["split_id", "target_group", "method", "metric"], as_index=False, sort=False
    ).agg(value=("value", "mean"), seed_count=("seed", "nunique"))
    fold_metrics = fold_metrics[
        ["split_id", "target_group", "method", "metric", "value", "seed_count"]
    ]
    expected_rows = 6 * len(PRIMARY_DYNAMICS_METHOD_ORDER) * len(DYNAMICS_METRICS)
    seed_count = targets["seed"].nunique()
    if (len(fold_metrics) != expected_rows or targets["seed"].isna().any()
            or not fold_metrics["seed_count"].eq(seed_count).all()):
        raise ValueError(
            f"Expected 6 folds x 7 methods x {len(DYNAMICS_METRICS)} metrics "
            f"with all {seed_count} input seeds per fold mean."
        )
    return fold_metrics


def build_tables(source_dir):
    """Return derived plotting tables; retain particle, fold and seed identifiers."""
    tables = {}

    def read(name):
        return pd.read_csv(source_dir / name, float_precision="round_trip")

    tables["dynamics_loto_fold_metrics.csv"] = _prepare_shared_loto_fold_metrics(read("dynamics_loto_targets.csv"))
    transition = read("full_data_transition_metrics.csv")
    rollout = read("full_data_rollout_metrics.csv")
    volume = read("full_data_volume_metrics.csv")
    keys = ["variant", "variant_label", "seed"]
    ts = transition.loc[transition.model_level.eq("reaction")].groupby(keys, as_index=False).agg(
        transition_sinkhorn=("sinkhorn_divergence", "mean"), transition_energy=("energy_distance", "mean"),
        transition_score_mae=("score_mean_mae", "mean"),
        transition_wasserstein_1=("wasserstein_1", "mean"), transition_count=("transition_id", "nunique"))
    rs = rollout.groupby(keys, as_index=False).agg(
        rollout_sinkhorn=("sinkhorn_divergence", "mean"), rollout_energy=("energy_distance", "mean"),
        rollout_score_mae=("score_mean_mae", "mean"),
        rollout_wasserstein_1=("wasserstein_1", "mean"), rollout_timepoint_count=("group_id", "nunique"))
    vs = volume.groupby(keys, as_index=False).agg(volume_mae=("absolute_error", "mean"),
        volume_rmse=("squared_error", lambda x: float(np.sqrt(x.mean()))), volume_bias=("signed_error", "mean"),
        volume_timepoint_count=("day", "count"))
    ablation = ts.merge(rs, on=keys).merge(vs, on=keys)
    assert len(ablation) == 15
    tables["full_data_ablation_by_seed.csv"] = ablation
    rows = []
    for values, part in volume.groupby(keys + ["treatment_clean"]):
        observed, predicted = part.observed_mean_volume.to_numpy(), part.predicted_volume.to_numpy()
        ss_res, ss_tot = float(np.sum((predicted-observed)**2)), float(np.sum((observed-observed.mean())**2))
        rows.append(dict(zip(keys + ["treatment_clean"], values), volume_r2=1-ss_res/ss_tot,
                         ss_res=ss_res, ss_tot=ss_tot, timepoint_count=part.day.nunique()))
    r2 = pd.DataFrame(rows).sort_values(["seed", "variant", "treatment_clean"]).reset_index(drop=True)
    assert len(r2) == 60 and r2.timepoint_count.eq(6).all()
    tables["full_data_volume_r2_by_treatment_seed.csv"] = r2
    return tables


def collect_sources(output_dir, volume_loto_dir, loss_ablation_dir, cell_loto_dir, dynamics_dir):
    source = output_dir / "source_data"
    source.mkdir(parents=True, exist_ok=True)

    def copy(path, name):
        if path.resolve() != (source / name).resolve():
            shutil.copy2(path, source / name)

    # The bundled table is the completed 48-fold paper evaluation.
    if volume_loto_dir is None:
        copy(WORK / "source_data/balanced_loto_predictions.csv", "balanced_loto_predictions.csv")
    else:
        volumes = pd.read_csv(volume_loto_dir / "combined/all_heldout_volume_metrics.csv")
        volumes = volumes.loc[volumes.day.isin(HELDOUT_DAY_ORDER) & volumes.model.eq("reaction")].copy()
        if len(volumes) != 48 or volumes.duplicated(["cv_seed", "treatment_clean", "day"]).any():
            raise ValueError("Expected 3 seeds x 4 treatments x 4 held-out days")
        volumes["source_run"] = volume_loto_dir.name
        volumes["source_fold"] = volumes["split_id"]
        volumes.to_csv(source / "balanced_loto_predictions.csv", index=False)

    for raw, name in [
        ("all_transition_distribution_metrics.csv", "full_data_transition_metrics.csv"),
        ("all_rollout_distribution_metrics.csv", "full_data_rollout_metrics.csv"),
        ("all_volume_metrics.csv", "full_data_volume_metrics.csv"),
    ]:
        frame = pd.read_csv(loss_ablation_dir / raw)
        frame = frame.loc[frame.variant.isin(FULL_DATA_ABLATION_ORDER)].copy()
        if name == "full_data_volume_metrics.csv":
            frame["signed_error"] = frame.predicted_volume - frame.observed_mean_volume
        frame["variant_label"] = frame.variant.map(VARIANT_LABELS)
        frame.to_csv(source / name, index=False)
    copy(cell_loto_dir / "paired_target_metrics.csv", "dynamics_loto_targets.csv")
    for relative, name in [
        ("pseudocell/tables/diffmap_sim_day1_6_pseudocells.csv.gz", "drift_background.csv.gz"),
        ("vtheta/plot_data/vtheta_dense_field.csv.gz", "drift_dense_field.csv.gz"),
        ("vtheta/source_data/pc_time_display_parameters.json", "drift_display_parameters.json"),
        ("robustness/dynamic_importance_by_delta.csv", "program_importance_by_delta.csv"),
    ]:
        copy(dynamics_dir / relative, name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=WORK)
    parser.add_argument("--volume-loto-dir", type=Path)
    parser.add_argument("--loss-ablation-dir", type=Path,
                        default=OUTPUTS / "training_loss_ablation/full_data_loss_all_seeds_combined")
    parser.add_argument("--cell-loto-dir", type=Path,
                        default=OUTPUTS / "cell_loto_benchmark/b1_shared_loto_all7_combined")
    parser.add_argument("--dynamics-dir", type=Path, default=REPO / "dynamics_analysis/outputs")
    args = parser.parse_args()
    collect_sources(args.output_dir, args.volume_loto_dir, args.loss_ablation_dir,
                    args.cell_loto_dir, args.dynamics_dir)
    tables = build_tables(args.output_dir / "source_data")
    table_dir = args.output_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        frame.to_csv(table_dir / name, index=False)
    print(f"Prepared {len(tables)} derived plotting tables in {table_dir}")


if __name__ == "__main__":
    main()
