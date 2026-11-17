"""Display-scale conversion for the paper's predicted and observed means."""
import numpy as np


def get_z_display_params(group_data, z_cols=None, q_low=0.01, q_high=0.99):
    """
    Compute robust min/max parameters for displaying zstd values on [0, 1].

    The stored values are z-score standardized values. For figures, robust01
    maps each z dimension's q_low--q_high range to 0--1 and clips outliers.
    """
    first_group = next(iter(group_data.values()))
    z_cols = z_cols or list(first_group["z_cols"])
    all_z = np.vstack([np.asarray(group["Z_all"]) for group in group_data.values()])
    params = {}
    for dim, col in enumerate(z_cols):
        lo, hi = np.quantile(all_z[:, dim], [q_low, q_high])
        if hi <= lo:
            lo, hi = float(all_z[:, dim].min()), float(all_z[:, dim].max())
        if hi <= lo:
            lo, hi = lo - 1.0, hi + 1.0
        params[col] = {"dim": dim, "lo": float(lo), "hi": float(hi)}
    return params


def transform_prediction_summary_for_display(
    prediction_summary,
    display_params,
):
    """
    Convert transition-level mean zstd summary to robust 0-1 display scale.

    Error columns are recomputed after transforming source/predicted/target
    means, so they remain interpretable in display units.
    """
    out = prediction_summary.copy()
    for idx, row in out.iterrows():
        z_col = row["z_col"]
        p = display_params[z_col]
        denom = p["hi"] - p["lo"]
        for col in ("source_mean", "predicted_mean", "target_mean"):
            out.at[idx, col] = np.clip((row[col] - p["lo"]) / denom, 0.0, 1.0)
    out["error"] = out["predicted_mean"] - out["target_mean"]
    out["abs_error"] = np.abs(out["error"])
    return out
