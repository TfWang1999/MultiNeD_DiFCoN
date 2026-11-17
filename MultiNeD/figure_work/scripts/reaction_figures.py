"""Local model plotting functions. Inputs are prepared tables; no model runtime."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from .constants import (
    COMPONENT_ABLATION_ORDER,
    FULL_DATA_ABLATION_LABELS,
    METHOD_LABELS,
    PRIMARY_DYNAMICS_METHOD_ORDER,
    TREATMENT_COLORS,
    TREATMENT_LABELS,
    TREATMENT_ORDER,
)
from .figure_style import apply_nature_style, _size, _set_limits, _style_axis

def _loto_summary(loto):
    return loto.loc[loto["model"].eq("reaction")].groupby(
        ["treatment_clean", "day"], as_index=False
    ).agg(
        observed=("observed_mean_volume", "first"),
        predicted_mean=("predicted_volume", "mean"),
        predicted_sd=("predicted_volume", "std"),
    )


def plot_state_reconstruction(prediction_summary, cfg, z_cols):
    """Plot predicted versus target CD8 program means from a prepared rollout summary."""
    cur = prediction_summary.loc[prediction_summary["z_col"].isin(z_cols)]
    if cur.empty:
        raise ValueError("No state-reconstruction rows match the requested programs.")
    colors, markers = cfg["treatment_colors"], cfg["program_markers"]
    fig, ax = plt.subplots(figsize=cfg["figsize"])
    for z_col in z_cols:
        for treatment in TREATMENT_ORDER:
            part = cur[(cur["z_col"] == z_col) & (cur["treatment_clean"] == treatment)]
            if not part.empty:
                ax.scatter(part["target_mean"], part["predicted_mean"], s=cfg["marker_size"] ** 2, color=colors[treatment],
                           marker=markers.get(z_col, "o"), edgecolor="white", linewidth=0.4, alpha=0.95)
    x, y = cur["target_mean"].to_numpy(float), cur["predicted_mean"].to_numpy(float)
    lo, hi = min(x.min(), y.min()), max(x.max(), y.max()); pad = 0.05 * (hi - lo if hi > lo else 1.0); limits = (lo - pad, hi + pad)
    ax.plot(limits, limits, linestyle="--", color=cfg["identity_color"], linewidth=cfg["line_width"])
    slope, intercept = np.polyfit(x, y, 1) if x.size >= 2 and np.std(x) > 0 else (np.nan, np.nan)
    y_fit = slope * x + intercept; ss_res, ss_tot = np.sum((y - y_fit) ** 2), np.sum((y - y.mean()) ** 2)
    r2, mae = (1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan), float(np.mean(np.abs(y - x)))
    if np.isfinite(slope):
        ax.plot(limits, [slope * limits[0] + intercept, slope * limits[1] + intercept], color=cfg["fit_color"], linewidth=cfg["line_width"])
    if cfg.get("xlim") is None: ax.set_xlim(limits)
    if cfg.get("ylim") is None: ax.set_ylim(limits)
    _style_axis(ax, cfg)
    ax.text(*cfg["metric_text_xy"], rf"$R^2={r2:.3f}$" + "\n" + rf"$\mathrm{{MAE}}={mae:.3f}$", transform=ax.transAxes,
            ha="left", va="top", fontsize=cfg["legend_size"])
    th = [Line2D([], [], marker="o", linestyle="none", color=colors[t], label=TREATMENT_LABELS[t], markersize=cfg["marker_size"])
          for t in TREATMENT_ORDER if t in set(cur["treatment_clean"])]
    tl = ax.legend(handles=th, title="Treatment", frameon=False, fontsize=cfg["legend_size"], title_fontsize=cfg["legend_size"], loc=cfg["legend_loc"])
    ax.add_artist(tl)
    ph = [Line2D([], [], marker=markers.get(z, "o"), linestyle="none", color="#666666", label=cfg["program_labels"].get(z, z),
                 markersize=cfg["marker_size"]) for z in z_cols]
    ax.legend(handles=ph, title="Program", frameon=False, fontsize=cfg["legend_size"], title_fontsize=cfg["legend_size"], loc=cfg["program_legend_loc"])
    fig.tight_layout(pad=0.5)
    return fig


def plot_loto_predictions(loto, cfg):
    """Plot strict LOTO observed targets and fold predictions on one axis."""
    summary = _loto_summary(loto)
    days, colors = cfg["days"], cfg["treatment_colors"]
    positions = np.arange(len(days), dtype=float); offsets = dict(zip(TREATMENT_ORDER, np.linspace(-0.30, 0.30, len(TREATMENT_ORDER))))
    fig, ax = plt.subplots(figsize=cfg["figsize"])
    for treatment in TREATMENT_ORDER:
        part = summary[summary["treatment_clean"] == treatment].set_index("day").reindex(days); x = positions + offsets[treatment]
        observed, predicted, sd = part["observed"].to_numpy(float), part["predicted_mean"].to_numpy(float), part["predicted_sd"].to_numpy(float)
        ax.vlines(x, observed, predicted, color=colors[treatment], alpha=0.32, linewidth=cfg["line_width"], zorder=1)
        ax.scatter(x, observed, s=cfg["marker_size"] ** 2, marker="o", facecolor="white", edgecolor=colors[treatment], linewidth=cfg["line_width"], zorder=4)
        ax.errorbar(x, predicted, yerr=sd, fmt="s", color=colors[treatment], ecolor=colors[treatment], markersize=cfg["marker_size"],
                    markerfacecolor=colors[treatment], markeredgecolor="white", markeredgewidth=0.35, elinewidth=cfg["line_width"], capsize=1.5, zorder=5)
    ax.set_xticks(positions, [str(day) for day in days]); ax.grid(axis="y", color="0.92", linewidth=0.55, zorder=0); _style_axis(ax, cfg)
    th = [Line2D([], [], marker="o", linestyle="none", color=colors[t], label=TREATMENT_LABELS[t], markersize=cfg["marker_size"]) for t in TREATMENT_ORDER]
    tl = ax.legend(handles=th, title="Treatment", frameon=False, fontsize=cfg["legend_size"], title_fontsize=cfg["legend_size"], loc=cfg["legend_loc"])
    ax.add_artist(tl)
    types = [Line2D([], [], marker="o", linestyle="none", markerfacecolor="white", markeredgecolor="#555555", label="Observed"),
             Line2D([], [], marker="s", linestyle="none", markerfacecolor="#555555", markeredgecolor="white", label="Held-out prediction")]
    ax.legend(handles=types, frameon=False, fontsize=cfg["legend_size"], loc=cfg.get("type_legend_loc", "upper center")); fig.tight_layout(pad=0.5)
    return fig


def plot_full_data_component_ablation_bars(ablation, volume_r2, cfg):
    """Horizontal barplot for full-data component ablations."""

    metric_specs = cfg["metric_specs"]

    order = COMPONENT_ABLATION_ORDER
    labels = [FULL_DATA_ABLATION_LABELS[v] for v in order]
    y = np.arange(len(order))
    variant_alpha = cfg.get("variant_alpha", 0.70)
    treatment_alpha = cfg.get("treatment_alpha", 0.80)

    fig, axes = plt.subplots(
        1, 4, figsize=cfg["figsize"], sharey=False,
        gridspec_kw={"width_ratios": [1.15, 1, 1, 1.15]}
    )

    for i, (ax, (metric, title, xlabel)) in enumerate(zip(axes[:3], metric_specs)):
        stats = ablation.groupby("variant")[metric].agg(["mean", "std"]).reindex(order)

        ax.barh(
            y, stats["mean"], xerr=stats["std"], height=0.55,
            color=[cfg["variant_colors"][v] for v in order],
            alpha=variant_alpha, edgecolor="none",
            error_kw={"elinewidth": cfg["line_width"], "capsize": 2.5}
        )

        ax.set_yticks(y)
        ax.set_yticklabels(labels if i == 0 else [""] * len(labels))
        ax.invert_yaxis()

        _style_axis(ax, cfg, title=title, xlabel=xlabel, ylabel=False)
        ax.spines["left"].set_visible(True)
        ax.tick_params(axis="y", length=3, labelleft=True)
        ax.grid(False)

    ax = axes[3]
    offsets = np.linspace(-0.3, 0.3, len(TREATMENT_ORDER))
    bar_height = 0.18

    for offset, treatment in zip(offsets, TREATMENT_ORDER):
        stats = (
            volume_r2.loc[volume_r2["treatment_clean"].eq(treatment)]
            .groupby("variant")["volume_r2"]
            .agg(["mean", "std"])
            .reindex(order)
        )

        ax.barh(
            y + offset, stats["mean"], xerr=stats["std"], height=bar_height,
            color=cfg["treatment_colors"][treatment],
            alpha=treatment_alpha, edgecolor="none",
            error_kw={"elinewidth": cfg["line_width"], "capsize": 1.8}
        )

    ax.set_yticks(y)
    ax.set_yticklabels([""] * len(labels))
    ax.invert_yaxis()

    _style_axis(ax, cfg, title="Volume reconstruction", xlabel="$R^2$", ylabel=False)
    _set_limits(ax, xlim=cfg["r2_xlim"])
    ax.set_xticks(cfg["r2_ticks"])
    ax.spines["left"].set_visible(True)
    ax.tick_params(axis="y", length=3, labelleft=True)
    ax.grid(False)

    handles = [
        Patch(facecolor=cfg["treatment_colors"][t], edgecolor="none",
              alpha=treatment_alpha, label=TREATMENT_LABELS[t])
        for t in TREATMENT_ORDER
    ]

    ax.legend(
        handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5),
        title="Treatment", title_fontsize=cfg["legend_size"],
        fontsize=cfg["legend_size"], frameon=False
    )

    if cfg.get("title"):
        fig.suptitle(cfg["title"], x=0.08, y=0.99, ha="left",
                     fontsize=cfg["title_size"], fontweight="bold")

    fig.subplots_adjust(
        left=0.12, right=0.88, top=0.90, bottom=0.18,
        wspace=cfg.get("wspace", 0.28)
    )

    return fig


def _set_day_xaxis(ax, days):
    days = days[np.isfinite(days)]
    if days.size:
        first_day = min(float(days.min()), 1.0)
        ticks = np.arange(np.floor(first_day), np.ceil(days.max()) + 1.0, 1.0)
        if 1.0 not in ticks:
            ticks = np.sort(np.append(ticks, 1.0))
    else:
        ticks = np.array([1.0])
    left, right = ax.get_xlim()
    ax.set_xlim(min(left, float(ticks.min())), max(right, float(ticks.max())))
    ax.set_xticks(ticks, [str(int(day)) for day in ticks])


def _add_dose_lines(ax, doses, label_fontsize):
    colors = {"Anti-PD1": "#1f77b4", "Anti-41BB": "#2ca02c"}
    dose_times = doses[["drug", "day"]].drop_duplicates()
    for drug, day in dose_times.itertuples(index=False, name=None):
        color = colors[drug]
        is_pd1 = drug == "Anti-PD1"
        ax.axvline(day, color=color, linestyle="--" if is_pd1 else ":", linewidth=1.15, alpha=1)
        ax.text(
            day, 1.025, "P" if is_pd1 else "B",
            transform=ax.get_xaxis_transform(), color=color, fontsize=label_fontsize,
            va="bottom", ha="center", clip_on=False,
        )


def plot_tumour_reconstruction(volume, cfg, dose_data):
    """Plot predicted tumour volume and observed mean with SEM for each treatment."""
    present = set(volume["treatment_clean"])
    treatments = [treatment for treatment in TREATMENT_ORDER if treatment in present]
    colors = cfg["treatment_colors"]
    markers = {"Control": "o", "Anti-PD1": "^", "Anti-41BB": "s", "Anti-PD1+Anti-41BB": "D"}
    line_width, marker_size = cfg["line_width"], cfg["marker_size"]
    legend_fontsize = cfg["legend_size"]
    fig, ax = plt.subplots(figsize=cfg["figsize"])
    r2_by_treatment = {}
    for treatment in treatments:
        part = volume.loc[volume["treatment_clean"].eq(treatment)]
        pred = part[part["kind"] == "predicted"].sort_values("time")
        obs = part[part["kind"] == "observed"].copy()
        color = colors[treatment]
        if not pred.empty:
            ax.plot(
                pred["time"],
                pred["volume"],
                color=color,
                linewidth=line_width,
                label=f"{treatment} predicted",
            )
        if not obs.empty:
            obs_summary = (
                obs.groupby("time")["volume"]
                .agg(mean="mean", std="std", n="count")
                .reset_index()
                .sort_values("time")
            )
            obs_summary["sem"] = obs_summary["std"].fillna(0.0) / np.sqrt(obs_summary["n"].clip(lower=1))
            ax.errorbar(
                obs_summary["time"],
                obs_summary["mean"],
                yerr=obs_summary["sem"],
                color=color,
                linestyle="none",
                marker=markers[treatment],
                markerfacecolor=color,
                markeredgecolor="white",
                markeredgewidth=0.4,
                markersize=marker_size,
                capsize=1.5,
                elinewidth=0.7,
                label=f"{treatment} observed mean +/- SEM",
            )
            if not pred.empty:
                pred_summary = (
                    pred.groupby("time")["volume"]
                    .mean()
                    .reset_index()
                    .sort_values("time")
                )
                x_pred = pred_summary["time"].to_numpy(dtype=float)
                y_pred = pred_summary["volume"].to_numpy(dtype=float)
                x_obs = obs_summary["time"].to_numpy(dtype=float)
                y_obs = obs_summary["mean"].to_numpy(dtype=float)
                if x_pred.size >= 1 and y_obs.size >= 2:
                    y_fit = np.interp(x_obs, x_pred, y_pred)
                    ss_res = float(np.sum((y_obs - y_fit) ** 2))
                    ss_tot = float(np.sum((y_obs - float(np.mean(y_obs))) ** 2))
                    r2_by_treatment[treatment] = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
                else:
                    r2_by_treatment[treatment] = np.nan
    _add_dose_lines(ax, dose_data, cfg.get("dose_font_size", legend_fontsize))
    label_fontsize = _size(cfg, "xlabel_size", "label_size")
    ax.set_ylabel(cfg["ylabel"], fontsize=label_fontsize)
    ax.set_xlabel(cfg["xlabel"], fontsize=label_fontsize)
    ax.tick_params(axis="both", labelsize=_size(cfg, "xtick_size", "tick_size"))
    _set_day_xaxis(ax, volume["time"].to_numpy(dtype=float))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    legend_ax = ax.inset_axes(cfg["legend_position"])
    legend_ax.set_axis_off()
    legend_ax.set_xlim(0.0, 1.0)
    legend_ax.set_ylim(0.0, 1.0)
    legend_ax.patch.set_alpha(0.0)
    block_h = 1.0 / max(1, len(treatments))
    for i, treatment in enumerate(treatments):
        color = colors[treatment]
        marker = markers[treatment]
        y_top = 1.0 - i * block_h
        y_pred = y_top - 0.25 * block_h
        y_obs = y_top - 0.75 * block_h
        y_mid = 0.5 * (y_pred + y_obs)
        display_name = TREATMENT_LABELS[treatment]
        legend_ax.text(0.00, y_mid, f"{display_name}:", va="center", ha="left", fontsize=legend_fontsize, color="0.15")
        legend_ax.plot([0.45, 0.57], [y_pred, y_pred], color=color, linewidth=line_width, solid_capstyle="round")
        legend_ax.text(0.60, y_pred, "predicted", va="center", ha="left", fontsize=legend_fontsize, color="0.15")
        legend_ax.plot(
            [0.51],
            [y_obs],
            color=color,
            marker=marker,
            linestyle="none",
            markersize=marker_size,
            markeredgewidth=0.5,
            markeredgecolor="white",
        )
        legend_ax.text(0.60, y_obs, "observed", va="center", ha="left", fontsize=legend_fontsize, color="0.15")
        r2 = r2_by_treatment.get(treatment, np.nan)
        r2_text = r"$R^2=\mathrm{NA}$" if not np.isfinite(r2) else rf"$R^2={r2:.3f}$"
        legend_ax.text(1.3, y_mid, r2_text, va="center", ha="right", fontsize=legend_fontsize, color="0.15")
    _style_axis(ax, cfg)
    return fig


def plot_tcell_activity_four_treatments(volume_table):
    font_size = 8
    apply_nature_style(font_size=font_size)
    pred = volume_table.loc[volume_table["kind"].eq("predicted")]
    fig, axes = plt.subplots(1, 2, figsize=(115 / 25.4, 55 / 25.4), sharex=True)
    for treatment in TREATMENT_ORDER:
        part = pred[pred["treatment_clean"] == treatment].sort_values("day")
        if part.empty:
            continue
        color = TREATMENT_COLORS[treatment]
        axes[0].plot(part["day"], part["N"], lw=1.0, color=color, label=treatment)
        axes[1].plot(part["day"], part["N"] * part["effector_activity"], lw=1.0, color=color, label=treatment)
    axes[0].set_ylabel(r"CD8 abundance $N(t)$", fontsize=font_size)
    axes[1].set_ylabel(r"Total activity $A_{\mathrm{CD8}}(t)$", fontsize=font_size)
    for ax in axes:
        ax.set_xlabel("Time (days)", fontsize=font_size)
        ax.set_xlim(1, 16)
        ax.set_xticks([1, 4, 7, 10, 13, 16])
        ax.tick_params(axis="both", labelsize=font_size)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for spine in ax.spines.values():
            spine.set_linewidth(0.7)
        ax.tick_params(width=0.6, length=2.8, pad=2)
    axes[0].set_title("CD8 abundance", fontsize=font_size, pad=4)
    axes[1].set_title("Total CD8 activity", fontsize=font_size, pad=4)
    axes[0].legend(
        labels=["Control", "anti-PD-1", "anti-4-1BB", "Combination"],
        frameon=False,
        loc="best",
        fontsize=font_size,
        ncol=1,
        handlelength=1.4,
    )
    fig.tight_layout()
    return fig


def plot_dynamics_benchmark_bars(fold_metrics, cfg):
    """Plot fold means and SEM, with each held-out fold shown as a point."""
    specs = cfg["metric_specs"]
    methods = PRIMARY_DYNAMICS_METHOD_ORDER
    reference = "project_balanced_sinkhorn_drift"
    missing_methods = set(methods) - set(fold_metrics["method"])
    if missing_methods:
        raise ValueError(f"Missing benchmark methods: {sorted(missing_methods)}")

    # Compute each method's mean, standard deviation and SEM across folds.
    stats = fold_metrics.groupby(["method", "metric"])["value"].agg(
        mean="mean", sd="std", n="count").reset_index()
    stats["sem"] = stats["sd"] / np.sqrt(stats["n"])

    # Sort methods by the selected metric.
    sort_values = stats.loc[stats["metric"].eq("sinkhorn_divergence")].set_index("method")["mean"]
    order = sorted(methods, key=lambda m: sort_values[m])

    # Create horizontal panels with a shared method axis.
    fig, axes = plt.subplots(1, len(specs), figsize=cfg["figsize"], sharey=True)
    y = np.arange(len(order))

    for ax, (metric, xlabel) in zip(axes, specs):
        part = stats.query("metric == @metric").set_index("method").loc[order]
        colors = [cfg["method_colors"][m] for m in order]

        # Horizontal bars show means, and error bars show SEM.
        ax.barh(
            y, part["mean"], xerr=part["sem"], height=0.62,
            color=colors, alpha=0.55, edgecolor=colors, linewidth=0.7, zorder=2,
            error_kw={"elinewidth": cfg["line_width"], "capsize": 2,
                      "capthick": cfg["line_width"], "ecolor": "#303030"},
        )

        # Overlay the six fold values on each method's bar.
        for yi, method in zip(y, order):
            values = fold_metrics.query("method == @method and metric == @metric").sort_values(["split_id", "target_group"])["value"].to_numpy(float)
            offsets = np.linspace(-0.18, 0.18, len(values))
            ax.scatter(values, yi + offsets, s=(cfg["marker_size"] * 0.9) ** 2, facecolor="white",
                    edgecolor=cfg["method_colors"][method], linewidth=0.7, alpha=0.9, zorder=4)

        # Draw a vertical dashed line at the reference method's mean.
        ax.axvline(
            part.loc[reference, "mean"], color=cfg["method_colors"][reference],
            linestyle=(0, (3, 2)), linewidth=cfg["line_width"], alpha=0.9, zorder=1,
        )

        # Configure the axes and hide the grid.
        ax.set_xlabel(xlabel, fontsize=cfg["label_size"]); ax.grid(False); ax.tick_params(axis="y", length=0)
        for spine in ("top", "right"): ax.spines[spine].set_visible(False)
        ax.spines["left"].set_visible(True)

    # Label methods with the first-ranked method at the top.
    axes[0].set_yticks(y, [METHOD_LABELS[method] for method in order], fontsize=cfg["label_size"])
    axes[0].invert_yaxis()
    if cfg.get("title"): fig.suptitle(cfg["title"], fontsize=cfg["label_size"])
    fig.tight_layout(pad=0.7)
    return fig
