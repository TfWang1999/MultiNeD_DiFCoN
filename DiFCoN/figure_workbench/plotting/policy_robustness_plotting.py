from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
        }
    )


from matplotlib.colors import LinearSegmentedColormap

def plot_temporal_attribution_summary(
    lag: pd.DataFrame, patient: pd.DataFrame, statistics: pd.DataFrame,
    figsize=(92 / 25.4, 62 / 25.4), font_size=7.0,
):
    """Temporal attribution + history perturbation."""
    configure_style()

    fig, axes = plt.subplots(1, 2, figsize=figsize, gridspec_kw={"width_ratios": [0.95, 1.0]})

    # A. Temporal attribution
    ax = axes[0]
    outputs = ["daily_total_logit", "pd1_fraction_logit"]
    lags = [-4, -3, -2, -1, 0]
    m = lag.pivot(index="output", columns="lag_from_current", values="integrated_gradient_fraction") \
           .reindex(index=outputs, columns=lags).to_numpy(float).T * 100

    cmap = LinearSegmentedColormap.from_list(
        "attr_blue", ["#F1F5F7", "#C9DDE7", "#82AEC5", "#4F86A8", "#2D6288"]
    )
    ax.imshow(m, cmap=cmap, aspect="auto", vmin=0, vmax=35)
    ax.set_box_aspect(1.75)

    for r in range(m.shape[0]):
        for c in range(m.shape[1]):
            ax.text(c, r, f"{m[r, c]:.0f}%", ha="center", va="center",
                    fontsize=font_size, color="black")

    ax.axhline(3.5, color="#6F767D", lw=0.7, ls="--")
    ax.set_xticks(range(2), ["Dose\namount", "PD-1\nallocation"])
    ax.set_yticks(range(5), [r"$x_{t-4}$", r"$x_{t-3}$", r"$x_{t-2}$", r"$x_{t-1}$", r"$x_t$"])
    # ax.set_ylabel("State history")
    ax.set_title("Temporal attribution", fontsize=font_size + 0.5, pad=4)
    ax.tick_params(axis="both", length=0, labelsize=font_size)

    ax.set_xticks(np.arange(-0.5, 2, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 5, 1), minor=True)
    ax.grid(which="minor", color="white", lw=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values(): spine.set_visible(False)

    # B. History perturbation
    ax = axes[1]
    order = ["current_only", "reverse_past"]
    labels = ["Current\nonly", "Past\nreversed"]
    colors = {"current_only": "#D77B39", "reverse_past": "#18378B"}
    rng = np.random.default_rng(20260901)
    ymax = float(patient["delta_loss"].max())
    ymin = min(-0.15, float(patient["delta_loss"].min()) - 0.03)

    for i, mode in enumerate(order):
        values = patient.loc[patient["history_intervention"].eq(mode), "delta_loss"].to_numpy(float)
        ax.scatter(np.full(len(values), i) + rng.uniform(-0.11, 0.11, len(values)), values,
                   s=18, color=colors[mode], alpha=0.4, linewidth=0, zorder=2)

        stat = statistics.loc[statistics["history_intervention"].eq(mode)].iloc[0]
        mean = float(stat["delta_loss_mean"])
        low = float(stat["bootstrap_mean_ci95_low"])
        high = float(stat["bootstrap_mean_ci95_high"])
        pval = float(stat["wilcoxon_two_sided_p"])

        ax.errorbar(i, mean, yerr=[[mean - low], [high - mean]], fmt="D", ms=4.8,
                    mfc="white", mec="#222222", mew=0.8, color="#222222",
                    capsize=2.8, elinewidth=1.0, zorder=5)
        ax.text(i, ymax + 0.025, f"$P$={pval:.1e}", ha="center", va="bottom",
                fontsize=font_size - 0.8, color="black")

    ax.axhline(0, color="#333333", lw=0.75, ls="--", zorder=1)
    ax.set_xticks(range(2), labels)
    ax.set_xlim(-0.45, 1.45)
    ax.set_ylim(ymin, ymax + 0.08)
    ax.set_ylabel(r"$\Delta \mathcal{L}_{\mathrm{strategy}}$")
    ax.set_title("History perturbation", fontsize=font_size + 0.5, pad=4)
    ax.tick_params(axis="both", labelsize=font_size)

    fig.tight_layout(w_pad=2.0)
    return fig
