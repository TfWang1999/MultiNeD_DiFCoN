"""Policy supplement panels drawn from the prepared evaluation tables."""

from contextlib import contextmanager

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

from .editing import ARIAL_MATH, PROGRAM_LABELS, style_axis_text
from .model_style import figure_style

COLORS = {
    "ink": "#2B2F33", "grid": "#E6E9EC", "blue": "#4C78A8",
    "teal": "#4C9A91", "orange": "#D7904E", "red": "#C45D62",
}

FEATURE_ORDER = (
    # Immune programs
    "score_precursor", "score_activation", "score_eff_early", "score_eff_cytox", "score_dysfunction",
    # Tumor burden, CD8 cell count and effector activity
    "log_C_abs", "log_C_rel", "log_N_abs", "log_N_rel", "activity_scaled",
    # Drug concentrations and previous actions
    "P_conc_scaled", "B_conc_scaled", "previous_action_pd1", "previous_action_b41bb",
    # Budget and time
    "remaining_budget_fraction", "time_fraction",
)
# Map data column names to display labels.
FEATURE_LABELS = {
    "score_precursor": r"$P_{\mathrm{pre}}$",
    "score_activation": r"$A_{\mathrm{act}}$",
    "score_eff_early": r"$E_{\mathrm{early}}$",
    "score_eff_cytox": r"$Y_{\mathrm{cyto}}$",
    "score_dysfunction": r"$D_{\mathrm{dysf}}$",
    "previous_action_pd1": r"$u_{P}(t-1)$",
    "previous_action_b41bb": r"$u_{B}(t-1)$",
    "log_C_abs": r"$\log(1+C)/\log(1+C_{\mathrm{ref}})$",
    "log_C_rel": r"$\log(C/C_0)$",
    "log_N_abs": r"$\log(1+N)/\log(1+N_{\mathrm{ref}})$",
    "log_N_rel": r"$\log(N/N_0)$",
    "activity_scaled": r"$\log(1+a)/\log(1+a_{\mathrm{ref}})$",
    "P_conc_scaled": r"$P/P_{\mathrm{cap}}$",
    "B_conc_scaled": r"$B/B_{\mathrm{cap}}$",
    "remaining_budget_fraction": r"$b_{\mathrm{remaining}}/b_{\mathrm{total}}$",
    "time_fraction": r"$(t-t_0)/(T-t_0)$",
}


@contextmanager
def _policy_style(font_size):
    with figure_style(font_size), mpl.rc_context({**ARIAL_MATH, "mathtext.cal": "STIXGeneral:italic"}):
        yield


def plot_training_history_overview(history, *, figsize, font_size=10):
    panels = (
        (("train_loss", "validation_loss"), ("Train", "Validation"), r"Objective, $\mathcal{L}_{\mathrm{strategy}}$"),
        (("train_final_tumor_loss", "train_tumor_auc_loss", "train_functional_activity_loss", "train_total_dose_loss"),
         (r"$c(T)$", r"$\mathrm{CTB}$", r"$\mathcal{L}_F$", r"$D_{\mathrm{tot}}/D_{\mathrm{ref}}$"), "Loss component"),
    )
    colors = [COLORS[key] for key in ("blue", "red", "teal", "orange")]
    with _policy_style(font_size):
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        for axis, (columns, labels, ylabel) in zip(axes, panels):
            for column, label, color in zip(columns, labels, colors):
                trace = history[["epoch", column]].dropna().sort_values("epoch")
                axis.plot(trace["epoch"], trace[column], color=color, label=label, lw=1.1,
                          marker="o" if column == "validation_loss" else None, ms=2.2)
            axis.set(xlabel="Epoch", ylabel=ylabel)
            axis.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
            axis.yaxis.set_major_locator(MaxNLocator(nbins=3))
            axis.grid(axis="y", color=COLORS["grid"], lw=0.45)
            style_axis_text(axis, label_size=font_size, tick_size=9, title_size=font_size)
        axes[0].legend(loc="upper right", fontsize=font_size)
        axes[1].legend(loc="lower center", bbox_to_anchor=(0.5, 0.10), ncol=4,
                       fontsize=font_size, handlelength=1.4, handletextpad=0.4, columnspacing=1.0)
        fig.subplots_adjust(left=0.08, right=0.99, bottom=0.23, top=0.97, wspace=0.24)
    return fig


def final_score_panel(summary, *, figsize, font_size=10):
    scores = ["final_score_precursor", "final_score_activation", "final_score_eff_early",
              "final_score_eff_cytox", "final_score_dysfunction"]
    means = summary.groupby("strategy")[scores].mean()
    x = np.arange(len(scores))
    with _policy_style(font_size):
        fig, axis = plt.subplots(figsize=figsize)
        for offset, strategy, label, color in (
            (-0.17, "Combine", "Combination", "#6F9D7C"),
            (0.17, "Optimal", "DiFCoN", "#C95E66"),
        ):
            axis.bar(x + offset, means.loc[strategy], width=0.34, label=label, color=color, alpha=0.85)
        axis.set_xticks(x, PROGRAM_LABELS, rotation=0, ha="center")
        axis.set(ylabel=r"Mean final score, $s(\bar z_j)$", title="Final immune-program scores")
        axis.grid(axis="y", color=COLORS["grid"], lw=0.55)
        axis.legend(loc="upper left", fontsize=9)
        fig.tight_layout()
    style_axis_text(axis, label_size=font_size, tick_size=9, title_size=font_size)
    axis.yaxis.set_major_locator(MaxNLocator(nbins=3))
    fig.tight_layout(pad=0.45)
    return fig


def plot_multiseed_metric_overview(data, *, figsize, font_size=10):
    """Show each seed's patient distribution and mean."""
    metrics = (
        ("final_C_rel", r"$c(T)$"), ("tumor_auc_rel", r"$\mathrm{CTB}$"),
        ("total_dose", r"$D_{\mathrm{tot}}$"), ("loss", r"$\mathcal{L}_{\mathrm{strategy}}$"),
    )
    seeds = (2031, 2032, 2033)
    colors = [COLORS[key] for key in ("blue", "orange", "teal")]
    with _policy_style(font_size):
        fig, axes = plt.subplots(1, 4, figsize=figsize)
        for axis, (metric, label) in zip(axes, metrics):
            values = [data.loc[data["model_seed"].eq(seed), metric].to_numpy(float) for seed in seeds]
            violin = axis.violinplot(values, positions=range(3), widths=0.72,
                                     showmeans=True, showextrema=False)
            for body, color in zip(violin["bodies"], colors):
                body.set(facecolor=color, edgecolor=color, alpha=0.72, linewidth=0.5)
            violin["cmeans"].set_segments([
                [(j - 0.20, v.mean()), (j + 0.20, v.mean())] for j, v in enumerate(values)
            ])
            violin["cmeans"].set(color=COLORS["ink"], linewidth=1.35, zorder=4)
            axis.set_xticks([1], [label])
            axis.tick_params(axis="x", length=0, pad=7)
            axis.set_xlim(-0.65, 2.65)
            axis.yaxis.set_major_locator(MaxNLocator(nbins=3))
            axis.grid(axis="y", color=COLORS["grid"], lw=0.45, zorder=0)
            style_axis_text(axis, label_size=font_size, tick_size=9, title_size=font_size)
        handles = [Patch(facecolor=color, alpha=0.72, label=f"Seed {seed}") for seed, color in zip(seeds, colors)]
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.52, 0.98), ncol=3, fontsize=9)
        fig.subplots_adjust(left=0.07, right=0.99, bottom=0.22, top=0.78, wspace=0.50)
    return fig


def plot_feature_attribution(feature, *, figsize, font_size=9.5, cmap="coolwarm"):
    """Show the 16 saved integrated-gradient values on a logarithmic color scale."""
    pivot = feature.pivot(index="feature", columns=["output", "history_role"],
                          values="absolute_integrated_gradient_mean")
    pivot = pivot.loc[list(FEATURE_ORDER)].sort_index(axis=1).fillna(0.0)
    matrix = pivot.to_numpy(dtype=float)
    norm = LogNorm(vmin=float(matrix[matrix > 0].min()), vmax=max(1e-12, float(matrix.max())))
    blocks = [f"{'Total' if output == 'daily_total_logit' else 'PD-1 allocation'}\n{role}"
              for output, role in pivot.columns]
    with _policy_style(font_size):
        fig, axis = plt.subplots(figsize=figsize)
        image = axis.imshow(matrix.T, aspect="auto", cmap=cmap, norm=norm)
        axis.set_xticks(range(len(FEATURE_ORDER)), [FEATURE_LABELS[key] for key in FEATURE_ORDER],
                        rotation=90, ha="center")
        axis.set_yticks(range(len(blocks)), blocks)
        colorbar = fig.colorbar(image, ax=axis, orientation="horizontal", location="top",
                               fraction=0.08, pad=0.16, aspect=60)
        ticks = (0.01, 0.1, 1, 10, 50)
        colorbar.set_ticks(ticks, labels=[f"{tick:g}" for tick in ticks])
        # Preserve the spacing of the original layout pass before hiding the label.
        colorbar.set_label("Absolute integrated-gradient attribution (log scale)")
        fig.tight_layout()
    style_axis_text(axis, label_size=10, tick_size=9, title_size=10)
    colorbar.ax.set_xlabel("")
    style_axis_text(colorbar.ax, label_size=font_size, tick_size=9)
    fig.tight_layout(pad=0.45)
    return fig


def openloop_mean_test_trajectories_panel(grouped, *, figsize, font_size=10):
    """Plot the saved means and SEM for the three dosing approaches."""
    strategies = (
        ("Fixed Combine", "Fixed combination", "o", "#6F9D7C"),
        ("Validation-derived open-loop", "Open-loop", "s", "#4C78A8"),
        ("Closed-loop", "Closed-loop", "^", "#C95E66"),
    )
    with _policy_style(font_size):
        fig, axis = plt.subplots(figsize=figsize)
        for strategy, label, marker, color in strategies:
            frame = grouped.loc[grouped["strategy"].eq(strategy)].sort_values("day")
            mean = frame["mean"].to_numpy(dtype=float)
            sem = frame["sem"].fillna(0.0).to_numpy(dtype=float)
            day = frame["day"].to_numpy(dtype=float)
            axis.plot(day, mean, lw=1.5, label=label, color=color, marker=marker, markersize=3.0,
                      markevery=max(1, len(day) // 9), markeredgecolor="white", markeredgewidth=0.35)
            axis.fill_between(day, mean - sem, mean + sem, color=color, alpha=0.1, linewidth=0)
        axis.set(xlabel="Day", ylabel=r"Mean tumor burden, $C(t)/C_0$")
        xmin = int(np.floor(grouped["day"].min()))
        xmax = int(np.ceil(grouped["day"].max()))
        axis.set_xticks(np.arange(xmin, xmax + 1, 2))
        fig.tight_layout()
    axis.set_title("Predicted tumor trajectories in the test cohort")
    axis.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    style_axis_text(axis, label_size=font_size, tick_size=9, title_size=font_size)
    fig.tight_layout(pad=0.45)
    return fig


def plot_validation_openloop_lines(schedules, *, figsize, font_size=10):
    """Show three-seed schedule means and sample standard deviations."""
    drugs = (
        ("action_PD1", "anti-PD-1", COLORS["blue"], "o"),
        ("action_B41BB", "anti-4-1BB", COLORS["orange"], "s"),
    )
    days = np.sort(schedules["day"].unique())
    with _policy_style(font_size):
        fig, axis = plt.subplots(figsize=figsize)
        for column, label, color, marker in drugs:
            values = schedules.pivot(index="model_seed", columns="day", values=column).reindex(columns=days)
            axis.errorbar(days, values.mean(), yerr=values.std(ddof=1),
                          marker=marker, markersize=3.5, linewidth=1.2, color=color,
                          capsize=2, elinewidth=0.85, markeredgecolor="white", markeredgewidth=0.35, label=label)
        tick_days = days[::3]
        axis.set_xticks(tick_days, [f"{day:g}" for day in tick_days])
        axis.set(xlabel="Decision day", ylabel="Mean daily dose",
                 xlim=(days.min() - 0.5, days.max() + 0.5), ylim=(-0.025, None))
        axis.yaxis.set_major_locator(MaxNLocator(nbins=4))
        axis.grid(axis="y", color=COLORS["grid"], lw=0.45)
        axis.set_axisbelow(True)
        axis.legend(loc="upper right", ncol=1, fontsize=9)
        fig.subplots_adjust(left=0.18, right=0.97, bottom=0.22, top=0.96)
    style_axis_text(axis, label_size=font_size, tick_size=9, title_size=font_size)
    return fig


def plot_locked_seed_stability(seed_metrics, *, figsize, font_size=10):
    """Show individual seed means and their cross-seed sample standard deviation."""
    methods = {
        "Amortized 1D-CNN": ("DiFCoN", COLORS["red"]),
        "Diagonal CMA-ES": ("CMA-ES", "#8064A2"),
        "SAC": ("SAC", COLORS["teal"]),
        "Validation-derived open loop": ("Fixed open-loop schedule", COLORS["blue"]),
    }
    metrics = (
        ("loss", "Composite objective"), ("final_C_rel", "Final relative tumor"),
        ("tumor_auc_rel", "Cumulative tumor burden"), ("total_dose", "Total dose"),
    )
    colors = [color for _, color in methods.values()]
    with _policy_style(font_size):
        fig, axes = plt.subplots(1, 4, figsize=figsize)
        for axis, (metric, label) in zip(axes, metrics):
            values = seed_metrics.pivot(index="model_seed", columns="strategy", values=metric).reindex(columns=list(methods))
            axis.bar(range(4), values.mean(), yerr=values.std(ddof=1),
                     width=0.60, color=colors, alpha=0.80, capsize=2.2,
                     error_kw={"ecolor": COLORS["ink"], "elinewidth": 0.85, "capthick": 0.85})
            offsets = np.linspace(-0.12, 0.12, len(values))
            for j, strategy in enumerate(methods):
                axis.scatter(j + offsets, values[strategy], s=9, color=COLORS["ink"],
                             edgecolors="white", linewidths=0.35, alpha=0.9, zorder=4)
            axis.set_xticks([1.5], [label])
            axis.tick_params(axis="x", length=0, pad=7)
            axis.tick_params(axis="y", pad=1)
            axis.set_xlim(-0.65, 3.65)
            axis.set_ylim(bottom=0)
            axis.yaxis.set_major_locator(MaxNLocator(nbins=3))
            axis.grid(axis="y", color=COLORS["grid"], lw=0.45)
            axis.set_axisbelow(True)
            style_axis_text(axis, label_size=font_size, tick_size=9, title_size=font_size)
        handles = [Patch(facecolor=color, alpha=0.80, label=label) for label, color in methods.values()]
        fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.96), fontsize=9)
        fig.subplots_adjust(left=0.06, right=0.99, top=0.82, bottom=0.18, wspace=0.32)
    return fig
