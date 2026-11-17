"""Plot the manuscript's policy panels from saved experiment results."""

from __future__ import annotations

from contextlib import contextmanager

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize

from . import baseline_visualization
from ._paths import BASELINE_DIR
from ._source_tables import read_source_table


COLORS = {
    "ink": "#2B2F33",
    "grey": "#777F87",
    "grid": "#E6E9EC",
    "blue": "#4C78A8",
    "blue_light": "#9DB9D4",
    "teal": "#4C9A91",
    "green": "#6D9E78",
    "orange": "#D7904E",
    "gold": "#C9A23B",
    "red": "#C45D62",
    "violet": "#8064A2",
}

STRATEGY_COLORS = {
    "Anti-41BB": "#D7A261",
    "Anti-PD1": "#5D7FAE",
    "Combine": "#6F9D7C",
    "Optimal": "#C95E66",
}

STRATEGY_LABELS = {
    "Anti-41BB": "anti-4-1BB",
    "Anti-PD1": "anti-PD-1",
    "Combine": "Combination",
    "Optimal": "DiFCoN",
}

@contextmanager
def figure_style(font_size: float = 7.0):
    """Temporary journal-oriented styling with editable SVG/PDF text."""
    with mpl.rc_context(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": font_size,
            "axes.labelsize": font_size,
            "axes.titlesize": font_size + 0.8,
            "xtick.labelsize": max(5.0, font_size - 0.5),
            "ytick.labelsize": max(5.0, font_size - 0.5),
            "legend.fontsize": max(5.0, font_size - 0.6),
            "axes.linewidth": 0.75,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    ):
        yield


def load_baseline_table(stem: str) -> pd.DataFrame:
    """Read the one baseline result table needed by a panel."""
    return pd.read_csv(BASELINE_DIR / "tables" / f"{stem}.csv")


def cohort_metric_panel(
    summary: pd.DataFrame,
    pairwise: pd.DataFrame,
    metric: str,
    *,
    ax: plt.Axes,
    ylabel: str,
    font_size: float,
) -> None:
    """Draw the four fixed strategies and their comparisons with DiFCoN."""
    strategies = tuple(STRATEGY_LABELS)
    positions = np.arange(len(strategies))
    pivot = summary.pivot(index="patient_id", columns="strategy", values=metric).reindex(columns=strategies)
    paired = pivot.dropna().to_numpy(dtype=float)
    values = [pivot[strategy].dropna().to_numpy() for strategy in strategies]
    metric_stats = pairwise.loc[pairwise["metric"].eq(metric)].set_index(["strategy_a", "strategy_b"])
    rng = np.random.default_rng(2031)

    with figure_style(font_size):
        for row in paired:
            ax.plot(positions, row, color="#C9CED3", alpha=0.32, lw=0.45, zorder=1)

        boxes = ax.boxplot(
            values, positions=positions, widths=0.55, patch_artist=True, showfliers=False,
            medianprops={"color": "white", "lw": 1.0}, whiskerprops={"lw": 0.75}, capprops={"lw": 0.75},
        )
        for x, (patch, strategy, vals) in enumerate(zip(boxes["boxes"], strategies, values)):
            color = STRATEGY_COLORS[strategy]
            patch.set(facecolor=color, edgecolor=color, alpha=0.78)
            ax.scatter(
                x + rng.uniform(-0.09, 0.09, len(vals)), vals, s=8,
                color=color, alpha=0.58, lw=0, zorder=3,
            )

        local_min, local_max = float(np.nanmin(paired)), float(np.nanmax(paired))
        span = max(local_max - local_min, 0.08)
        bracket_base = local_max + 0.025 * span
        bracket_step = 0.1 * span
        bracket_height = 0.025 * span
        for level, strategy in enumerate(strategies[:-1]):
            pvalue = float(metric_stats.loc[(strategy, "Optimal"), "p_value"])
            symbol = baseline_visualization._significance_symbol(pvalue)
            baseline_visualization._draw_significance_bracket(
                ax, level, len(strategies) - 1,
                bracket_base + level * bracket_step, bracket_height, symbol,
                text_offset=1.0 if symbol == "ns" else 0.01,
            )

        bottom = local_min - 0.08 * span
        if local_min >= 0:
            bottom = max(0.0, bottom)
        ax.set_ylim(bottom, bracket_base + (len(strategies) - 2) * bracket_step + 0.05 * span)
        ax.set_xticks(positions, ["anti-\n4-1BB", "anti-\nPD-1", "Combo", "DiFCoN"], rotation=0, ha="center")
        ax.set_ylabel(ylabel, fontsize=font_size)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="x", labelsize=6.5, pad=2)
        ax.tick_params(axis="y", labelsize=font_size)
        ax.grid(False)


def baseline_patient_action_allocation_panel(
    *,
    figsize: tuple[float, float] = (3.2, 2.6),
    font_size: float = 7.0,
) -> plt.Figure:
    """Plot each test patient by total dose of each drug and color by final tumor burden."""

    # Sum each drug's doses from the test actions and read final tumor burden from the summary.
    summary = load_baseline_table("test_strategy_summary")
    actions = load_baseline_table("test_actions")
    dose = actions.loc[actions["strategy"].eq("Optimal"), ["patient_id", "action_PD1", "action_B41BB"]].groupby("patient_id", as_index=False).sum()
    dose = dose.rename(columns={"action_PD1": "total_PD1", "action_B41BB": "total_B41BB"})
    outcome = summary.loc[summary["strategy"].eq("Optimal"), ["patient_id", "final_C_rel"]].dropna().drop_duplicates("patient_id")
    patient = dose.merge(outcome, on="patient_id", how="inner").sort_values("patient_id")

    # Keep finite values to avoid Matplotlib warnings.
    cols = ["total_PD1", "total_B41BB", "final_C_rel"]
    patient = patient[np.isfinite(patient[cols]).all(axis=1)].copy()
    if patient.empty:
        raise ValueError("No finite test data for strategy 'Optimal'")

    # Use blue for low burden, gray near C(T)/C0 = 1, and red for high burden.
    vmin, vmax_data = 0.0, float(patient["final_C_rel"].max())
    vmax = np.ceil(vmax_data * 5) / 5
    center = 1.0
    center_pos = np.clip((center - vmin) / (vmax - vmin), 0.001, 0.999)
    tumor_cmap = LinearSegmentedColormap.from_list(
        "tumor_outcome",
        [(0.0, "#4C78A8"), (center_pos, "#9A9AA1"), (1.0, "#C65F5F")]
    )
    norm = Normalize(vmin=vmin, vmax=vmax)

    with figure_style(font_size):
        fig, ax = plt.subplots(figsize=figsize)

        sc = ax.scatter(
            patient["total_PD1"], patient["total_B41BB"], c=patient["final_C_rel"],
            cmap=tumor_cmap, norm=norm, s=26, alpha=0.92,
            edgecolor="white", linewidth=0.45, zorder=3,
        )

        ax.set(
            xlabel=r"Total anti-PD-1 dose, $\sum u_P(t)$",
            ylabel=r"Total anti-4-1BB dose, $\sum u_B(t)$",
        )
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.grid(False)

        # Use only 0, 1 and 2 as colorbar ticks.
        cb = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.05)
        cb.set_label(r"Final tumor, $C(T)/C_0$", fontsize=font_size)
        ticks = [t for t in (0, 1, 2) if t <= vmax + 1e-9]
        if not ticks: ticks = [0]
        cb.set_ticks(ticks)
        cb.set_ticklabels([str(int(t)) if float(t).is_integer() else f"{t:g}" for t in ticks])
        cb.ax.tick_params(labelsize=max(5.0, font_size - 0.5), length=2, pad=1)
        cb.outline.set_linewidth(0.5)

        fig.tight_layout()
    return fig


def representative_timeseries_panel(
    panel: str, trajectory: pd.DataFrame, *, ax: plt.Axes, font_size: float,
) -> None:
    """Draw the selected patient's tumor, activity, or abundance trajectory."""
    column, ylabel = {
        "tumor": ("C_rel", r"$C(t)/C(0)$"),
        "activity": ("activity", r"$a(t)=\frac{1}{N(t)}\int \rho(z,t)a(z)\,\mathrm{d}z$"),
        "N": ("N", r"$N(t)=\int \rho(z,t)\,\mathrm{d}z$"),
    }[panel]
    day_min, day_max = float(trajectory["day"].min()), float(trajectory["day"].max())
    day_ticks = np.arange(int(np.ceil(day_min)), int(np.floor(day_max)) + 1, 2, dtype=int)

    with figure_style(font_size):
        for strategy, label in STRATEGY_LABELS.items():
            part = trajectory.loc[trajectory["strategy"].eq(strategy)].sort_values("day")
            if part.empty:
                continue
            odd_idx = np.flatnonzero(np.rint(part["day"].to_numpy(float)).astype(int) % 2 == 1)
            color = baseline_visualization.PALETTE[strategy]
            ax.plot(
                part["day"], part[column], color=color, lw=1.5,
                marker=baseline_visualization.STRATEGY_MARKERS[strategy],
                markevery=odd_idx.tolist(), markersize=4.2,
                markerfacecolor=color, markeredgecolor="white", markeredgewidth=0.35,
                label=label,
            )

        ax.set(xlabel="Day", ylabel=ylabel, xlim=(day_min - 0.5, day_max + 0.5))
        ax.set_xticks(day_ticks)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(False)


PROGRAM_COLORS = {
    "precursor": "#7E6AA2",    # muted purple
    "activation": "#C97FB2",   # soft magenta
    "eff_early": "#5FA8B8",    # muted cyan-teal
    "eff_cytox": "#9A7B6F",    # muted brown
    "dysfunction": "#A6A34A",  # muted olive
}


def patient_scores_panel(trajectory: pd.DataFrame, *, ax: plt.Axes, font_size: float) -> None:
    """Draw the selected patient's five program scores under Combine and DiFCoN."""
    styles = {"Combine": ("--", "o", 1.0, 0.72), "Optimal": ("-", "s", 1.55, 1.0)}
    series = []
    for strategy, style in styles.items():
        part = trajectory.loc[trajectory["strategy"].eq(strategy)].sort_values("day")
        if part.empty:
            continue
        odd_idx = np.flatnonzero(np.rint(part["day"].to_numpy()).astype(int) % 2 == 1)
        series.append((part, odd_idx.tolist(), style))

    day_min, day_max = float(trajectory["day"].min()), float(trajectory["day"].max())
    day_ticks = np.arange(int(np.ceil(day_min)), int(np.floor(day_max)) + 1, 2, dtype=int)
    with figure_style(font_size):
        for score, color in PROGRAM_COLORS.items():
            for part, markers, (linestyle, marker, width, alpha) in series:
                ax.plot(
                    part["day"], part[f"score_{score}"], color=color, lw=width, alpha=alpha,
                    ls=linestyle, marker=marker, markevery=markers, markersize=3.5,
                    markerfacecolor=color, markeredgecolor="white", markeredgewidth=0.3,
                )

        ax.set(xlabel="Day", ylabel="Program score", xlim=(day_min - 0.5, day_max + 0.5))
        ax.set_xticks(day_ticks)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(False)


def openloop_feedback_tradeoff_panel(*, figsize=(90 / 25.4, 60 / 25.4), font_size=8.0) -> plt.Figure:
    """Plot patient-level closed-loop dose-efficacy trade-off."""
    differences = read_source_table('openloop','feedback_patient_differences').set_index('patient_id')
    dx = differences['dose_open_minus_closed']
    dy = differences['tumour_open_minus_closed']

    with figure_style(font_size):
        fig, ax = plt.subplots(figsize=figsize)
        ax.axhline(0, color=COLORS["grey"], lw=0.7, ls="--")
        ax.axvline(0, color=COLORS["grey"], lw=0.7, ls="--")
        ax.scatter(dx, dy, s=22, color="#C95E66", alpha=0.75, linewidth=0)

        xlim = np.nanmax(np.abs(dx)) * 1.08
        ylim = np.nanmax(np.abs(dy)) * 1.08
        ax.set_xlim(-xlim, xlim)
        ax.set_ylim(-ylim, ylim)

        kw = dict(fontsize=font_size - 0.7, color="black")

        ax.annotate("", xy=(0.90, 0.07), xytext=(0.10, 0.07), xycoords="axes fraction",
                    arrowprops=dict(arrowstyle="<->", color=COLORS["grey"], lw=0.7))
        ax.text(0.10, 0.11, "More dose", transform=ax.transAxes, ha="left", va="bottom", **kw)
        ax.text(0.90, 0.11, "Dose saving", transform=ax.transAxes, ha="right", va="bottom", **kw)

        ax.annotate("", xy=(1.01, 0.95), xytext=(1.01, 0.05), xycoords="axes fraction",
                    arrowprops=dict(arrowstyle="<->", color=COLORS["grey"], lw=0.7))
        ax.text(1.05, 0.80, "Better control", transform=ax.transAxes, ha="left", va="center", rotation=90, **kw)
        ax.text(1.05, 0.20, "Worse control", transform=ax.transAxes, ha="left", va="center", rotation=90, **kw)
        ax.set(xlabel="Dose difference", ylabel="Tumor-control difference")
        ax.set_title("Closed-loop vs open-loop", pad=2, fontsize=font_size - 0.2)
        fig.tight_layout()

    return fig


LOSS_ABLATION_LABELS = {"wC0": r"$w_T=0$", "wCTB0": r"$w_{A}=0$", "wF0": r"$w_F=0$", "wE0": r"$w_E=0$"}
LOSS_ABLATION_COLORS = {"wC0": "#C96A6A", "wCTB0": "#D39A5B", "wF0": "#7FA37A", "wE0": "#8B77A8"}

LOSS_METRIC_ORDER = ("C_relative", "CTB", "L_F", "D_total")
LOSS_METRIC_LABELS = {
    "C_relative": r"$C_{\mathrm{relative}}$",
    "CTB": r"$\mathrm{CTB}$",
    "L_F": r"$\mathcal{L}_F$",
    "D_total": r"$D_{\mathrm{total}}$",
}


def seed2031_loss_ablation_bar_panel(
    *,
    figsize: tuple[float, float] = (4.7, 2.7),
    font_size: float = 7.0,
) -> plt.Figure:
    """Plot four loss ablations across four metrics with a reference line at x = 1 for Full."""

    table = read_source_table("objective", "seed2031_means_ratios").set_index("variant")
    ratio_columns = [f"{metric}_relative" for metric in LOSS_METRIC_ORDER]

    # Draw bars for the ablations and a reference line for Full.
    ablation_order = ("wC0", "wCTB0", "wF0", "wE0")
    y = np.arange(len(LOSS_METRIC_ORDER))*1.1

    bar_h = 0.2
    offsets = np.linspace(-1.5, 1.5, len(ablation_order)) * bar_h *1.1

    with figure_style(font_size):
        fig, ax = plt.subplots(figsize=figsize)

        for off, variant in zip(offsets, ablation_order):
            vals = table.loc[variant, ratio_columns].to_numpy(float)

            ax.barh(
                y + off, vals,
                height=bar_h * 0.8,
                color=LOSS_ABLATION_COLORS[variant],
                alpha=0.90,
                edgecolor="none",
                label=LOSS_ABLATION_LABELS[variant],
                zorder=3,
            )

        # Full objective = 1
        ax.axvline(
            1.0,
            color="#777F87",
            lw=1,
            ls="--",
            zorder=10,
        )

        ax.set_yticks(
            y,
            [LOSS_METRIC_LABELS[m] for m in LOSS_METRIC_ORDER]
        )
        ax.invert_yaxis()

        ax.set_xlabel("Relative value (Full = 1)")
        ax.grid(False)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.legend(
            loc="lower center",
            bbox_to_anchor=(0.8, 0.6),
            ncol=2,
            frameon=False,
            fontsize=font_size - 0.4,
            columnspacing=0.8,
            handletextpad=0.35,
        )

        fig.tight_layout()

    return fig


def plot_observation_full_vs_practical(
    *, figsize: tuple[float, float] = (6.2, 2.35), font_size: float = 7.0,
) -> plt.Figure:
    """Compare full and reduced observations using saved Holm-adjusted tests."""
    patient_avg = read_source_table("observation", "patient_seed_means")
    tests = read_source_table("observation", "paired_tests")
    settings = ("Full state", "Practical state")
    tests = tests.loc[
        tests["setting_a"].eq(settings[0]) & tests["setting_b"].eq(settings[1])
    ].set_index("metric")
    metric_specs = (
        ("final_C_rel", r"Final tumor, $C(T)/C_0$"),
        ("total_dose", r"$D_{\mathrm{total}}=\sum [u_P(t)+u_B(t)]$"),
    )
    colors = {"Full state": "#6F9D7C", "Practical state": "#5D7FAE"}
    rng = np.random.default_rng(2031)
    x = np.arange(2)

    with figure_style(font_size):
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        for ax, (metric, ylabel) in zip(axes, metric_specs):
            pivot = patient_avg.pivot(index="patient_id", columns="setting", values=metric).reindex(columns=settings).dropna()
            values = [pivot[setting].to_numpy(float) for setting in settings]
            paired = pivot.to_numpy(float)
            for row in paired:
                ax.plot(x, row, color="#C9CED3", lw=0.45, alpha=0.32, zorder=1)

            boxes = ax.boxplot(
                values, positions=x, widths=0.52, patch_artist=True, showfliers=False,
                medianprops={"color": "white", "lw": 1.0},
                whiskerprops={"lw": 0.75}, capprops={"lw": 0.75},
            )
            for xpos, (patch, setting, vals) in enumerate(zip(boxes["boxes"], settings, values)):
                color = colors[setting]
                patch.set(facecolor=color, edgecolor=color, alpha=0.78)
                jitter = rng.uniform(-0.08, 0.08, len(vals))
                ax.scatter(xpos + jitter, vals, s=8, color=color, alpha=0.58, edgecolor="none", zorder=3)

            ax.set_xticks(x, ["Full\ninput", "Reduced\ninput"])
            ax.set_ylabel(ylabel)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(axis="x", labelsize=font_size, pad=2)
            ax.tick_params(axis="y", labelsize=font_size - 0.3)
            ax.grid(False)

            result = tests.loc[metric]
            pvalue = float(result["p_holm"])
            symbol = baseline_visualization._significance_symbol(pvalue) if np.isfinite(pvalue) else "ns"
            local_min, local_max = float(np.nanmin(paired)), float(np.nanmax(paired))
            span = max(local_max - local_min, 1e-3)
            y = local_max + 0.035 * span
            height = 0.025 * span
            ax.plot([0, 0, 1, 1], [y, y + height, y + height, y], color="#444444", lw=0.65, clip_on=False, zorder=5)
            ax.text(
                0.15, y + height, symbol, ha="center", va="bottom",
                fontsize=font_size - 0.4, fontweight="bold", color="#444444",
            )
            bottom = local_min - 0.06 * span
            if local_min >= 0:
                bottom = max(0.0, bottom)
            ax.set_ylim(bottom, y + height + 0.10 * span)
            mean_diff = float(result["mean_difference"])
            ax.text(
                1.04, 1.0, rf"$\Delta_{{R-F}}={mean_diff:.3f}$",
                transform=ax.transAxes, ha="right", va="top", fontsize=font_size - 0.4,
            )

        fig.subplots_adjust(left=0.14, right=0.965, top=0.97, bottom=0.20, wspace=0.45)
    return fig


def plot_openloop_control_comparison(
    *, figsize: tuple[float, float] = (110 / 25.4, 55 / 25.4), font_size: float = 7.0,
) -> plt.Figure:
    """Compare open-loop and closed-loop outcomes paired by virtual patient."""
    order = ("Validation-derived open-loop", "Closed-loop")
    tests = read_source_table("openloop", "paired_tests")
    tests = tests.loc[tests["strategy_a"].eq(order[0]) & tests["strategy_b"].eq(order[1])].set_index("metric")
    patient_df = read_source_table("openloop", "patient_seed_means")
    colors = {"Validation-derived open-loop": "#B07D3C", "Closed-loop": "#7A5AA6"}
    specs = (
        ("final_C_rel", r"Final tumor, $C(T)/C_0$"),
        ("total_dose", r"$D_{\mathrm{total}}=\sum [u_P(t)+u_B(t)]$"),
    )
    rng = np.random.default_rng(2031)
    x = np.arange(2)

    with figure_style(font_size):
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        for ax, (metric, ylabel) in zip(axes, specs):
            pivot = patient_df.pivot(index="patient_id", columns="strategy", values=metric).reindex(columns=order).dropna()
            values = [pivot[strategy].to_numpy(float) for strategy in order]
            paired = pivot.to_numpy(float)
            for row in paired:
                ax.plot(x, row, color="#D0D4D8", lw=0.45, alpha=0.50, zorder=1)

            boxes = ax.boxplot(
                values, positions=x, widths=0.50, patch_artist=True, showfliers=False,
                medianprops={"color": "white", "lw": 1.0},
                whiskerprops={"lw": 0.75}, capprops={"lw": 0.75},
            )
            for xpos, (patch, strategy, vals) in enumerate(zip(boxes["boxes"], order, values)):
                color = colors[strategy]
                patch.set(facecolor=color, edgecolor=color, alpha=0.78)
                jitter = rng.uniform(-0.07, 0.07, len(vals))
                ax.scatter(xpos + jitter, vals, s=8, color=color, alpha=0.60, edgecolor="none", zorder=3)

            ymin, ymax = float(np.nanmin(paired)), float(np.nanmax(paired))
            span = max(ymax - ymin, 0.08)
            base = ymax + 0.035 * span
            height = 0.018 * span
            result = tests.loc[metric]
            ax.plot([0, 0, 1, 1], [base, base + height, base + height, base],
                    color="#333333", lw=0.7, clip_on=False, zorder=5)
            ax.text(
                0.5 - 0.32, base + height, str(result["significance"]), ha="center", va="bottom",
                fontsize=font_size - 0.4, fontweight="bold", color="#333333", clip_on=False, zorder=6,
            )

            if metric == "final_C_rel":
                delta = float(result["mean_difference_a_minus_b"])
                ax.text(
                    1.08, 1.02, rf"$\Delta_{{\mathrm{{OL-CL}}}}={delta:.3f}$",
                    transform=ax.transAxes, ha="right", va="top", fontsize=font_size - 0.2,
                )
            bottom = max(0.0, ymin - 0.08 * span) if ymin >= 0 else ymin - 0.08 * span
            ax.set_ylim(bottom, base + height + 0.06 * span)
            ax.set_xticks(x, ["Open-loop", "Closed-loop"])
            ax.set_ylabel(ylabel)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(axis="x", labelsize=font_size, pad=2)
            ax.tick_params(axis="y", labelsize=font_size - 0.2)
            ax.grid(False)

        fig.subplots_adjust(left=0.14, right=0.965, top=0.97, bottom=0.20, wspace=0.45)
    return fig
