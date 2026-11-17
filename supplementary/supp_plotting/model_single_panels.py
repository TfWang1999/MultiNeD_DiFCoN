"""Program sensitivity weights and their dependence on perturbation size."""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from .editing import ARIAL_MATH, PROGRAM_LABELS
from .model_dynamics import FIELD_TREATMENTS, PROGRAM_COLUMNS
from .model_style import figure_style

DISPLAY_TREATMENTS = ("Control", "anti-PD-1", "anti-4-1BB", "Combination")


def plot_grouped_treatment_weights(importance, *, figsize, font_size=10,
                                   colors=("#8B929E", "#6796B5", "#D4A063", "#B77D99")):
    """Show the mean and SD at the reference perturbation of 0.5."""
    half = importance.loc[importance["delta_sd"].eq(0.5)].set_index(["treatment", "program"])
    with figure_style(font_size), mpl.rc_context(ARIAL_MATH):
        fig, ax = plt.subplots(figsize=figsize)
        x = np.arange(len(PROGRAM_COLUMNS))
        bar_width = 0.18
        for j, (treatment, label, color) in enumerate(zip(FIELD_TREATMENTS, DISPLAY_TREATMENTS, colors)):
            frame = half.loc[treatment].reindex(list(PROGRAM_COLUMNS))
            ax.bar(x + (j - 1.5) * bar_width, frame["alpha_mean"],
                   yerr=frame["alpha_sd"], width=bar_width, color=color, label=label,
                   capsize=1.8, linewidth=0,
                   error_kw={"ecolor": "#444444", "elinewidth": 0.65, "capthick": 0.65},
                   zorder=3)
        ax.axhline(0, color="#777777", lw=0.7, zorder=2)
        ax.set_xticks(x, PROGRAM_LABELS, fontsize=font_size)
        ax.set(xlabel="Immune program", ylabel=r"Program sensitivity weight, $\alpha$", xlim=(-0.6, 4.6))
        ax.tick_params(axis="y", labelsize=9)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.grid(False)
        ax.legend(loc="lower left", bbox_to_anchor=(0.01, 0.025), ncol=2, fontsize=9,
                  frameon=False, handlelength=1.4, columnspacing=1.0, labelspacing=0.35, borderaxespad=0)
        fig.subplots_adjust(left=0.19, right=0.985, bottom=0.22, top=0.965)
    return fig


def plot_grouped_delta_sensitivity(importance, *, figsize, font_size=10,
                                   colors=("#687798", "#43978F", "#C79248", "#BD6D72", "#9674AA")):
    """Use bars for the reference weights and markers for all three perturbations."""
    means = (importance.pivot(index=["program", "delta_sd"], columns="treatment", values="alpha_mean")
             .reindex(columns=FIELD_TREATMENTS))
    deltas = (0.25, 0.5, 1.0)
    markers = ("o", "s", "^")
    with figure_style(font_size), mpl.rc_context(ARIAL_MATH):
        fig, ax = plt.subplots(figsize=figsize)
        x = np.arange(len(FIELD_TREATMENTS))
        for j, (program, color) in enumerate(zip(PROGRAM_COLUMNS, colors)):
            values = means.loc[program]
            positions = x + (j - 2) * 0.16
            ax.bar(positions, values.loc[0.5], width=0.13, color=color,
                   alpha=0.65, linewidth=0, zorder=2)
            for k, (delta, marker) in enumerate(zip(deltas, markers)):
                ax.scatter(positions + (k - 1) * 0.045, values.loc[delta],
                           s=2.8**2, marker=marker, color=color,
                           edgecolors="#333333", linewidths=0.35, zorder=4)
        ax.axhline(0, color="#777777", lw=0.7, zorder=1)
        ax.set(xlabel="Treatment", ylabel=r"Program sensitivity weight, $\alpha$",
               xlim=(-0.55, 3.55), ylim=(-0.23, 0.21))
        ax.set_xticks(x, DISPLAY_TREATMENTS, rotation=20, ha="right", fontsize=9)
        ax.tick_params(axis="y", labelsize=9)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.grid(False)
        handles = [Line2D([], [], color="#555555", marker=marker, linestyle="none",
                          markersize=2.8, label=rf"$\delta={delta:g}$")
                   for delta, marker in zip(deltas, markers)]
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.48, 0.93),
                   ncol=3, fontsize=9, frameon=False, handlelength=1.0, columnspacing=1.0,
                   handletextpad=0.4, borderaxespad=0)
        fig.subplots_adjust(left=0.19, right=0.79, bottom=0.27, top=0.84)
        cmap = ListedColormap(colors)
        boundaries = np.arange(len(PROGRAM_COLUMNS) + 1)
        norm = BoundaryNorm(boundaries, cmap.N)
        cax = fig.add_axes([0.84, 0.32, 0.035, 0.46])
        colorbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                               boundaries=boundaries, ticks=np.arange(len(PROGRAM_COLUMNS)) + 0.5)
        colorbar.set_ticklabels(PROGRAM_LABELS)
        colorbar.ax.tick_params(labelsize=9, length=0, pad=3)
        colorbar.outline.set_linewidth(0.5)
    return fig
