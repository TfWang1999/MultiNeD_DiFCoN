"""Held-out volume error and unsmoothed training curves."""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from MultiNeD.figure_work.scripts.constants import HELDOUT_DAY_ORDER, TREATMENT_LABELS, TREATMENT_ORDER
from MultiNeD.figure_work.scripts.figure_style import _style_axis
from .model_style import figure_style


def plot_loto_day_error(loto, *, figsize, cmap, font_size=10):
    errors = (loto["predicted_volume"] - loto["observed_mean_volume"]).abs()
    heat = (loto.assign(absolute_error=errors)
            .groupby(["treatment_clean", "day"])["absolute_error"].mean().unstack()
            .reindex(index=TREATMENT_ORDER, columns=HELDOUT_DAY_ORDER))
    with figure_style(font_size):
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(heat, ax=ax, cmap=cmap, annot=True, fmt=".0f",
                    annot_kws={"fontsize": 9}, linewidths=0.35, linecolor="white",
                    cbar_kws={"shrink": 0.75})
        ax.set_yticklabels([TREATMENT_LABELS[treatment] for treatment in TREATMENT_ORDER], rotation=0)
        _style_axis(ax, {"title_size": font_size, "label_size": font_size, "tick_size": 9},
                    title="Tumor-volume prediction error\nby held-out day", xlabel="Held-out day", ylabel=False)
        colorbar = fig.axes[-1]
        colorbar.set_ylabel(r"MAE (mm$^3$)", fontsize=9)
        colorbar.tick_params(labelsize=9)
        fig.tight_layout(pad=0.55)
    fig.tight_layout(pad=0.45)
    return fig


def plot_training_history(history, components, *, figsize, font_size=10):
    panels = (
        ([("loss", "Total", "#202020")], "Training objective", "Total loss"),
        (components, "Loss component", "Loss components"),
    )
    with figure_style(font_size):
        fig, axes = plt.subplots(1, 2, figsize=figsize, sharex=True)
        for ax, (curves, ylabel, title) in zip(axes, panels):
            for column, label, color in curves:
                values = pd.to_numeric(history[column], errors="raise")
                ax.plot(history["epoch"], values, label=label, color=color, linewidth=1.0)
            ax.set_xlabel("Epoch", fontsize=font_size)
            ax.set_ylabel(ylabel, fontsize=font_size)
            ax.set_title(title, fontsize=font_size)
            ax.tick_params(labelsize=9)
            ax.legend(frameon=False, fontsize=9, loc="upper right")
        fig.tight_layout()
    return fig
