"""Shared colors and significance annotations for the fixed paper panels."""

from __future__ import annotations

import numpy as np
from matplotlib.colors import LinearSegmentedColormap


STRATEGY_MARKERS = {
    "Anti-41BB": "D",
    "Anti-PD1": "o",
    "Combine": "p",
    "Optimal": "s",
}


PALETTE = {
    "Anti-41BB": "#D6A36A",
    "Anti-PD1": "#5D7FB5",
    "Combine": "#6F9B7B",
    "Optimal": "#C95F66",
    "train": "#7B86B6",
    "test": "#C95F66",
    "pd1": "#4C78A8",
    "b41bb": "#E08C35",
    "neutral": "#6F7278",
    "grid": "#E5E7EB",
}


SCORE_LABELS = {
    "precursor": r"$P_{\mathrm{pre}}$",
    "activation": r"$A_{\mathrm{act}}$",
    "eff_early": r"$E_{\mathrm{early}}$",
    "eff_cytox": r"$Y_{\mathrm{cyto}}$",
    "dysfunction": r"$D_{\mathrm{dysf}}$",
}

PD1_ACTION_CMAP = LinearSegmentedColormap.from_list(
    "pd1_action",
    [(0, "#FFFFFF"), (0.05, "#E2EAF2"), (0.25, "#C1CFDD"), (0.60, "#8FA7BF"), (1, "#5D7FAE")],
)
B41BB_ACTION_CMAP = LinearSegmentedColormap.from_list(
    "b41bb_action",
    [(0, "#FFFFFF"), (0.05, "#F3E8D9"), (0.25, "#E1C7A5"), (0.60, "#C7A071"), (1, "#A97B4E")],
)


def _significance_symbol(pvalue: float) -> str:
    if not np.isfinite(pvalue):
        return "n/a"
    if pvalue < 1e-4:
        return "****"
    if pvalue < 1e-3:
        return "***"
    if pvalue < 1e-2:
        return "**"
    if pvalue < 5e-2:
        return "*"
    return "ns"


def _draw_significance_bracket(
    ax,
    x1: float,
    x2: float,
    y: float,
    height: float,
    symbol: str,
    text_offset: float = -0.05,
) -> None:
    """Draw a short significance bracket matching the reference figure."""

    top = y + height

    ax.plot(
        [x1, x1, x2, x2],
        [y, top, top, y],
        color="#222222",
        lw=0.8,
        solid_capstyle="round",
        clip_on=False,
        zorder=6,
    )

    ax.text(
        (x1 + x2) / 2.0,
        top + text_offset * height,
        symbol,
        ha="center",
        va="baseline",
        fontsize=6.7,
        fontweight="bold",
        color="#111111",
        clip_on=False,
        zorder=7,
    )
