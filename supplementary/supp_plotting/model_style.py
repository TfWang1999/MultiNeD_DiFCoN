"""Model notebook defaults and the shared per-panel style."""

from contextlib import contextmanager

import matplotlib as mpl


def apply_global_style():
    """Set the model notebook defaults used by the density overview."""
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "font.size": 7, "axes.titlesize": 8, "axes.labelsize": 7,
        "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6,
        "lines.linewidth": 0.8, "lines.markersize": 4, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
        "svg.fonttype": "path", "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.facecolor": "white",
    })


@contextmanager
def figure_style(font_size):
    with mpl.rc_context({
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
        "svg.fonttype": "path",
        "pdf.fonttype": 42,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }):
        yield
