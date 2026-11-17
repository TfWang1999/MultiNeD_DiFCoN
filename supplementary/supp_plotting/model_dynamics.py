"""Cellular LOTO, projected drift and state density panels."""

import json

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import MaxNLocator
from scipy.ndimage import convolve1d

from MultiNeD.figure_work.scripts.constants import METHOD_LABELS, TREATMENT_LABELS, TREATMENT_ORDER
from MultiNeD.figure_work.scripts.figure_data import load_table
from .editing import style_axis_text
from .model_style import figure_style

DYNAMICS_METHODS = (
    "persistence", "treatment_linear_shift", "entropic_ot_barycentric",
    "official_ot_cfm", "mioflow", "tigon", "project_balanced_sinkhorn_drift",
)
FIELD_TREATMENTS = ("Control", "Anti-PD1", "Anti-4-1BB", "Combination")
PROGRAM_COLUMNS = {
    "Precursor": "P_precursor_zstd",
    "Activation": "Activation_zstd",
    "Early effector": "Eff_Early_zstd",
    "Cytotoxic": "Eff_Cytox_zstd",
    "Dysfunction": "Dysf_zstd",
}


def plot_loto_w1_heatmap(paired, *, figsize, cmap, font_size=10):
    """Show the mean W1 across the input seeds for each held-out fold."""
    value_columns = [f"wasserstein_1__{method}" for method in DYNAMICS_METHODS]
    fold_order = [
        "day4__Control", "day4__AntiPD1", "day6__Control", "day6__AntiPD1",
        "day6__Anti41BB", "day6__AntiPD1plusAnti41BB",
    ]
    labels = ["D4 Ctrl", "D4 PD-1", "D6 Ctrl", "D6 PD-1", "D6 4-1BB", "D6 Combo"]
    matrix = paired.groupby("target_group")[value_columns].mean().reindex(fold_order).to_numpy(float).T
    with figure_style(font_size):
        fig, axis = plt.subplots(figsize=figsize)
        image = axis.imshow(matrix, cmap=cmap, aspect="auto")
        axis.set_xticks(np.arange(len(labels)), labels, rotation=35, ha="right")
        axis.set_yticks(np.arange(len(DYNAMICS_METHODS)), [METHOD_LABELS[method] for method in DYNAMICS_METHODS])
        axis.set(xlabel="Held-out target", ylabel="Dynamics method")
        axis.set_title("Held-out W1 across\nfolds and methods", pad=6)
        colorbar = fig.colorbar(image, ax=axis, fraction=0.035, pad=0.025)
        # Keep the paper layout's first pass before adding the colorbar line break.
        colorbar.set_label("Wasserstein-1 (lower is better)")
        fig.tight_layout()
    axis.title.set_fontsize(font_size)
    colorbar.set_label("Wasserstein-1\n(lower is better)")
    style_axis_text(colorbar.ax, label_size=9, tick_size=9)
    fig.tight_layout(pad=0.45)
    return fig


def load_dense_field(directory):
    background = load_table("drift_background.csv.gz", directory)
    background["treatment"] = background["treatment"].replace({
        "Anti-41BB": "Anti-4-1BB", "Anti-PD1+Anti-41BB": "Combination",
    })
    dense = load_table("drift_dense_field.csv.gz", directory)
    parameters = json.loads((directory / "source_data/drift_display_parameters.json").read_text(encoding="utf-8"))
    fields = {}
    for treatment in FIELD_TREATMENTS:
        frame = dense.loc[dense["treatment"].eq(treatment)].sort_values(["y", "x"])
        x, y = np.sort(frame.x.unique()), np.sort(frame.y.unique())
        if len(x) != len(y) or len(frame) != len(x) * len(y):
            raise ValueError(f"Expected a square saved drift grid for {treatment}")
        shape = (len(y), len(x))
        fields[treatment] = (x, y, frame.display_U.to_numpy().reshape(shape),
                             frame.display_V.to_numpy().reshape(shape))
    return background, fields, parameters["main_limits"]


def draw_dense_field(bundle, program, *, figsize, font_size=10, tick_size=9, cmap="RdBu_r"):
    background, fields, limits = bundle
    column = PROGRAM_COLUMNS[program]
    values = background[column].to_numpy()
    low, high = np.quantile(values[np.isfinite(values)], (0.02, 0.98))
    norm = TwoSlopeNorm(vmin=min(float(low), -1e-12), vcenter=0, vmax=max(float(high), 1e-12))
    with mpl.rc_context({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": font_size, "axes.titlesize": font_size, "axes.labelsize": font_size,
        "legend.fontsize": font_size, "xtick.labelsize": tick_size, "ytick.labelsize": tick_size,
        "svg.fonttype": "path", "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.7, "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white",
    }):
        fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=True, sharey=True, layout="constrained")
        for ax, treatment in zip(axes.flat, FIELD_TREATMENTS):
            points = background.loc[background["treatment"].eq(treatment)]
            image = ax.scatter(points.DC1, points.DC2, c=points[column], cmap=cmap, norm=norm,
                               s=3.0, alpha=0.52, marker="o", edgecolors="none", rasterized=True, zorder=1)
            x, y, u, v = fields[treatment]
            ax.streamplot(x, y, np.ma.masked_invalid(u), np.ma.masked_invalid(v),
                          color="#35434F", density=1.05, linewidth=0.65, arrowsize=0.65,
                          minlength=0.025, maxlength=2.6, integration_direction="both",
                          broken_streamlines=True, zorder=3)
            ax.set_title(treatment.replace("Anti-PD1", "anti-PD-1"))
            ax.set(xlim=limits[0], ylim=limits[1], xlabel="DC1", ylabel="DC2")
            ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
        lower, upper = background[column].min() < norm.vmin, background[column].max() > norm.vmax
        extend = "both" if lower and upper else ("min" if lower else ("max" if upper else "neither"))
        colorbar = fig.colorbar(image, ax=list(axes.flat), location="right", shrink=0.8, pad=0.02, extend=extend)
        colorbar.set_label(f"{program} score\n(z-standardized)")
        colorbar.ax.tick_params(labelsize=tick_size)
    return fig


def prepare_density_arrays(particles):
    groups = {
        treatment: list(part.groupby("time", sort=True))
        for treatment, part in particles.groupby("treatment_clean", sort=False)
    }
    arrays = {}
    for program in PROGRAM_COLUMNS.values():
        pooled = particles[program].to_numpy()
        padding = max(np.ptp(pooled) * 0.02, 1e-6)
        edges = np.linspace(pooled.min() - padding, pooled.max() + padding, 41)
        for treatment in TREATMENT_ORDER:
            times, frames = zip(*groups[treatment])
            masses = np.column_stack([
                np.histogram(frame[program], bins=edges, weights=frame.get("particle_weight"))[0]
                for frame in frames
            ])
            probabilities = masses / masses.sum(axis=0)
            sd = probabilities.std(axis=0, ddof=0)
            heat = (probabilities - probabilities.mean(axis=0)) / np.where(sd > 0, sd, 1)
            for axis in (0, 1):
                heat = convolve1d(heat, [0.25, 0.5, 0.25], axis=axis, mode="nearest")
            arrays[treatment, program] = (np.asarray(times), edges, heat)
    return arrays


def plot_state_density_overview(particles, *, figsize, cmap="viridis", font_size=10):
    arrays = prepare_density_arrays(particles)
    fig, axes = plt.subplots(4, 5, figsize=figsize, sharex=True, sharey="col", constrained_layout=True)
    for row, treatment in enumerate(TREATMENT_ORDER):
        for col, (title, program) in enumerate(PROGRAM_COLUMNS.items()):
            axis = axes[row, col]
            times, edges, heat = arrays[treatment, program]
            image = axis.imshow(heat, origin="lower", aspect="auto", interpolation="nearest",
                                extent=(times.min(), times.max(), edges.min(), edges.max()),
                                cmap=cmap, vmin=-2.0, vmax=2.0)
            axis.set_xticks([1, 4, 7, 10, 13, 17])
            axis.set_xlim(1, 17)
            axis.tick_params(labelsize=8)
            if row == 0:
                axis.set_title(title, fontsize=font_size)
            if col == 0:
                axis.set_ylabel(TREATMENT_LABELS[treatment] + "\nState score (z)", fontsize=font_size)
            if row == 3:
                axis.set_xlabel("Time (days)", fontsize=font_size)
    colorbar = fig.colorbar(image, ax=axes, shrink=0.82, fraction=0.018, pad=0.015)
    colorbar.set_label("Within-time standardized\nbin probability", fontsize=font_size)
    colorbar.ax.tick_params(labelsize=8)
    return fig
