"""Central style and export settings for the reaction-model paper figures."""

from copy import deepcopy
from pathlib import Path

import matplotlib as mpl

from .constants import HELDOUT_DAY_ORDER, TREATMENT_COLORS

DEFAULT_STYLE = {
    "font_family": "Arial",
    "font_size": 7,
    "title_size": 8,
    "label_size": 7,
    "xlabel_size": None,
    "ylabel_size": None,
    "tick_size": 6,
    "xtick_size": None,
    "ytick_size": None,
    "legend_size": 6,
    "line_width": 0.8,
    "marker_size": 4,
    "figsize": (4.0, 3.0),
    "title": None,
    "xlabel": None,
    "ylabel": None,
    "xlim": None,
    "ylim": None,
    "legend_loc": "best",
}

METHOD_COLORS = {
    "persistence": "#4C78A8",
    "mioflow": "#4C9A91",
    "treatment_linear_shift": "#A9AEB3",
    "project_balanced_sinkhorn_drift": "#C45D62",
    "entropic_ot_barycentric": "#D7904E",
    "tigon": "#A9AEB3",
    "official_ot_cfm": "#7563A8",
}

FIGURES = {
    "tumour_reconstruction": {
        "figsize": (100 / 25.4, 55 / 25.4), "xlabel": "Time (days)", "ylabel": r"Tumor volume (mm$^3$)",
        "label_size": 7.5, "tick_size": 6.5, "legend_size": 5.8, "line_width": 1.0, "marker_size": 4,
        "legend_position": (0.32, 0.55, 0.40, 0.42), "treatment_colors": TREATMENT_COLORS,
    },
    "state_reconstruction": {
        "figsize": (95 / 25.4, 60 / 25.4), "xlabel": "Target mean", "ylabel": "Predicted mean",
        "label_size": 7, "tick_size": 6, "legend_size": 6, "marker_size": 5.3,
        "legend_loc": "lower right", "program_legend_loc": "upper left", "metric_text_xy": (0.04, 0.96),
        "treatment_colors": TREATMENT_COLORS, "identity_color": "#777777", "fit_color": "#B64342",
        "program_labels": {"P_precursor_zstd": r"$P_{\mathrm{pre}}$", "Activation_zstd": r"$A_{\mathrm{act}}$",
                           "Eff_Early_zstd": r"$E_{\mathrm{early}}$", "Eff_Cytox_zstd": r"$Y_{\mathrm{cyto}}$",
                           "Dysf_zstd": r"$D_{\mathrm{dysf}}$"},
        "program_markers": {"P_precursor_zstd": "o", "Activation_zstd": "^", "Eff_Early_zstd": "s",
                            "Eff_Cytox_zstd": "D", "Dysf_zstd": "P"},
    },
    "loto_prediction": {
        "figsize": (5.5, 3.8), "title": None, "xlabel": "Held-out day", "ylabel": r"Tumor volume (mm$^3$)",
        "label_size": 8, "tick_size": 7, "legend_size": 6, "line_width": 0.8, "marker_size": 4,
        "xlim": (-0.55, 3.55), "ylim": (0, None), "legend_loc": "upper left", "type_legend_loc": "upper center",
        "days": HELDOUT_DAY_ORDER, "treatment_colors": TREATMENT_COLORS,
    },
    "dynamics_benchmark_folds": {
        "figsize": (6.2, 2.8), "title": "Shared conditional LOTO benchmark", "xlabel": None, "ylabel": None,
        "label_size": 7, "tick_size": 6, "legend_size": 6, "line_width": 0.8, "marker_size": 4,
        "metric_specs": [
            ("sinkhorn_divergence", "Debiased transport cost"),
            ("energy_distance", "Energy distance"),
            ("score_mean_mae", "Program-mean MAE"),
            ("wasserstein_1", "Wasserstein-1 distance"),
        ],
        "method_colors": METHOD_COLORS,
    },
    "full_data_component_ablation": {
        "figsize": (7.2, 3.2), "title": "Full-data component ablation (in-sample)", "xlabel": None, "ylabel": None,
        "label_size": 6.5, "tick_size": 5.8, "legend_size": 5.5, "line_width": 0.8, "marker_size": 3.2,
        "metric_specs": [("transition_sinkhorn", "One-step prediction", "Debiased transport cost"),
                         ("rollout_sinkhorn", "Multi-step rollout", "Debiased transport cost"),
                         ("volume_mae", "Volume reconstruction", "MAE")],
        "variant_colors": {
            "balanced_ot_local_rollout_pretrained": "#204E6F",
            "balanced_ot_local_only_pretrained": "#D97732",
            "balanced_ot_rollout_only_pretrained": "#C45D62",
            "balanced_ot_local_rollout_no_pretrain": "#7563A8",
        },
        "treatment_colors": TREATMENT_COLORS,
        "r2_xlim": (0.74, 1.012), "r2_ticks": [0.75, 0.80, 0.85, 0.90, 0.95, 1.00],
    },
}


def get_figure_config(name):
    """Return an independent merged config so notebook edits never affect other figures."""
    if name not in FIGURES:
        raise KeyError(f"Unknown figure config: {name}")
    return deepcopy({**DEFAULT_STYLE, **FIGURES[name]})


def apply_global_style():
    """Apply the shared defaults for Fig. 2."""
    cfg = DEFAULT_STYLE
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": [cfg["font_family"], "DejaVu Sans", "Liberation Sans"],
        "font.size": cfg["font_size"], "axes.titlesize": cfg["title_size"], "axes.labelsize": cfg["label_size"],
        "xtick.labelsize": cfg["tick_size"], "ytick.labelsize": cfg["tick_size"], "legend.fontsize": cfg["legend_size"],
        "lines.linewidth": cfg["line_width"], "lines.markersize": cfg["marker_size"], "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
        "svg.fonttype": "path", "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.facecolor": "white",
    })


def save_figure(fig, output_dir, stem, bbox_inches="tight"):
    """Export Fig. 2 as SVG, PDF and PNG, with SVG text outlined."""
    output_dir = Path(output_dir)
    paths = [output_dir / f"{stem}.{suffix}" for suffix in ("svg", "pdf", "png")]
    output_dir.mkdir(parents=True, exist_ok=True)
    with mpl.rc_context({"svg.fonttype": "path"}):
        for path in paths:
            options = {"dpi": 300} if path.suffix == ".png" else {}
            fig.savefig(path, bbox_inches=bbox_inches, pad_inches=0.02, facecolor="white", **options)
    return paths


def apply_nature_style(font_size):
    """Apply the compact journal style used from Fig. 2b onward."""
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "svg.fonttype": "none",
            "font.size": font_size,
            "axes.titlesize": font_size + 1,
            "axes.labelsize": font_size,
            "xtick.labelsize": font_size,
            "ytick.labelsize": font_size,
            "legend.fontsize": font_size,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.8,
            "ytick.major.size": 2.8,
            "lines.linewidth": 1.4,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _set_limits(ax, xlim=None, ylim=None):
    if xlim is not None:
        ax.set_xlim(left=xlim[0], right=xlim[1])
    if ylim is not None:
        ax.set_ylim(bottom=ylim[0], top=ylim[1])


def _style_axis(ax, cfg, title=None, xlabel=None, ylabel=None):
    title = cfg.get("title") if title is None else title
    xlabel = cfg.get("xlabel") if xlabel is None else xlabel
    ylabel = cfg.get("ylabel") if ylabel is None else ylabel
    for title_loc in ("left", "center", "right"):
        ax.set_title("", loc=title_loc)
    if title:
        ax.set_title(title, fontsize=cfg["title_size"], loc=cfg.get("title_loc", "left"))
    ax.set_xlabel("" if not xlabel else xlabel, fontsize=_size(cfg, "xlabel_size", "label_size"))
    ax.set_ylabel("" if not ylabel else ylabel, fontsize=_size(cfg, "ylabel_size", "label_size"))
    ax.tick_params(axis="x", labelsize=_size(cfg, "xtick_size", "tick_size"))
    ax.tick_params(axis="y", labelsize=_size(cfg, "ytick_size", "tick_size"))
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    _set_limits(ax, cfg.get("xlim"), cfg.get("ylim"))



def _size(cfg, specific, inherited):
    return cfg.get(specific) if cfg.get(specific) is not None else cfg[inherited]
