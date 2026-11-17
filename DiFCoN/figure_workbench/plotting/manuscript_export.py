"""Typography and flat PNG/SVG/PDF export for Figs. 4–6."""

from pathlib import Path

import matplotlib as mpl
from matplotlib.legend import Legend
from matplotlib.text import Text

from ._paths import FIGURE_DIR


DEFAULT_FONT_SIZES = dict.fromkeys(
    ("base", "title", "axis_label", "xtick", "ytick", "legend", "legend_title",
     "annotation", "colorbar_label", "colorbar_tick", "offset"),
    8,
)


def apply_font_sizes(fig, sizes):
    """Apply the plotting cell's explicit sizes, including helper-created text."""
    # Materialize automatic ticks before sizing their labels and offset text.
    fig.canvas.draw()
    for text in fig.findobj(match=Text):
        text.set_fontsize(sizes["annotation"])
    if fig._suptitle is not None:
        fig._suptitle.set_fontsize(sizes["title"])
    for text in (fig._supxlabel, fig._supylabel):
        if text is not None:
            text.set_fontsize(sizes["axis_label"])
    for ax in fig.axes:
        is_colorbar = getattr(ax, "_colorbar", None) is not None
        label_size = sizes["colorbar_label" if is_colorbar else "axis_label"]
        ax.xaxis.label.set_fontsize(label_size)
        ax.yaxis.label.set_fontsize(label_size)
        for title in (ax.title, ax._left_title, ax._right_title):
            title.set_fontsize(sizes["title"])
        for name in ("x", "y"):
            tick_size = sizes["colorbar_tick" if is_colorbar else f"{name}tick"]
            ax.tick_params(axis=name, which="both", labelsize=tick_size)
            getattr(ax, f"{name}axis").get_offset_text().set_fontsize(sizes["offset"])
    # findobj also reaches figure legends and legends retained with add_artist.
    for legend in fig.findobj(match=Legend):
        for text in legend.get_texts():
            text.set_fontsize(sizes["legend"])
        legend.get_title().set_fontsize(sizes["legend_title"])


def save_main_panel(fig, stem, *, font_sizes=DEFAULT_FONT_SIZES, bbox_inches="tight"):
    """Write all formats together; SVG glyphs are paths, PNG is 600 dpi."""
    if Path(stem).name != stem or "/" in stem or "\\" in stem:
        raise ValueError("Panel stem must be a filename without a directory")
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    with mpl.rc_context({"svg.fonttype": "path", "pdf.fonttype": 42}):
        apply_font_sizes(fig, font_sizes)
        paths = []
        for suffix in ("png", "svg", "pdf"):
            path = FIGURE_DIR / f"{stem}.{suffix}"
            fig.savefig(path, dpi=600, bbox_inches=bbox_inches)
            paths.append(path)
    return paths
