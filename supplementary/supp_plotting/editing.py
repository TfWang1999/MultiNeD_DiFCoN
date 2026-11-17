"""Shared supplementary panel dimensions, text sizing and exports."""

from pathlib import Path
from tempfile import NamedTemporaryFile

from IPython.display import display
import matplotlib as mpl
import matplotlib.pyplot as plt

PROGRAM_LABELS = (
    r"$P_{\mathrm{pre}}$", r"$A_{\mathrm{act}}$", r"$E_{\mathrm{early}}$",
    r"$Y_{\mathrm{cyto}}$", r"$D_{\mathrm{dysf}}$",
)
ARIAL_MATH = {
    "mathtext.fontset": "custom", "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold",
}
EXPORT_FONT = {"font.family": "Arial", "font.sans-serif": ["Arial"], **ARIAL_MATH}


def figure_size(width_mm, height_mm=None, aspect_ratio=1.5):
    """Convert millimetres to inches, deriving the height when needed."""
    height_mm = width_mm / aspect_ratio if height_mm is None else height_mm
    return width_mm / 25.4, height_mm / 25.4


def style_axis_text(axis, *, label_size, tick_size, title_size=None):
    axis.xaxis.label.set_fontsize(label_size)
    axis.yaxis.label.set_fontsize(label_size)
    axis.tick_params(labelsize=tick_size)
    if title_size is not None:
        axis.title.set_fontsize(title_size)


def save_figure_file(fig, path, **kwargs):
    """Save a complete file before replacing the previous export."""
    with NamedTemporaryFile(dir=path.parent, prefix=".supp_", suffix=path.suffix, delete=False) as file:
        pending = Path(file.name)
    try:
        fig.savefig(pending, **kwargs)
        pending.replace(path)
    finally:
        pending.unlink(missing_ok=True)


def _save_panel_files(fig, stem, output_dir, *, formats=("svg", "pdf", "png"), pdf_fonttype=42):
    output_dir.mkdir(parents=True, exist_ok=True)
    with mpl.rc_context({"savefig.bbox": None, "svg.fonttype": "path", "pdf.fonttype": pdf_fonttype}):
        for suffix in formats:
            save_figure_file(fig, output_dir / f"{stem}.{suffix}",
                             dpi=600, bbox_inches=None, facecolor="white")


def _show_panel(fig, stem):
    display(fig)
    plt.close(fig)
    print(stem)


def export_panel(fig, stem, *, output_dir):
    with mpl.rc_context({**EXPORT_FONT, "mathtext.sf": "Arial", "mathtext.cal": "STIXGeneral:italic"}):
        for text in fig.findobj(mpl.text.Text):
            text.set_fontfamily("Arial")
            text.set_math_fontfamily("custom")
        _save_panel_files(fig, stem, output_dir)
    _show_panel(fig, stem)


def export_human_panel(fig, stem, displayed_table, *, output_dir):
    _save_panel_files(fig, stem, output_dir)
    source_dir = output_dir / "source_data"
    source_dir.mkdir(parents=True, exist_ok=True)
    displayed_table.to_csv(source_dir / f"{stem}_displayed.csv", index=False)
    _show_panel(fig, stem)


def export_combined_function_panels(fig, output_dir):
    stem = "Human_Function_AB_A4_width"
    with mpl.rc_context(EXPORT_FONT):
        _save_panel_files(fig, stem, output_dir, formats=("pdf", "svg"), pdf_fonttype=3)
    _show_panel(fig, stem)
