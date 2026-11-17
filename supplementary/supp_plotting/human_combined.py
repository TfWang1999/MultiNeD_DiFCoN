"""The full-width S5 layout with shared enrichment and Top50 data."""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

from .editing import ARIAL_MATH, figure_size
from .human import COLORS, draw_enrichment_bubbles, top50_matrix
from .model_style import figure_style


def plot_combined_function_panels(
    enrichment, annotations, *, width_mm=210, height_mm=200,
    font_size=9, gene_font_size=8.5, panel_label_size=14,
    color_max=32, bubble_scale=800, term_wrap=44,
    beta_ylim=(-0.025, 0.040), beta_height_scale=0.60,
):
    """Draw enrichment above the aligned Top50 weights and memberships."""
    genes, themes, matrix = top50_matrix(annotations)
    assert len(enrichment) == 39 and matrix.shape == (50, 8)

    def axis_mm(fig, left, top, width, height):
        return fig.add_axes([left / width_mm, 1 - (top + height) / height_mm,
                             width / width_mm, height / height_mm])

    def text_mm(fig, left, top, text, **kwargs):
        return fig.text(left / width_mm, 1 - top / height_mm, text,
                        va="top", **kwargs)

    with figure_style(font_size), mpl.rc_context(ARIAL_MATH):
        fig = plt.figure(figsize=figure_size(width_mm, height_mm))

        # Panel a shows all original FDR values and gene ratios for 13 themes across three gene sets.
        text_mm(fig, 6, 7, "a", fontsize=panel_label_size, fontweight="bold")
        ax = axis_mm(fig, 100, 22, 67, 80)
        cmap, norm = draw_enrichment_bubbles(
            ax, enrichment, bubble_scale=bubble_scale, term_wrap=term_wrap, color_max=color_max,
        )
        ax.tick_params(axis="x", labeltop=True, labelbottom=False, length=0, pad=6, labelsize=font_size)
        ax.tick_params(axis="y", length=0, pad=5, labelsize=font_size)

        handles = [Line2D([], [], marker="o", linestyle="none", color="#888888",
                          markerfacecolor="#BBBBBB", markersize=np.sqrt(bubble_scale * ratio),
                          label=f"{ratio:.0%}") for ratio in (0.02, 0.08, 0.15)]
        fig.legend(handles=handles, title="Gene ratio", loc="upper left",
                   bbox_to_anchor=(178 / width_mm, 1 - 72 / height_mm),
                   ncol=1, frameon=False, fontsize=font_size, title_fontsize=font_size,
                   handlelength=1.3, labelspacing=0.8, borderaxespad=0)
        cax = axis_mm(fig, 179, 27, 3, 34)
        colorbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="vertical")
        colorbar.set_label(r"$-\log_{10}$(BH-FDR)", fontsize=font_size, labelpad=3)
        colorbar.ax.tick_params(labelsize=gene_font_size, length=2)
        colorbar.outline.set_linewidth(0.5)
        text_mm(fig, 178, 64, "Open: q >= 0.05\n-: no overlap.",
                fontsize=gene_font_size, linespacing=1.15)

        # Panel b aligns the coefficients of 50 genes above all 400 membership entries for eight themes.
        text_mm(fig, 6, 103, "b", fontsize=panel_label_size, fontweight="bold")
        beta_top = 111
        beta_height = 34 * beta_height_scale
        weight_ax = axis_mm(fig, 43, beta_top, 161, beta_height)
        annotation_ax = axis_mm(fig, 43, beta_top + beta_height + 7, 161, 31)
        x = np.arange(len(genes))
        colors = np.where(genes["cd8score_gene_weight"].ge(0), COLORS["red"], COLORS["violet"])
        weight_ax.bar(x, genes["cd8score_gene_weight"], color=colors, width=0.70)
        weight_ax.axhline(0, color="#666666", lw=0.6)
        weight_ax.set(xlim=(-0.5, len(genes) - 0.5), ylim=beta_ylim,
                      ylabel="Frozen gene\n" + r"weight $\beta$")
        weight_ax.set_xticks([])
        weight_ax.set_yticks([-0.02, 0, 0.02, 0.04])
        weight_ax.tick_params(axis="y", labelsize=gene_font_size)
        weight_ax.grid(axis="y", color="#E1E3E5", lw=0.4)
        annotation_ax.imshow(matrix.to_numpy(dtype=int).T, aspect="auto", interpolation="none",
                             cmap=ListedColormap(["#F1F1F1", COLORS["teal"]]), vmin=0, vmax=1)
        annotation_ax.set_yticks(range(len(themes)), themes["display_label"], fontsize=font_size)
        annotation_ax.set_xticks(x, genes["human_symbol"], rotation=90,
                                 ha="center", fontsize=gene_font_size)
        annotation_ax.tick_params(axis="y", length=0, pad=4)
        annotation_ax.tick_params(axis="x", length=0, pad=3)
        annotation_ax.set_xticks(np.arange(-0.5, len(genes)), minor=True)
        annotation_ax.set_yticks(np.arange(-0.5, len(themes)), minor=True)
        annotation_ax.grid(which="minor", color="white", lw=0.35)
        annotation_ax.tick_params(which="minor", left=False, bottom=False)
        for spine in annotation_ax.spines.values():
            spine.set_visible(False)
        handles = [Line2D([], [], color=color, lw=4, label=label)
                   for color, label in ((COLORS["red"], "Positive beta"),
                                        (COLORS["violet"], "Negative beta"),
                                        (COLORS["teal"], "Annotated"))]
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.59, 8 / height_mm),
                   ncol=3, fontsize=font_size, frameon=False, handlelength=1.6, columnspacing=1.2)
    return fig
