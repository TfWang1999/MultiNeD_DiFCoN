"""Human validation panels and their shared display tables."""

import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.lines import Line2D

from .editing import style_axis_text
from .model_style import figure_style

COLORS = {
    "ink": "#30343B", "grid": "#D9DDE1", "blue": "#4C78A8",
    "red": "#C45D62", "orange": "#D7904E", "teal": "#4C9A91", "violet": "#8064A2",
}

FUNCTION_TERMS = [
    ("GO Biological Process", "T Cell Activation (GO:0042110)", "T-cell activation"),
    ("GO Biological Process", "Regulation of Natural Killer Cell Mediated Cytotoxicity (GO:0042269)", "NK-cell cytotoxicity regulation"),
    ("GO Biological Process", "Lymphocyte Differentiation (GO:0030098)", "Lymphocyte differentiation"),
    ("Reactome", "Cytokine Signaling in Immune System", "Cytokine signaling"),
    ("Reactome", "Signaling by Interleukins", "Interleukin signaling"),
    ("Reactome", "RUNX3 Regulates Immune Response and Cell Migration", "RUNX3: immune response and migration"),
    ("MSigDB Hallmark", "IL-2/STAT5 Signaling", "IL-2/STAT5 signaling"),
    ("MSigDB Hallmark", "Interferon Gamma Response", "Interferon-gamma response"),
    ("MSigDB Hallmark", "TNF-alpha Signaling via NF-kB", "TNF-NF-kB signaling"),
    ("MSigDB Hallmark", "IL-6/JAK/STAT3 Signaling", "IL-6/JAK/STAT3 signaling"),
    ("MSigDB Hallmark", "Inflammatory Response", "Inflammatory response"),
    ("MSigDB Hallmark", "Apoptosis", "Apoptosis"),
    ("MSigDB Hallmark", "Hypoxia", "Hypoxia"),
]
# The eight functional themes in Fig. 3e.
MAIN_THEMES = [
    (FUNCTION_TERMS[i][0], FUNCTION_TERMS[i][1], label)
    for i, label in [(6, "IL-2/STAT5"), (7, "IFN-gamma"), (8, "TNF-NF-kB"),
                     (9, "IL-6/JAK/STAT3"), (0, "T-cell activation"),
                     (1, "NK-cell regulation"), (5, "RUNX3 immune"), (12, "Hypoxia")]
]


def _label_axis(axis, xlabel, title, xlim):
    axis.set(xlabel=xlabel, xlim=xlim)
    if title:
        axis.set_title(title, pad=9)
    axis.grid(axis="x", color=COLORS["grid"], lw=0.45, zorder=0)


def _metastatic_score_rows(frame):
    return frame.loc[frame["cohort"].eq("Metastatic only") & frame["score"].eq("CD8score_Control")]


def metastatic_risk_table(patients, statistics):
    """Count patients at risk using the saved median cutpoint and inclusion criteria."""
    statistics = _metastatic_score_rows(statistics)
    data = patients.loc[patients["sample_type"].eq("Metastatic")]
    data = data.dropna(subset=["os_time_days", "os_event", "CD8score_Control"])
    data = data.loc[data["os_time_days"].gt(0)]
    threshold = float(statistics["median_cutpoint"].iloc[0])
    groups = np.where(data["CD8score_Control"].ge(threshold), "High", "Low")
    rows = []
    for group in ("High", "Low"):
        time = data.loc[groups == group, "os_time_days"]
        for year in (0, 5, 10, 15, 20, 25):
            rows.append(dict(group=group, time_years=float(year), at_risk=int(time.ge(year * 365.25).sum())))
    return pd.DataFrame(rows)


def plot_metastatic_km(curves, statistics, risk_counts, *, figsize, font_size=10):
    curves = _metastatic_score_rows(curves)
    statistics = _metastatic_score_rows(statistics)
    with figure_style(font_size):
        fig, (axis, risk_axis) = plt.subplots(
            2, 1, figsize=figsize, sharex=True, gridspec_kw={"height_ratios": [4.2, 1.25], "hspace": 0.9},
        )
        for y, (group, color) in enumerate(zip(("High", "Low"), (COLORS["blue"], COLORS["orange"]))):
            part = curves.loc[curves["group"].eq(group)].sort_values("time_days")
            n = int(statistics.loc[statistics["group"].eq(group), "n"].iloc[0])
            axis.step(part["time_days"] / 365.25, part["survival"], where="post", color=color,
                      lw=1.45, label=f"{group} SW-CD8 (n={n})")
            for row in risk_counts.loc[risk_counts["group"].eq(group)].itertuples():
                risk_axis.text(row.time_years, 1 - y, str(int(row.at_risk)), ha="center", va="center", color=color)
        axis.text(0.04, 0.09, f"Log-rank P = {statistics['logrank_p_value'].iloc[0]:.2g}", transform=axis.transAxes)
        axis.set(ylabel="Overall survival", ylim=(0, 1.03))
        _label_axis(axis, "Time (years)", "TCGA-SKCM metastatic subset", (0, 26))
        axis.tick_params(labelbottom=True)
        axis.xaxis.set_label_position("bottom")
        axis.set_xticks(sorted(risk_counts["time_years"].unique()))
        axis.legend(labels=["High SW-CD8", "Low SW-CD8"], loc="upper right", fontsize=9)
        risk_axis.set_yticks([1, 0], ["High", "Low"])
        risk_axis.set_ylim(-0.45, 1.65)
        risk_axis.set_title("Number at risk", loc="left", fontsize=font_size, pad=5)
        risk_axis.tick_params(axis="both", length=0, labelbottom=False)
        risk_axis.tick_params(axis="y", pad=18)
        for spine in risk_axis.spines.values():
            spine.set_visible(False)
        fig.subplots_adjust(left=0.17, right=0.96, bottom=0.08, top=0.88)
    style_axis_text(axis, label_size=font_size, tick_size=9, title_size=font_size)
    style_axis_text(risk_axis, label_size=font_size, tick_size=9)
    return fig


def longitudinal_display_table(trajectories, summary):
    return trajectories.merge(
        summary.rename(columns={"mean": "cohort_mean", "sd": "cohort_sd", "n": "cohort_n", "sem": "cohort_sem"}),
        on="timepoint", how="left", validate="many_to_one",
    )


def plot_longitudinal_patient_trajectories(trajectories, summary, *, figsize, font_size=10):
    visits = ["Baseline", "Follow Up 1", "Follow Up 2", "Follow Up 3", "Follow Up 4"]
    patients = sorted(trajectories["patient_alias"].unique())
    summary = summary.set_index("timepoint").loc[visits]
    colors = (COLORS["blue"], COLORS["orange"], COLORS["violet"], COLORS["teal"])
    with figure_style(font_size):
        fig, axis = plt.subplots(figsize=figsize)
        for patient, color in zip(patients, colors):
            part = trajectories.loc[trajectories["patient_alias"].eq(patient)].set_index("timepoint").loc[visits]
            axis.plot(range(5), part["delta_CD8score"], "o-", color=color, lw=0.9, ms=3, alpha=0.60, label=patient)
        axis.errorbar(range(5), summary["mean"], yerr=summary["sem"], color=COLORS["ink"], marker="o",
                      ms=4, lw=1.7, capsize=3, elinewidth=1.1, label="Mean ± SEM (n=4)", zorder=5)
        axis.axhline(0, color="#777777", lw=0.8, ls="--")
        axis.set_xticks(range(5), ["Baseline", "FU1", "FU2", "FU3", "FU4"])
        axis.set_ylabel(r"$\Delta$SW-CD8 from baseline")
        _label_axis(axis, "Human follow-up visit", "Longitudinal SW-CD8:\npatients and mean", (-0.15, 4.15))
        axis.legend(loc="upper left", ncol=2, fontsize=9)
        fig.subplots_adjust(left=0.19, right=0.96, bottom=0.20, top=0.81)
    style_axis_text(axis, label_size=font_size, tick_size=9, title_size=font_size)
    return fig


def select_enrichment_terms(tables, terms):
    """Select displayed terms while retaining the full-library BH-FDR."""
    rows = []
    for group, frame in tables.items():
        indexed = frame.set_index(["library", "term"])
        for order, (library, term, label) in enumerate(terms):
            row = indexed.loc[library, term].to_dict()
            rows.append({**row, "library": library, "term": term, "display_label": label,
                         "display_group": group, "display_order": order})
    return pd.DataFrame(rows)


def draw_enrichment_bubbles(axis, displayed, *, bubble_scale, term_wrap, color_max=32):
    groups = ("All CD8score", "Positive weights", "Negative weights")
    norm = Normalize(0, color_max)
    cmap = plt.get_cmap("viridis")
    short = {"GO Biological Process": "GO", "Reactome": "Reactome", "MSigDB Hallmark": "Hallmark"}
    counts = []
    for x, group in enumerate(groups):
        part = displayed.loc[displayed["display_group"].eq(group)].sort_values("display_order")
        counts.append(int(part["query_gene_count"].iloc[0]))
        for row in part.itertuples():
            if row.overlap_count == 0:
                axis.text(x, row.display_order, "-", ha="center", va="center", color="#888888")
            else:
                color = cmap(norm(-np.log10(max(float(row.bh_fdr), np.finfo(float).tiny))))
                axis.scatter(x, row.display_order, s=bubble_scale * row.gene_ratio,
                             facecolors=color if row.bh_fdr < 0.05 else "white", edgecolors=color, linewidths=0.7)
    labels = displayed.loc[displayed["display_group"].eq(groups[0])].sort_values("display_order")
    axis.set_yticks(labels["display_order"],
                    [textwrap.fill(f"{row.display_label} [{short[row.library]}]", term_wrap) for row in labels.itertuples()])
    axis.set_xticks(range(3), [f"{name}\n(n={n})" for name, n in zip(("All", "Positive", "Negative"), counts)])
    axis.set(xlim=(-0.5, 2.5), ylim=(len(labels) - 0.5, -0.5))
    for i in range(1, len(labels)):
        if labels.iloc[i]["library"] != labels.iloc[i - 1]["library"]:
            axis.axhline(i - 0.5, color="#cccccc", lw=0.6)
    for spine in axis.spines.values():
        spine.set_visible(False)
    return cmap, norm


def plot_cd8score_functional_matrix(displayed, *, figsize, font_size=8):
    bubble_scale = 500
    with figure_style(font_size):
        fig, axis = plt.subplots(figsize=figsize)
        cmap, norm = draw_enrichment_bubbles(axis, displayed, bubble_scale=bubble_scale, term_wrap=33)
        axis.tick_params(axis="both", length=0)
        axis.tick_params(axis="x", labeltop=True, labelbottom=False, pad=7)
        fig.subplots_adjust(left=0.57, right=0.97, bottom=0.23, top=0.84)
        cax = fig.add_axes([0.66, 0.115, 0.28, 0.02])
        colorbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal")
        colorbar.set_label(r"$-\log_{10}$(BH-FDR)", labelpad=2)
        handles = [Line2D([], [], marker="o", ls="", color="#888888", markerfacecolor="#bbbbbb",
                          markersize=np.sqrt(bubble_scale * r), label=f"{r:.0%}") for r in (0.02, 0.08, 0.15)]
        fig.legend(handles=handles, title="Gene ratio", loc="lower left", bbox_to_anchor=(0.08, 0.055), ncol=3)
        fig.text(0.08, 0.026, "Open: q >= 0.05; -: no overlap. Background: 56,296 detectable TCGA symbols.",
                 fontsize=font_size - 1)
    return fig


def build_top50_annotations(genes, enrichment, terms):
    """Build the gene-by-term membership table used by S5b and S3_05."""
    selected = select_enrichment_terms({"All CD8score": enrichment}, terms)
    memberships = [(term, set(str(term.overlap_genes).split(";")) if pd.notna(term.overlap_genes) else set())
                   for term in selected.itertuples()]
    rows = []
    for gene in genes.sort_values("abs_model_weight_rank").itertuples():
        for term, members in memberships:
            rows.append(dict(abs_model_weight_rank=gene.abs_model_weight_rank,
                             human_ensembl_gene=gene.human_ensembl_gene, human_symbol=gene.human_symbol,
                             cd8score_gene_weight=gene.cd8score_gene_weight, library=term.library, term=term.term,
                             display_label=term.display_label, display_order=term.display_order,
                             annotated=gene.human_symbol in members))
    return pd.DataFrame(rows)


def top50_matrix(annotations):
    genes = annotations.drop_duplicates("human_ensembl_gene").sort_values("abs_model_weight_rank")
    terms = annotations[["display_order", "display_label"]].drop_duplicates().sort_values("display_order")
    matrix = annotations.pivot(index="human_ensembl_gene", columns="display_order", values="annotated")
    return genes, terms, matrix.loc[genes["human_ensembl_gene"], terms["display_order"]]


def plot_top50_function_annotation(annotations, *, figsize, font_size=7.5):
    genes, terms, matrix = top50_matrix(annotations)
    with figure_style(font_size):
        fig, (weight_axis, annotation_axis) = plt.subplots(
            1, 2, figsize=figsize, gridspec_kw={"width_ratios": [1.3, 3.3]}, sharey=True,
        )
        y = np.arange(len(genes))
        colors = np.where(genes["cd8score_gene_weight"].ge(0), COLORS["red"], COLORS["violet"])
        weight_axis.barh(y, genes["cd8score_gene_weight"], color=colors, height=0.68)
        weight_axis.axvline(0, color="#555555", lw=0.6)
        weight_axis.set_yticks(y, genes["human_symbol"])
        weight_axis.tick_params(axis="y", length=0, pad=3)
        _label_axis(weight_axis, r"Gene coefficient $\beta_g$", None, (-0.025, 0.04))
        weight_axis.set_title("Gene coefficients", pad=7, fontsize=font_size)
        annotation_axis.imshow(matrix.to_numpy(dtype=int), aspect="auto", interpolation="none",
                               cmap=ListedColormap(["#f1f1f1", COLORS["teal"]]), vmin=0, vmax=1)
        annotation_axis.set_xticks(range(len(terms)), terms["display_label"], rotation=45, ha="right")
        annotation_axis.tick_params(axis="y", left=False, labelleft=False)
        annotation_axis.tick_params(axis="x", length=0)
        annotation_axis.set_xticks(np.arange(-0.5, len(terms)), minor=True)
        annotation_axis.set_yticks(np.arange(-0.5, len(genes)), minor=True)
        annotation_axis.grid(which="minor", color="white", lw=0.45)
        annotation_axis.tick_params(which="minor", left=False, bottom=False)
        weight_axis.set_ylim(len(genes) - 0.5, -0.5)
        fig.subplots_adjust(left=0.16, right=0.98, bottom=0.17, top=0.92, wspace=0.08)
        handles = [Line2D([], [], color=COLORS[color], lw=4, label=label)
                   for color, label in (("red", "Positive beta"), ("violet", "Negative beta"), ("teal", "Annotated"))]
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.57, 0.055), ncol=3)
    return fig
