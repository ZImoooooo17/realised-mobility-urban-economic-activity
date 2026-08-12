import os
from pathlib import Path
import textwrap

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-casa-dissertation")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = [
    ROOT / "writing" / "figures" / "main",
    ROOT / "dissertation" / "figures" / "chapter3",
]


COLORS = {
    "ink": "#24313a",
    "muted": "#5d6b75",
    "accent": "#587a96",
    "accent_dark": "#36566f",
    "line": "#9aabba",
    "fill": "#f7fafc",
    "fill_strong": "#eef4f8",
    "fill_emphasis": "#e8f2f8",
    "white": "#ffffff",
}

BOX_LW = 0.9
BOX_ROUND = 0.014


def wrapped(text, width):
    return "\n".join(textwrap.wrap(text, width=width, break_long_words=False))


def add_box(
    ax,
    x,
    y,
    w,
    h,
    title,
    subtitle=None,
    fc=None,
    ec=None,
    lw=BOX_LW,
    title_size=8.8,
    subtitle_size=7.3,
    title_weight="semibold",
    rounding=BOX_ROUND,
    zorder=2,
):
    fc = fc or COLORS["fill"]
    ec = ec or COLORS["line"]
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.008,rounding_size={rounding}",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
        zorder=zorder,
    )
    ax.add_patch(patch)

    if subtitle:
        ax.text(
            x + w / 2,
            y + h * 0.67,
            title,
            ha="center",
            va="center",
            fontsize=title_size,
            fontweight=title_weight,
            color=COLORS["ink"],
            zorder=zorder + 1,
        )
        ax.text(
            x + w / 2,
            y + h * 0.31,
            subtitle,
            ha="center",
            va="center",
            fontsize=subtitle_size,
            color=COLORS["muted"],
            linespacing=1.05,
            zorder=zorder + 1,
        )
    else:
        ax.text(
            x + w / 2,
            y + h / 2,
            title,
            ha="center",
            va="center",
            fontsize=title_size,
            fontweight=title_weight,
            color=COLORS["ink"],
            linespacing=1.15,
            zorder=zorder + 1,
        )
    return patch


def add_label(ax, x, y, text):
    ax.text(
        x,
        y,
        text.upper(),
        ha="left",
        va="center",
        fontsize=7.0,
        fontweight="bold",
        color=COLORS["accent_dark"],
        alpha=0.92,
    )


def arrow(ax, x1, y1, x2, y2, lw=0.85):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=8,
            linewidth=lw,
            color=COLORS["line"],
            shrinkA=2,
            shrinkB=2,
            zorder=1,
        )
    )


def branch_connector(ax, points, lw=0.62):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    ax.plot(xs, ys, color=COLORS["line"], linewidth=lw, zorder=1, solid_capstyle="round")


def main():
    for output_dir in OUTPUTS:
        output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12.8, 8.0))
    fig.patch.set_facecolor(COLORS["white"])
    ax.set_facecolor(COLORS["white"])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Stage 1: research questions
    add_label(ax, 0.06, 0.895, "Research Questions")
    rq_y, rq_w, rq_h = 0.812, 0.275, 0.058
    rq_xs = [0.06, 0.3625, 0.665]
    rq_titles = ["RQ1", "RQ2", "RQ3"]
    rq_subs = [
        "Mobility-Employment\nAssociation",
        "Added Value\nBeyond Accessibility",
        "Model\nComparison",
    ]
    for x, title, sub in zip(rq_xs, rq_titles, rq_subs):
        add_box(
            ax,
            x,
            rq_y,
            rq_w,
            rq_h,
            title,
            sub,
            fc=COLORS["fill_strong"],
            ec=COLORS["accent"],
            lw=BOX_LW,
            title_size=8.2,
            subtitle_size=6.8,
        )

    # Stage 2: multi-source data
    y_data, h_data = 0.646, 0.105
    add_label(ax, 0.06, 0.775, "Multi-source Urban Data")
    data_groups = [
        ("Behavioural\nMobility", "Locomizer\nMobility"),
        ("Economic\nOutcome", "BRES\nEmployment"),
        ("Accessibility", "PTAL\nAccessibility"),
        ("Socio-demographic\nContext", "Population"),
        ("Built\nEnvironment", "Amenity POIs\nRetail POIs"),
    ]
    gap = 0.014
    w = (0.88 - gap * 4) / 5
    for i, (group, source) in enumerate(data_groups):
        x = 0.06 + i * (w + gap)
        highlighted = i == 0
        add_box(
            ax,
            x,
            y_data,
            w,
            h_data,
            group,
            source,
            fc=COLORS["fill_emphasis"] if highlighted else COLORS["fill"],
            ec=COLORS["accent"] if highlighted else COLORS["line"],
            lw=BOX_LW,
            title_size=7.4,
            subtitle_size=6.6,
        )

    for x in rq_xs:
        arrow(ax, x + rq_w / 2, rq_y - 0.008, 0.5, y_data + h_data + 0.014)

    # Stage 3: spatial integration and variable construction
    add_label(ax, 0.06, 0.616, "Spatial Integration and Variable Construction")
    container = FancyBboxPatch(
        (0.06, 0.480),
        0.88,
        0.098,
        boxstyle="round,pad=0.010,rounding_size=0.018",
        linewidth=0.95,
        edgecolor=COLORS["line"],
        facecolor=COLORS["white"],
        zorder=0,
    )
    ax.add_patch(container)
    chain = [
        ("Greater London", ""),
        ("LSOA 2021", ""),
        ("Spatial\nHarmonisation", ""),
        ("Variable\nConstruction", ""),
        ("Master Analysis\nDataset", "n = 4,994 LSOAs"),
    ]
    chain_w, chain_h, chain_gap = 0.145, 0.062, 0.028
    chain_y = 0.498
    chain_x0 = 0.087
    centers = []
    for i, (title, sub) in enumerate(chain):
        x = chain_x0 + i * (chain_w + chain_gap)
        centers.append((x + chain_w / 2, chain_y + chain_h / 2))
        add_box(
            ax,
            x,
            chain_y,
            chain_w,
            chain_h,
            title,
            sub,
            fc=COLORS["fill_emphasis"] if i == 4 else COLORS["fill"],
            ec=COLORS["accent"] if i == 4 else COLORS["line"],
            title_size=7.0,
            subtitle_size=6.2,
            lw=BOX_LW,
        )
    for (x1, y1), (x2, y2) in zip(centers[:-1], centers[1:]):
        arrow(ax, x1 + chain_w / 2 - 0.008, y1, x2 - chain_w / 2 + 0.008, y2, lw=0.75)
    arrow(ax, 0.5, y_data - 0.008, 0.5, 0.578)

    # Stage 4: progressive analytical strategy
    add_label(ax, 0.06, 0.455, "Progressive Analytical Strategy")
    strategy_box = FancyBboxPatch(
        (0.06, 0.308),
        0.88,
        0.108,
        boxstyle=f"round,pad=0.010,rounding_size={BOX_ROUND}",
        linewidth=BOX_LW,
        edgecolor=COLORS["accent"],
        facecolor=COLORS["white"],
        zorder=0,
    )
    ax.add_patch(strategy_box)
    main_methods = [
        ("ESDA", "Spatial Structure"),
        ("OLS", "Global Benchmark"),
        ("SAR / SEM", "Spatial Dependence"),
        ("XGBoost", "Nonlinearity; Preferred Explainable Model"),
        ("SHAP", "Interpretability"),
    ]
    m_y, m_h = 0.361, 0.042
    method_layout = [
        (0.095, 0.145),
        (0.265, 0.145),
        (0.435, 0.145),
        (0.608, 0.172),
        (0.805, 0.135),
    ]
    method_centers = []
    for (x, m_w), (title, sub) in zip(method_layout, main_methods):
        method_centers.append((x + m_w / 2, m_y + m_h / 2))
        fc = COLORS["fill_strong"] if title in {"XGBoost", "SHAP"} else COLORS["fill"]
        ec = COLORS["accent"] if title in {"XGBoost", "SHAP"} else COLORS["line"]
        add_box(
            ax,
            x,
            m_y,
            m_w,
            m_h,
            title,
            sub,
            fc=fc,
            ec=ec,
            lw=BOX_LW,
            title_size=6.9,
            subtitle_size=5.1 if title == "XGBoost" else 5.4,
            rounding=BOX_ROUND,
        )
    for idx, ((x1, y1), (x2, y2)) in enumerate(zip(method_centers[:-1], method_centers[1:])):
        w1 = method_layout[idx][1]
        w2 = method_layout[idx + 1][1]
        arrow(ax, x1 + w1 / 2 - 0.006, y1, x2 - w2 / 2 + 0.006, y2, lw=0.72)
    # Random Forest remains part of the analytical strategy as a secondary comparison branch.
    rf_x, rf_y, rf_w, rf_h = 0.455, 0.315, 0.118, 0.027
    add_box(
        ax,
        rf_x,
        rf_y,
        rf_w,
        rf_h,
        "Random Forest",
        "Comparison Model",
        fc=COLORS["fill"],
        ec=COLORS["line"],
        lw=0.75,
        title_size=6.3,
        subtitle_size=5.0,
        rounding=BOX_ROUND,
    )
    sar_center = method_centers[2]
    rf_center_x = rf_x + rf_w / 2
    branch_connector(
        ax,
        [
            (sar_center[0], m_y + 0.002),
            (sar_center[0], rf_y + rf_h / 2),
            (rf_x - 0.006, rf_y + rf_h / 2),
        ],
        lw=0.55,
    )
    arrow(ax, 0.5, 0.480, 0.5, 0.416)

    # Stage 5: model evaluation and contribution
    add_label(ax, 0.06, 0.275, "Evaluation and Contribution")
    eval_y, eval_h = 0.210, 0.042
    metrics = [
        "Adjusted R2",
        "RMSE",
        "AIC",
        "Residual Moran's I",
        "Spatial Cross-validation",
    ]
    metric_gap = 0.012
    metric_w = (0.88 - metric_gap * 4) / 5
    for i, metric in enumerate(metrics):
        add_box(
            ax,
            0.06 + i * (metric_w + metric_gap),
            eval_y,
            metric_w,
            eval_h,
            wrapped(metric, 16),
            fc=COLORS["fill"],
            title_size=7.2,
            lw=BOX_LW,
        )
    arrow(ax, 0.5, 0.308, 0.5, eval_y + eval_h + 0.006)

    add_box(
        ax,
        0.06,
        0.112,
        0.25,
        0.050,
        "Evidence for RQ1-RQ3",
        fc=COLORS["fill_strong"],
        ec=COLORS["accent"],
        title_size=8.3,
        lw=BOX_LW,
    )
    contrib = FancyBboxPatch(
        (0.345, 0.111),
        0.595,
        0.064,
        boxstyle=f"round,pad=0.008,rounding_size={BOX_ROUND}",
        linewidth=BOX_LW,
        edgecolor=COLORS["accent"],
        facecolor=COLORS["white"],
        zorder=2,
    )
    ax.add_patch(contrib)
    ax.text(
        0.6425,
        0.154,
        "Research Contribution",
        ha="center",
        va="center",
        fontsize=8.0,
        fontweight="semibold",
        color=COLORS["ink"],
        zorder=3,
    )
    ax.text(
        0.6425,
        0.131,
        wrapped(
            "Behavioural mobility indicators complement conventional accessibility measures for explaining the spatial distribution of urban economic activity.",
            104,
        ),
        ha="center",
        va="center",
        fontsize=7.4,
        color=COLORS["muted"],
        linespacing=0.98,
        zorder=3,
    )
    arrow(ax, 0.31, 0.137, 0.345, 0.137)

    for output_dir in OUTPUTS:
        stem = output_dir / "figure_3_1_overall_research_framework_v3"
        fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.08)
        fig.savefig(stem.with_suffix(".png"), dpi=450, bbox_inches="tight", pad_inches=0.08)

    plt.close(fig)


if __name__ == "__main__":
    main()
