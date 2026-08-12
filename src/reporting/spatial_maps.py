from pathlib import Path
import re

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1 import make_axes_locatable


ROOT = Path(__file__).resolve().parents[2]
MASTER_GPKG = ROOT / "data" / "processed" / "master" / "master_analysis_dataset_final_v1_0.gpkg"
EMPLOYMENT_LISA = ROOT / "outputs" / "esda" / "tables" / "lisa_employment_results.csv"
MOBILITY_LISA = ROOT / "outputs" / "esda" / "tables" / "lisa_mobility_results.csv"
OUT_DIR = ROOT / "writing" / "figures" / "main"


def borough_name(lsoa_name: str) -> str:
    return re.sub(r"\s+\d{3}[A-Z]$", "", lsoa_name)


def add_panel_label(ax, text: str) -> None:
    ax.set_title(text, fontsize=13.6, fontweight="semibold", pad=9)


def save_figure(fig, stem: str) -> None:
    fig.savefig(OUT_DIR / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.05, facecolor="white")
    fig.savefig(OUT_DIR / f"{stem}.png", dpi=450, bbox_inches="tight", pad_inches=0.05, facecolor="white")
    plt.close(fig)


def draw_base_boundaries(ax, plot_gdf, boroughs, outline) -> None:
    boroughs.boundary.plot(ax=ax, color="#7f7f7f", linewidth=0.24, zorder=3)
    outline.plot(ax=ax, color="#404040", linewidth=0.70, zorder=4)
    ax.set_axis_off()
    ax.set_aspect("equal")


def figure_4_2(gdf: gpd.GeoDataFrame) -> None:
    plot_gdf = gdf.copy()
    plot_gdf["display_employment"] = np.log1p(plot_gdf["employees_count"])
    plot_gdf["display_inflow"] = np.log1p(plot_gdf["mobility_inflow_total"])
    plot_gdf["display_entropy"] = plot_gdf["flow_entropy"]
    plot_gdf["borough"] = plot_gdf["LSOA21NM"].map(borough_name)
    boroughs = plot_gdf.dissolve(by="borough", as_index=False)
    outline = plot_gdf.dissolve().boundary
    bounds = plot_gdf.total_bounds
    xpad = (bounds[2] - bounds[0]) * 0.018
    ypad = (bounds[3] - bounds[1]) * 0.018

    panels = [
        ("display_employment", "Workplace Employment", "log1p employees", "magma", "Highest employment\nconcentration", 0.50, 0.55),
        ("display_inflow", "Mobility Inflow", "log1p inflow", "viridis", "Major mobility\nhub", 0.45, 0.48),
        ("display_entropy", "Flow Entropy", "entropy", "cividis", "Highest flow\ndiversity", 0.60, 0.42),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.42), facecolor="white")
    for idx, (ax, (col, title, cbar_title, cmap, callout, tx, ty)) in enumerate(zip(axes, panels)):
        norm = Normalize(vmin=float(plot_gdf[col].min()), vmax=float(plot_gdf[col].max()))
        plot_gdf.plot(
            column=col,
            ax=ax,
            cmap=cmap,
            norm=norm,
            linewidth=0.018,
            edgecolor="#f1f1f1",
            zorder=1,
        )
        draw_base_boundaries(ax, plot_gdf, boroughs, outline)
        ax.set_xlim(bounds[0] - xpad, bounds[2] + xpad)
        ax.set_ylim(bounds[1] - ypad, bounds[3] + ypad)
        add_panel_label(ax, f"({chr(97 + idx)}) {title}")

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="4.2%", pad=0.035)
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        cb = fig.colorbar(sm, cax=cax)
        cb.ax.tick_params(labelsize=9.4, length=2.5, width=0.5)
        cb.set_label(cbar_title, fontsize=10.4, labelpad=5)

        ax.text(
            tx,
            ty,
            callout,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=9.8,
            fontweight="semibold",
            color="#1f2b33",
            bbox={
                "boxstyle": "round,pad=0.28,rounding_size=0.10",
                "facecolor": "white",
                "edgecolor": "#8b8b8b",
                "linewidth": 0.55,
                "alpha": 0.90,
            },
            zorder=10,
        )
    fig.subplots_adjust(left=0.01, right=0.985, top=0.90, bottom=0.02, wspace=0.22)
    save_figure(fig, "figure_4_2_core_spatial_distribution")


CLUSTER_ORDER = ["High-High", "Low-Low", "High-Low", "Low-High", "Non-significant"]
CLUSTER_COLORS = {
    "High-High": "#c9443f",
    "Low-Low": "#2c7bb6",
    "High-Low": "#fdae61",
    "Low-High": "#abd9e9",
    "Non-significant": "#f5f5f5",
}


def lisa_panel(ax, base_gdf, table, title, summary_lines, boroughs, outline, bounds) -> None:
    merged = base_gdf[["LSOA21CD", "geometry"]].merge(
        table[["LSOA21CD", "lisa_cluster_category"]], on="LSOA21CD", how="left"
    )
    merged["lisa_cluster_category"] = pd.Categorical(
        merged["lisa_cluster_category"].fillna("Non-significant"),
        categories=CLUSTER_ORDER,
        ordered=True,
    )
    for category in CLUSTER_ORDER[::-1]:
        subset = merged[merged["lisa_cluster_category"] == category]
        if not subset.empty:
            subset.plot(
                ax=ax,
                color=CLUSTER_COLORS[category],
                edgecolor="#eeeeee",
                linewidth=0.018,
                antialiased=True,
                zorder=1,
            )
    draw_base_boundaries(ax, merged, boroughs, outline)
    xpad = (bounds[2] - bounds[0]) * 0.02
    ypad = (bounds[3] - bounds[1]) * 0.02
    ax.set_xlim(bounds[0] - xpad, bounds[2] + xpad)
    ax.set_ylim(bounds[1] - ypad, bounds[3] + ypad)
    ax.set_title(title, fontsize=14.0, fontweight="semibold", pad=9)
    ax.text(
        0.055,
        0.075,
        "\n".join(summary_lines),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10.2,
        linespacing=1.16,
        bbox={
            "boxstyle": "round,pad=0.34,rounding_size=0.10",
            "facecolor": "white",
            "edgecolor": "#8e8e8e",
            "linewidth": 0.60,
            "alpha": 0.94,
        },
        zorder=10,
    )


def figure_4_4(gdf: gpd.GeoDataFrame) -> None:
    base_gdf = gdf.copy()
    base_gdf["borough"] = base_gdf["LSOA21NM"].map(borough_name)
    boroughs = base_gdf.dissolve(by="borough", as_index=False)
    outline = base_gdf.dissolve().boundary
    bounds = base_gdf.total_bounds
    employment = pd.read_csv(EMPLOYMENT_LISA)
    mobility = pd.read_csv(MOBILITY_LISA)
    emp_counts = employment["lisa_cluster_category"].value_counts()
    mob_counts = mobility["lisa_cluster_category"].value_counts()
    emp_sig = int(employment["significant_p_0_05"].sum())
    mob_sig = int(mobility["significant_p_0_05"].sum())

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.05), facecolor="white")
    lisa_panel(
        axes[0],
        base_gdf,
        employment,
        "(a) Workplace Employment",
        [
            "Panel summary",
            f"High-High: {int(emp_counts.get('High-High', 0))}",
            f"Low-Low: {int(emp_counts.get('Low-Low', 0))}",
            f"Significant: {emp_sig}",
        ],
        boroughs,
        outline,
        bounds,
    )
    lisa_panel(
        axes[1],
        base_gdf,
        mobility,
        "(b) Mobility Inflow",
        [
            "Panel summary",
            f"High-High: {int(mob_counts.get('High-High', 0))}",
            f"Low-Low: {int(mob_counts.get('Low-Low', 0))}",
            f"Significant: {mob_sig}",
        ],
        boroughs,
        outline,
        bounds,
    )
    handles = [Patch(facecolor=CLUSTER_COLORS[c], edgecolor="#6e6e6e", label=c) for c in CLUSTER_ORDER]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=5,
        frameon=True,
        title="LISA category",
        fontsize=10.3,
        title_fontsize=11.0,
        borderpad=0.45,
        handlelength=1.25,
        columnspacing=1.1,
    )
    fig.subplots_adjust(left=0.015, right=0.985, top=0.92, bottom=0.14, wspace=0.08)
    save_figure(fig, "figure_4_4_lisa_cluster_maps")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gdf = gpd.read_file(MASTER_GPKG)
    figure_4_2(gdf)
    figure_4_4(gdf)


if __name__ == "__main__":
    main()
