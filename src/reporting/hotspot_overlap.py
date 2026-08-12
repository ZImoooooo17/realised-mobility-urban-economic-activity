from pathlib import Path
import re

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MASTER_GPKG = ROOT / "data" / "processed" / "master" / "master_analysis_dataset_final_v1_0.gpkg"
OVERLAP_CSV = ROOT / "outputs" / "esda" / "tables" / "lisa_employment_mobility_overlap.csv"
OUTPUT_STEM = ROOT / "writing" / "figures" / "main" / "figure_4_5_employment_mobility_hotspot_overlap"

WIDTH_PX = 2612
HEIGHT_PX = 2130
DPI = 450

COLORS = {
    "Both HH": "#6a00a8",
    "Employment HH only": "#d55e00",
    "Mobility HH only": "#0072b2",
    "Non-significant": "#f5f5f5",
}
ORDER = ["Both HH", "Employment HH only", "Mobility HH only", "Non-significant"]

OLD_SUMMARY_BOX_AXES = (0.055, 0.075)
SUMMARY_BOX_AXES = (0.105, 0.090)


def borough_name(lsoa_name: str) -> str:
    return re.sub(r"\s+\d{3}[A-Z]$", "", lsoa_name)


def classify(row: pd.Series) -> str:
    if row["both_high_high"]:
        return "Both HH"
    if row["employment_cluster_category"] == "High-High" and row["mobility_cluster_category"] != "High-High":
        return "Employment HH only"
    if row["mobility_cluster_category"] == "High-High" and row["employment_cluster_category"] != "High-High":
        return "Mobility HH only"
    return "Non-significant"


def main() -> None:
    gdf = gpd.read_file(MASTER_GPKG)
    overlap = pd.read_csv(OVERLAP_CSV)
    plot_gdf = gdf[["LSOA21CD", "LSOA21NM", "geometry"]].merge(overlap, on="LSOA21CD", how="inner", validate="one_to_one")
    plot_gdf["hotspot_class"] = plot_gdf.apply(classify, axis=1)
    plot_gdf["hotspot_class"] = pd.Categorical(plot_gdf["hotspot_class"], categories=ORDER, ordered=True)
    plot_gdf["borough"] = plot_gdf["LSOA21NM"].map(borough_name)

    counts = plot_gdf["hotspot_class"].value_counts().reindex(ORDER, fill_value=0)
    expected_counts = {"Both HH": 53, "Employment HH only": 88, "Mobility HH only": 383}
    for category, expected in expected_counts.items():
        actual = int(counts[category])
        if actual != expected:
            raise ValueError(f"{category} count changed: expected {expected}, got {actual}")

    boroughs = plot_gdf.dissolve(by="borough", as_index=False)
    london_outline = plot_gdf.dissolve().boundary

    bounds = plot_gdf.total_bounds
    xpad = (bounds[2] - bounds[0]) * 0.025
    ypad = (bounds[3] - bounds[1]) * 0.025

    fig, ax = plt.subplots(figsize=(WIDTH_PX / DPI, HEIGHT_PX / DPI), dpi=DPI, facecolor="white")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set_facecolor("white")

    for category in ORDER[::-1]:
        subset = plot_gdf[plot_gdf["hotspot_class"] == category]
        if not subset.empty:
            subset.plot(
                ax=ax,
                color=COLORS[category],
                edgecolor="#e6e6e6",
                linewidth=0.035,
                antialiased=True,
            )

    boroughs.boundary.plot(ax=ax, color="#7d7d7d", linewidth=0.35)
    london_outline.plot(ax=ax, color="#444444", linewidth=0.9)

    ax.set_xlim(bounds[0] - xpad, bounds[2] + xpad)
    ax.set_ylim(bounds[1] - ypad, bounds[3] + ypad)
    ax.set_aspect("equal")
    ax.set_axis_off()

    handles = [Patch(facecolor=COLORS[category], edgecolor="#777777", label=category) for category in ORDER]
    ax.legend(
        handles=handles,
        loc="upper right",
        bbox_to_anchor=(0.965, 0.925),
        frameon=True,
        facecolor="white",
        edgecolor="#9a9a9a",
        title="Hotspot class",
        fontsize=8.1,
        title_fontsize=9.2,
        borderpad=0.45,
        labelspacing=0.25,
        handlelength=1.35,
    )

    summary = "Hotspot Overlap\n\nBoth HH .......... 53\nEmployment HH .... 88\nMobility HH ...... 383"
    ax.text(
        SUMMARY_BOX_AXES[0],
        SUMMARY_BOX_AXES[1],
        summary,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.3,
        linespacing=1.08,
        bbox={
            "boxstyle": "round,pad=0.35,rounding_size=0.12",
            "facecolor": "white",
            "edgecolor": "#9a9a9a",
            "linewidth": 0.7,
            "alpha": 0.94,
        },
        zorder=10,
    )

    fig.savefig(OUTPUT_STEM.with_suffix(".pdf"), facecolor="white")
    fig.savefig(OUTPUT_STEM.with_suffix(".png"), dpi=DPI, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
