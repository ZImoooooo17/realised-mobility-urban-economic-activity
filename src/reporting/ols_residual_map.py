#!/usr/bin/env python3
"""Re-export the Chapter 4 OLS residual map from frozen prediction outputs.

This script does not estimate or transform the model. It joins the existing
standardised residuals to the final master geography and refreshes only the
publication presentation of the residual-map figure.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-casa-dissertation")

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import TwoSlopeNorm


ROOT = Path(__file__).resolve().parents[2]
MASTER_GPKG = ROOT / "data/processed/master/master_analysis_dataset_final_v1_0.gpkg"
PREDICTIONS = ROOT / "outputs/modelling/ols/ols_predictions.csv"
OUT = ROOT / "writing/figures/main/figure_4_9_ols_residual_map"


def main() -> None:
    gdf = gpd.read_file(MASTER_GPKG)
    residuals = pd.read_csv(PREDICTIONS)
    merged = gdf.merge(
        residuals[["LSOA21CD", "standardised_residual"]],
        on="LSOA21CD",
        how="left",
    )
    merged["residual_display"] = merged["standardised_residual"].clip(-3, 3)

    fig, ax = plt.subplots(figsize=(8.8, 8.0))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    norm = TwoSlopeNorm(vmin=-3, vcenter=0, vmax=3)
    merged.plot(
        column="residual_display",
        ax=ax,
        cmap="RdBu_r",
        norm=norm,
        linewidth=0.03,
        edgecolor="#ffffff",
        missing_kwds={"color": "#eeeeee", "edgecolor": "#ffffff", "linewidth": 0.02},
    )

    ax.set_axis_off()
    ax.set_title(
        "OLS Residual Map",
        loc="left",
        fontsize=14,
        fontweight="bold",
        color="#2b2b2b",
        pad=10,
    )

    sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=norm)
    sm._A = []
    cbar = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.035, shrink=0.72)
    cbar.set_label("Standardised residual", fontsize=10)
    cbar.set_ticks([-3, -2, -1, 0, 1, 2, 3])
    cbar.set_ticklabels(["≤-3", "-2", "-1", "0", "1", "2", "≥3"])
    cbar.ax.tick_params(labelsize=9)

    ax.text(
        0.01,
        -0.045,
        "Display scale clipped at ±3; residuals from combined OLS predictions.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        color="#666666",
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.12)
    fig.savefig(OUT.with_suffix(".png"), dpi=450, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


if __name__ == "__main__":
    main()
