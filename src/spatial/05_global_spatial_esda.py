"""Run core Spatial ESDA for the frozen master dataset.

Reads only:
data/processed/master/master_analysis_dataset_final_v1_0.gpkg

Writes maps and tables under outputs/esda/. This script intentionally does not
modify source data, transform variables, remove outliers, run Local Moran/LISA,
OLS, spatial regression, or machine learning.
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs/esda"
MAP_DIR = OUTPUT_DIR / "maps"
TABLE_DIR = OUTPUT_DIR / "tables"
MPLCONFIG_DIR = OUTPUT_DIR / ".mplconfig"
CONDA_PREFIX = Path(sys.prefix)
PROJ_DATA = CONDA_PREFIX / "share/proj"
MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))
if PROJ_DATA.exists():
    os.environ["PROJ_DATA"] = str(PROJ_DATA)
    os.environ["PROJ_LIB"] = str(PROJ_DATA)

import geopandas as gpd
import libpysal
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import esda
from esda import Moran
from libpysal.weights import Queen
from libpysal.weights.spatial_lag import lag_spatial
from PIL import Image


MASTER_GPKG = PROJECT_ROOT / "data/processed/master/master_analysis_dataset_final_v1_0.gpkg"
EXPECTED_ROWS = 4994
EXPECTED_CRS = "EPSG:27700"
REQUIRED_VARIABLES = [
    "LSOA21CD",
    "employees_count",
    "mobility_inflow_total",
    "flow_entropy",
    "weekday_weekend_ratio",
    "ptal_ai_mean",
    "population_density_km2",
]

MAP_SPECS = [
    {
        "variable": "employees_count",
        "title": "Workplace Employees",
        "filename": "figure_4_1_employees_count_map.png",
        "cmap": "YlOrRd",
        "legend_label": "Employees",
    },
    {
        "variable": "mobility_inflow_total",
        "title": "March 2026 Mobility Inflow",
        "filename": "figure_4_2_mobility_inflow_map.png",
        "cmap": "PuBuGn",
        "legend_label": "Extrapolated users",
    },
    {
        "variable": "flow_entropy",
        "title": "Origin Flow Entropy",
        "filename": "figure_4_3_flow_entropy_map.png",
        "cmap": "YlGnBu",
        "legend_label": "Entropy",
    },
    {
        "variable": "weekday_weekend_ratio",
        "title": "Weekday-Weekend Mobility Ratio",
        "filename": "figure_4_4_weekday_weekend_ratio_map.png",
        "cmap": "BuPu",
        "legend_label": "Ratio",
    },
    {
        "variable": "ptal_ai_mean",
        "title": "PTAL Mean Accessibility Index",
        "filename": "figure_4_5_ptal_map.png",
        "cmap": "GnBu",
        "legend_label": "PTAL AI",
    },
    {
        "variable": "population_density_km2",
        "title": "Population Density",
        "filename": "figure_4_6_population_density_map.png",
        "cmap": "YlGn",
        "legend_label": "Persons per km2",
    },
]

MORAN_VARIABLES = [
    {
        "variable": "employees_count",
        "label": "Workplace employees",
        "scatter_filename": "figure_4_7_moran_scatter_employment.png",
    },
    {
        "variable": "mobility_inflow_total",
        "label": "Mobility inflow total",
        "scatter_filename": "figure_4_8_moran_scatter_mobility.png",
    },
    {
        "variable": "flow_entropy",
        "label": "Flow entropy",
        "scatter_filename": "figure_4_9_moran_scatter_entropy.png",
    },
    {
        "variable": "weekday_weekend_ratio",
        "label": "Weekday-weekend ratio",
        "scatter_filename": "",
    },
    {
        "variable": "ptal_ai_mean",
        "label": "PTAL mean accessibility index",
        "scatter_filename": "",
    },
    {
        "variable": "population_density_km2",
        "label": "Population density per km2",
        "scatter_filename": "",
    },
]

PERMUTATIONS = 999
RANDOM_SEED = 20260722
CLASSIFICATION_METHOD = "Quantiles"
CLASSIFICATION_K = 5


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_dirs() -> None:
    MAP_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)


def log_environment() -> dict[str, str]:
    info = {
        "sys_executable": sys.executable,
        "python_version": sys.version.replace("\n", " "),
        "libpysal_version": libpysal.__version__,
        "esda_version": esda.__version__,
    }
    for key, value in info.items():
        print(f"{key}: {value}")
    return info


def read_frozen_gpkg() -> gpd.GeoDataFrame:
    if not MASTER_GPKG.exists():
        raise FileNotFoundError(MASTER_GPKG)
    return gpd.read_file(MASTER_GPKG)


def validate_input(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    geom_types = sorted(gdf.geometry.geom_type.dropna().unique().tolist())
    missing_required = [col for col in REQUIRED_VARIABLES if col not in gdf.columns]
    valid_geom_types = all(gt in {"Polygon", "MultiPolygon"} for gt in geom_types)
    rows = [
        {
            "check": "row_count",
            "value": len(gdf),
            "status": "PASS" if len(gdf) == EXPECTED_ROWS else "FAIL",
            "expected": EXPECTED_ROWS,
        },
        {
            "check": "crs",
            "value": str(gdf.crs),
            "status": "PASS" if str(gdf.crs).upper() == EXPECTED_CRS else "FAIL",
            "expected": EXPECTED_CRS,
        },
        {
            "check": "geometry_type",
            "value": "|".join(geom_types),
            "status": "PASS" if valid_geom_types else "FAIL",
            "expected": "Polygon or MultiPolygon",
        },
        {
            "check": "missing_geometries",
            "value": int(gdf.geometry.isna().sum()),
            "status": "PASS" if int(gdf.geometry.isna().sum()) == 0 else "FAIL",
            "expected": 0,
        },
        {
            "check": "invalid_geometries",
            "value": int((~gdf.geometry.is_valid).sum()),
            "status": "PASS" if int((~gdf.geometry.is_valid).sum()) == 0 else "FAIL",
            "expected": 0,
        },
        {
            "check": "duplicate_lsoa21cd",
            "value": int(gdf["LSOA21CD"].duplicated().sum()) if "LSOA21CD" in gdf else "missing_column",
            "status": "PASS" if "LSOA21CD" in gdf and int(gdf["LSOA21CD"].duplicated().sum()) == 0 else "FAIL",
            "expected": 0,
        },
        {
            "check": "required_variables",
            "value": "missing: " + "|".join(missing_required) if missing_required else "all present",
            "status": "PASS" if not missing_required else "FAIL",
            "expected": "|".join(REQUIRED_VARIABLES),
        },
    ]
    validation = pd.DataFrame(rows)
    validation.to_csv(TABLE_DIR / "spatial_input_validation.csv", index=False)
    if (validation["status"] == "FAIL").any():
        failed = validation[validation["status"] == "FAIL"][["check", "value", "expected"]]
        raise RuntimeError(f"Spatial input validation failed:\n{failed.to_string(index=False)}")
    return validation


def construct_queen_weights(gdf: gpd.GeoDataFrame):
    ids = gdf["LSOA21CD"].astype(str).tolist()
    weights = Queen.from_dataframe(gdf, ids=ids, use_index=False)
    weights.transform = "R"
    if list(weights.id_order) != ids:
        raise RuntimeError("Queen weights id_order does not match GeoDataFrame LSOA21CD order.")
    return weights


def weights_diagnostics(gdf: gpd.GeoDataFrame, weights) -> pd.DataFrame:
    ids = gdf["LSOA21CD"].astype(str).tolist()
    counts = np.array([len(weights.neighbors[area_id]) for area_id in ids], dtype=float)
    island_ids = [str(area_id) for area_id in weights.islands]
    diagnostics = pd.DataFrame(
        [
            {
                "weights_type": "libpysal.weights.Queen",
                "order": 1,
                "standardisation": "row-standardised",
                "observations": len(gdf),
                "mean_neighbours": float(np.mean(counts)),
                "median_neighbours": float(np.median(counts)),
                "minimum_neighbours": int(np.min(counts)),
                "maximum_neighbours": int(np.max(counts)),
                "number_of_islands": len(island_ids),
                "island_ids": "|".join(island_ids),
                "number_of_connected_components": int(weights.n_components),
            }
        ]
    )
    diagnostics.to_csv(TABLE_DIR / "spatial_weights_diagnostics.csv", index=False)
    return diagnostics


def plot_maps(gdf: gpd.GeoDataFrame) -> None:
    bounds = gdf.total_bounds
    xpad = (bounds[2] - bounds[0]) * 0.02
    ypad = (bounds[3] - bounds[1]) * 0.02
    print(f"Map classification method: {CLASSIFICATION_METHOD}, k={CLASSIFICATION_K}")
    for spec in MAP_SPECS:
        print(f"{spec['variable']}: classification method {CLASSIFICATION_METHOD}, k={CLASSIFICATION_K}")
        fig, ax = plt.subplots(figsize=(8, 8), facecolor="white")
        gdf.plot(
            column=spec["variable"],
            ax=ax,
            scheme=CLASSIFICATION_METHOD,
            k=CLASSIFICATION_K,
            cmap=spec["cmap"],
            linewidth=0.05,
            edgecolor="#f7f7f7",
            legend=True,
            legend_kwds={
                "title": spec["legend_label"],
                "loc": "lower left",
                "frameon": True,
                "fontsize": 8,
                "title_fontsize": 9,
            },
            missing_kwds={"color": "lightgrey", "label": "Missing"},
        )
        ax.set_title(spec["title"], fontsize=14, pad=12)
        ax.set_xlim(bounds[0] - xpad, bounds[2] + xpad)
        ax.set_ylim(bounds[1] - ypad, bounds[3] + ypad)
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(MAP_DIR / spec["filename"], dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)


def plot_moran_scatter(
    gdf: gpd.GeoDataFrame,
    weights,
    moran_results: pd.DataFrame,
) -> None:
    for spec in MORAN_VARIABLES:
        if not spec["scatter_filename"]:
            continue
        x_raw = gdf[spec["variable"]].to_numpy(dtype=float)
        x = (x_raw - np.mean(x_raw)) / np.std(x_raw, ddof=0)
        lag = lag_spatial(weights, x)
        slope = float(moran_results.loc[moran_results["variable"] == spec["variable"], "moran_i"].iloc[0])
        fig, ax = plt.subplots(figsize=(7, 7), facecolor="white")
        ax.scatter(x, lag, s=12, alpha=0.45, color="#2f6f9f", edgecolors="none")
        xx = np.array([np.nanmin(x), np.nanmax(x)])
        ax.plot(xx, slope * xx, color="#b33a3a", linewidth=2, label=f"Moran slope = {slope:.3f}")
        ax.axhline(0, color="#333333", linewidth=0.9)
        ax.axvline(0, color="#333333", linewidth=0.9)
        ax.set_xlabel(f"Standardised {spec['label']}")
        ax.set_ylabel(f"Spatial lag of standardised {spec['label']}")
        ax.set_title(f"Moran Scatterplot: {spec['label']} (I = {slope:.3f})", fontsize=13, pad=10)
        ax.legend(loc="upper left", frameon=True)
        ax.grid(True, linewidth=0.4, alpha=0.25)
        fig.tight_layout()
        fig.savefig(MAP_DIR / spec["scatter_filename"], dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)


def previous_moran_results() -> pd.DataFrame | None:
    path = TABLE_DIR / "global_moran_results.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def global_moran(gdf: gpd.GeoDataFrame, weights) -> pd.DataFrame:
    rows = []
    for spec in MORAN_VARIABLES:
        values = gdf[spec["variable"]].to_numpy(dtype=float)
        np.random.seed(RANDOM_SEED)
        result = Moran(values, weights, transformation="r", permutations=PERMUTATIONS, two_tailed=True)
        rows.append(
            {
                "variable": spec["variable"],
                "label": spec["label"],
                "weights": "libpysal Queen row-standardised",
                "permutations": PERMUTATIONS,
                "random_seed": RANDOM_SEED,
                "moran_i": float(result.I),
                "analytical_expected_i": float(result.EI),
                "analytical_z_score": float(result.z_norm),
                "analytical_p_value": float(result.p_norm),
                "permutation_p_value": float(result.p_sim),
                "permutation_mean_i": float(result.EI_sim),
                "permutation_standard_deviation": float(result.seI_sim),
                "permutation_z_score": float(result.z_sim),
            }
        )
    moran_df = pd.DataFrame(rows)
    moran_df.to_csv(TABLE_DIR / "global_moran_results.csv", index=False)
    return moran_df


def compare_previous_to_pysal(previous: pd.DataFrame | None, pysal_results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in pysal_results.iterrows():
        variable = row["variable"]
        prev_row = None
        if previous is not None and "variable" in previous.columns:
            matches = previous[previous["variable"] == variable]
            if not matches.empty:
                prev_row = matches.iloc[0]

        def previous_value(*names: str):
            if prev_row is None:
                return np.nan
            for name in names:
                if name in prev_row.index:
                    return prev_row[name]
            return np.nan

        previous_i = previous_value("moran_i")
        pysal_i = row["moran_i"]
        previous_z = previous_value("z_score", "analytical_z_score", "permutation_z_score")
        pysal_z = row["analytical_z_score"]
        previous_p = previous_value("permutation_p_value", "p_sim")
        pysal_p = row["permutation_p_value"]
        abs_i_diff = abs(float(previous_i) - float(pysal_i)) if pd.notna(previous_i) else np.nan
        abs_z_diff = abs(float(previous_z) - float(pysal_z)) if pd.notna(previous_z) else np.nan
        rows.append(
            {
                "variable": variable,
                "previous_moran_i": previous_i,
                "pysal_moran_i": pysal_i,
                "absolute_moran_i_difference": abs_i_diff,
                "moran_i_difference_gt_0_000001": bool(abs_i_diff > 0.000001) if pd.notna(abs_i_diff) else pd.NA,
                "previous_z_score": previous_z,
                "pysal_analytical_z_score": pysal_z,
                "absolute_z_score_difference": abs_z_diff,
                "previous_permutation_p_value": previous_p,
                "pysal_permutation_p_value": pysal_p,
                "notes": "Differences may reflect libpysal Queen topology and esda inference conventions versus prior custom implementation."
                if pd.notna(abs_i_diff) and abs_i_diff > 0.000001
                else "",
            }
        )
    comparison = pd.DataFrame(rows)
    comparison.to_csv(TABLE_DIR / "global_moran_comparison_custom_vs_pysal.csv", index=False)
    return comparison


def verify_outputs(
    source_hash_before: str,
    source_mtime_before: float,
    moran_df: pd.DataFrame,
    weights,
    env_info: dict[str, str],
) -> pd.DataFrame:
    required_pngs = [spec["filename"] for spec in MAP_SPECS] + [
        spec["scatter_filename"] for spec in MORAN_VARIABLES if spec["scatter_filename"]
    ]
    required_tables = [
        TABLE_DIR / "spatial_input_validation.csv",
        TABLE_DIR / "spatial_weights_diagnostics.csv",
        TABLE_DIR / "global_moran_results.csv",
        TABLE_DIR / "global_moran_comparison_custom_vs_pysal.csv",
    ]
    rows = []

    def add(check: str, value, status: str, notes: str = "") -> None:
        rows.append({"check": check, "value": value, "status": status, "notes": notes})

    for table in required_tables:
        add(f"table_exists__{table.name}", table.exists(), "PASS" if table.exists() else "FAIL", str(table))
    add(
        "python_executable_recorded",
        env_info["sys_executable"],
        "PASS",
    )
    add("libpysal_imported_successfully", env_info["libpysal_version"], "PASS")
    add("esda_imported_successfully", env_info["esda_version"], "PASS")
    add("official_queen_object_used", type(weights).__module__ + "." + type(weights).__name__, "PASS" if isinstance(weights, Queen) else "FAIL")
    add("observations_aligned_with_weights", weights.n, "PASS" if weights.n == EXPECTED_ROWS else "FAIL")
    add("islands_documented", len(weights.islands), "PASS")
    for name in required_pngs:
        path = MAP_DIR / name
        if not path.exists():
            add(f"png_exists__{name}", False, "FAIL", str(path))
            continue
        try:
            with Image.open(path) as image:
                image.verify()
            readable = True
        except Exception as exc:
            readable = False
            add(f"png_readable__{name}", False, "FAIL", str(exc))
        else:
            add(f"png_readable__{name}", readable, "PASS", str(path))
    finite_stats = bool(
        np.isfinite(
            moran_df[
                [
                    "moran_i",
                    "analytical_expected_i",
                    "analytical_z_score",
                    "analytical_p_value",
                    "permutation_p_value",
                    "permutation_mean_i",
                    "permutation_standard_deviation",
                ]
            ].to_numpy()
        ).all()
    )
    add("moran_statistics_finite", finite_stats, "PASS" if finite_stats else "FAIL")
    source_hash_after = file_sha256(MASTER_GPKG)
    source_mtime_after = MASTER_GPKG.stat().st_mtime
    add("source_hash_unchanged", source_hash_before == source_hash_after, "PASS" if source_hash_before == source_hash_after else "FAIL")
    add(
        "source_mtime_unchanged",
        source_mtime_before == source_mtime_after,
        "PASS" if source_mtime_before == source_mtime_after else "FAIL",
    )
    add("no_regression_run", True, "PASS")
    add("no_local_moran_or_lisa_run", True, "PASS")
    qa = pd.DataFrame(rows)
    qa.to_csv(TABLE_DIR / "spatial_esda_qa.csv", index=False)
    if (qa["status"] == "FAIL").any():
        raise RuntimeError(f"Spatial ESDA QA failed:\n{qa[qa['status'] == 'FAIL'].to_string(index=False)}")
    return qa


def main() -> None:
    ensure_dirs()
    env_info = log_environment()
    source_hash_before = file_sha256(MASTER_GPKG)
    source_mtime_before = MASTER_GPKG.stat().st_mtime
    gdf = read_frozen_gpkg()
    validate_input(gdf)
    previous_results = previous_moran_results()
    weights = construct_queen_weights(gdf)
    diagnostics = weights_diagnostics(gdf, weights)
    plot_maps(gdf)
    moran_df = global_moran(gdf, weights)
    comparison = compare_previous_to_pysal(previous_results, moran_df)
    plot_moran_scatter(gdf, weights, moran_df)
    qa = verify_outputs(source_hash_before, source_mtime_before, moran_df, weights, env_info)

    print("Spatial ESDA complete.")
    print("Weights summary:")
    print(diagnostics.to_string(index=False))
    print("Global Moran's I results:")
    print(
        moran_df[
            [
                "variable",
                "moran_i",
                "analytical_expected_i",
                "analytical_z_score",
                "analytical_p_value",
                "permutation_p_value",
                "permutation_mean_i",
                "permutation_standard_deviation",
            ]
        ].to_string(index=False)
    )
    print("Comparison with previous custom results:")
    print(comparison.to_string(index=False))
    print("QA summary:")
    print(qa["status"].value_counts().to_string())
    print("No regression, Local Moran/LISA, OLS, spatial regression, machine learning, or dataset modification was performed.")


if __name__ == "__main__":
    main()
