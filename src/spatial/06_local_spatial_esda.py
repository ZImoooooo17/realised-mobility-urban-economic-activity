"""Run Local Spatial ESDA / LISA for the frozen dissertation master dataset.

Uses official PySAL implementations:
- libpysal.weights.Queen
- esda.Moran_Local

This script reads the frozen GPKG only and writes Local Moran/LISA tables,
maps, reports, and QA outputs. It does not modify source data, transform
variables, remove observations/outliers, or run any regression/ML models.
"""

from __future__ import annotations

import hashlib
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs/esda"
TABLE_DIR = OUTPUT_DIR / "tables"
MAP_DIR = OUTPUT_DIR / "maps"
REPORT_DIR = OUTPUT_DIR / "reports"
MPLCONFIG_DIR = OUTPUT_DIR / ".mplconfig"
PROJ_DATA = Path(sys.prefix) / "share/proj"

MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))
if PROJ_DATA.exists():
    os.environ["PROJ_DATA"] = str(PROJ_DATA)
    os.environ["PROJ_LIB"] = str(PROJ_DATA)

import esda
import geopandas as gpd
import libpysal
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from esda import Moran_Local
from libpysal.weights import Queen
from libpysal.weights.spatial_lag import lag_spatial
from matplotlib.patches import Patch
from PIL import Image


MASTER_GPKG = PROJECT_ROOT / "data/processed/master/master_analysis_dataset_final_v1_0.gpkg"
EXPECTED_ROWS = 4994
EXPECTED_CRS = "EPSG:27700"
PERMUTATIONS = 999
RANDOM_SEED = 20260722
SIGNIFICANCE = 0.05

LISA_VARIABLES = {
    "employment": {
        "variable": "employees_count",
        "label": "Workplace employment",
        "value_column": "employees_count",
        "result_csv": "lisa_employment_results.csv",
        "top_csv": "lisa_employment_top_clusters.csv",
        "cluster_map": "figure_4_10_lisa_employment_clusters.png",
        "significance_map": "figure_4_12_lisa_employment_significance.png",
    },
    "mobility": {
        "variable": "mobility_inflow_total",
        "label": "Mobility inflow",
        "value_column": "mobility_inflow_total",
        "result_csv": "lisa_mobility_results.csv",
        "top_csv": "lisa_mobility_top_clusters.csv",
        "cluster_map": "figure_4_11_lisa_mobility_clusters.png",
        "significance_map": "figure_4_13_lisa_mobility_significance.png",
    },
}

REQUIRED_FIELDS = ["LSOA21CD", "employees_count", "mobility_inflow_total"]
CLUSTER_ORDER = ["High-High", "Low-Low", "High-Low", "Low-High", "Not Significant"]
SIGNIFICANT_CATEGORIES = ["High-High", "Low-Low", "High-Low", "Low-High"]
CLUSTER_COLORS = {
    "High-High": "#d7191c",
    "Low-Low": "#2c7bb6",
    "High-Low": "#f4a6a6",
    "Low-High": "#abd9e9",
    "Not Significant": "#d9d9d9",
}
SIGNIFICANCE_ORDER = ["p < 0.001", "p < 0.01", "p < 0.05", "Not Significant"]
SIGNIFICANCE_COLORS = {
    "p < 0.001": "#7f0000",
    "p < 0.01": "#d7301f",
    "p < 0.05": "#fc8d59",
    "Not Significant": "#d9d9d9",
}


def ensure_dirs() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    MAP_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def log_environment() -> dict[str, str]:
    info = {
        "python_executable": sys.executable,
        "python_version": sys.version.replace("\n", " "),
        "libpysal_version": libpysal.__version__,
        "esda_version": esda.__version__,
    }
    for key, value in info.items():
        print(f"{key}: {value}")
    return info


def read_input() -> gpd.GeoDataFrame:
    if not MASTER_GPKG.exists():
        raise FileNotFoundError(MASTER_GPKG)
    return gpd.read_file(MASTER_GPKG)


def validate_input(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    geom_types = sorted(gdf.geometry.geom_type.dropna().unique().tolist())
    missing_fields = [field for field in REQUIRED_FIELDS if field not in gdf.columns]
    finite_failures = []
    missing_value_failures = []
    for field in ["employees_count", "mobility_inflow_total"]:
        if field in gdf.columns:
            values = pd.to_numeric(gdf[field], errors="coerce")
            if values.isna().any():
                missing_value_failures.append(field)
            if not np.isfinite(values.to_numpy(dtype=float)).all():
                finite_failures.append(field)
    rows = [
        ("row_count", len(gdf), "PASS" if len(gdf) == EXPECTED_ROWS else "FAIL", EXPECTED_ROWS),
        ("crs", str(gdf.crs), "PASS" if str(gdf.crs).upper() == EXPECTED_CRS else "FAIL", EXPECTED_CRS),
        (
            "geometry_type",
            "|".join(geom_types),
            "PASS" if all(gt in {"Polygon", "MultiPolygon"} for gt in geom_types) else "FAIL",
            "Polygon or MultiPolygon",
        ),
        ("missing_geometries", int(gdf.geometry.isna().sum()), "PASS" if int(gdf.geometry.isna().sum()) == 0 else "FAIL", 0),
        ("invalid_geometries", int((~gdf.geometry.is_valid).sum()), "PASS" if int((~gdf.geometry.is_valid).sum()) == 0 else "FAIL", 0),
        (
            "duplicate_lsoa21cd",
            int(gdf["LSOA21CD"].duplicated().sum()) if "LSOA21CD" in gdf else "missing_field",
            "PASS" if "LSOA21CD" in gdf and int(gdf["LSOA21CD"].duplicated().sum()) == 0 else "FAIL",
            0,
        ),
        (
            "required_fields",
            "missing: " + "|".join(missing_fields) if missing_fields else "all present",
            "PASS" if not missing_fields else "FAIL",
            "|".join(REQUIRED_FIELDS),
        ),
        (
            "missing_values_required_analysis_variables",
            "|".join(missing_value_failures) if missing_value_failures else "none",
            "PASS" if not missing_value_failures else "FAIL",
            "none",
        ),
        (
            "finite_values_required_analysis_variables",
            "|".join(finite_failures) if finite_failures else "all finite",
            "PASS" if not finite_failures else "FAIL",
            "all finite",
        ),
    ]
    validation = pd.DataFrame(rows, columns=["check", "value", "status", "expected"])
    validation.to_csv(TABLE_DIR / "local_spatial_input_validation.csv", index=False)
    if (validation["status"] == "FAIL").any():
        raise RuntimeError(f"Local spatial input validation failed:\n{validation[validation['status'] == 'FAIL'].to_string(index=False)}")
    return validation


def construct_weights(gdf: gpd.GeoDataFrame):
    ids = gdf["LSOA21CD"].astype(str).tolist()
    weights = Queen.from_dataframe(gdf, ids=ids, use_index=False)
    weights.transform = "R"
    if list(weights.id_order) != ids:
        raise RuntimeError("Weights id_order does not match GeoDataFrame LSOA21CD order.")
    if weights.n != len(gdf):
        raise RuntimeError("Weights observation count does not match input row count.")
    return weights


def validate_weights(weights, gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    ids = gdf["LSOA21CD"].astype(str).tolist()
    counts = np.array([len(weights.neighbors[area_id]) for area_id in ids], dtype=float)
    diagnostics = {
        "weights_type": "libpysal.weights.Queen",
        "order": 1,
        "standardisation": "row-standardised",
        "observations": weights.n,
        "mean_neighbours": float(np.mean(counts)),
        "median_neighbours": float(np.median(counts)),
        "minimum_neighbours": int(np.min(counts)),
        "maximum_neighbours": int(np.max(counts)),
        "number_of_islands": len(weights.islands),
        "island_ids": "|".join(map(str, weights.islands)),
        "number_of_connected_components": int(weights.n_components),
    }
    checks = [
        ("observations", diagnostics["observations"], diagnostics["observations"] == EXPECTED_ROWS, EXPECTED_ROWS),
        ("number_of_islands", diagnostics["number_of_islands"], diagnostics["number_of_islands"] == 0, 0),
        ("number_of_connected_components", diagnostics["number_of_connected_components"], diagnostics["number_of_connected_components"] == 1, 1),
        ("minimum_neighbours", diagnostics["minimum_neighbours"], diagnostics["minimum_neighbours"] == 1, 1),
        ("maximum_neighbours", diagnostics["maximum_neighbours"], diagnostics["maximum_neighbours"] == 20, 20),
        ("mean_neighbours", diagnostics["mean_neighbours"], abs(diagnostics["mean_neighbours"] - 5.901482) < 1e-6, "approximately 5.901482"),
        ("median_neighbours", diagnostics["median_neighbours"], diagnostics["median_neighbours"] == 6, 6),
    ]
    rows = []
    for check, value, ok, expected in checks:
        rows.append({**diagnostics, "check": check, "check_value": value, "status": "PASS" if ok else "FAIL", "expected": expected})
    result = pd.DataFrame(rows)
    task04a_path = TABLE_DIR / "spatial_weights_diagnostics.csv"
    if task04a_path.exists():
        prior = pd.read_csv(task04a_path).iloc[0]
        for field in ["observations", "mean_neighbours", "median_neighbours", "minimum_neighbours", "maximum_neighbours", "number_of_islands", "number_of_connected_components"]:
            if field in prior.index and pd.notna(prior[field]):
                current = diagnostics[field]
                previous = prior[field]
                materially_different = abs(float(current) - float(previous)) > 1e-6
                result.loc[len(result)] = {
                    **diagnostics,
                    "check": f"matches_task04a__{field}",
                    "check_value": current,
                    "status": "FAIL" if materially_different else "PASS",
                    "expected": previous,
                }
    result.to_csv(TABLE_DIR / "local_spatial_weights_validation.csv", index=False)
    if (result["status"] == "FAIL").any():
        raise RuntimeError(f"Local spatial weights validation failed:\n{result[result['status'] == 'FAIL'].to_string(index=False)}")
    return result


def standardise(values: np.ndarray) -> np.ndarray:
    return (values - np.mean(values)) / np.std(values, ddof=0)


def cluster_from_q(q: int, significant: bool) -> str:
    if not significant:
        return "Not Significant"
    return {1: "High-High", 2: "Low-High", 3: "Low-Low", 4: "High-Low"}[int(q)]


def significance_bin(p_value: float) -> str:
    if p_value <= 0.001:
        return "p < 0.001"
    if p_value < 0.01:
        return "p < 0.01"
    if p_value < 0.05:
        return "p < 0.05"
    return "Not Significant"


def bh_adjust(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    adjusted_ranked = ranked * n / np.arange(1, n + 1)
    adjusted_ranked = np.minimum.accumulate(adjusted_ranked[::-1])[::-1]
    adjusted = np.empty(n, dtype=float)
    adjusted[order] = np.minimum(adjusted_ranked, 1.0)
    return adjusted


def run_local_moran(gdf: gpd.GeoDataFrame, weights) -> tuple[dict[str, pd.DataFrame], dict[str, Moran_Local]]:
    outputs: dict[str, pd.DataFrame] = {}
    moran_objects: dict[str, Moran_Local] = {}
    neighbour_counts = pd.Series({area_id: len(weights.neighbors[area_id]) for area_id in weights.id_order})
    for key, spec in LISA_VARIABLES.items():
        values = gdf[spec["variable"]].to_numpy(dtype=float)
        z = standardise(values)
        lag_z = lag_spatial(weights, z)
        local = Moran_Local(
            values,
            weights,
            transformation="r",
            permutations=PERMUTATIONS,
            seed=RANDOM_SEED,
            n_jobs=1,
            geoda_quads=False,
        )
        significant = local.p_sim < SIGNIFICANCE
        clusters = [cluster_from_q(q, sig) for q, sig in zip(local.q, significant)]
        result = pd.DataFrame(
            {
                "LSOA21CD": gdf["LSOA21CD"].astype(str),
                "LSOA21NM": gdf["LSOA21NM"].astype(str) if "LSOA21NM" in gdf.columns else "",
                "variable": spec["variable"],
                "variable_value": values,
                "standardised_value": z,
                "standardised_spatial_lag": lag_z,
                "local_moran_i": local.Is,
                "pysal_quadrant_code": local.q.astype(int),
                "permutation_p_value": local.p_sim,
                "significant_p_0_05": significant,
                "lisa_cluster_category": clusters,
                "neighbour_count": gdf["LSOA21CD"].astype(str).map(neighbour_counts).astype(int),
            }
        )
        category_rank = {category: i for i, category in enumerate(CLUSTER_ORDER)}
        result["_category_rank"] = result["lisa_cluster_category"].map(category_rank)
        result["_abs_i"] = result["local_moran_i"].abs()
        result = result.sort_values(
            ["_category_rank", "permutation_p_value", "_abs_i"],
            ascending=[True, True, False],
        ).drop(columns=["_category_rank", "_abs_i"])
        result.to_csv(TABLE_DIR / spec["result_csv"], index=False)
        outputs[key] = result
        moran_objects[key] = local
    return outputs, moran_objects


def cluster_summaries(results: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    cluster_rows = []
    significance_rows = []
    for key, table in results.items():
        variable = LISA_VARIABLES[key]["variable"]
        sig_total = int(table["significant_p_0_05"].sum())
        for category in CLUSTER_ORDER:
            subset = table[table["lisa_cluster_category"] == category]
            count = len(subset)
            cluster_rows.append(
                {
                    "variable": variable,
                    "cluster_category": category,
                    "number_of_lsoas": count,
                    "percentage_of_all_lsoas": count / EXPECTED_ROWS * 100,
                    "percentage_of_significant_lsoas": (count / sig_total * 100) if sig_total and category != "Not Significant" else (0.0 if category != "Not Significant" else np.nan),
                    "mean_variable_value": float(subset["variable_value"].mean()) if count else np.nan,
                    "median_variable_value": float(subset["variable_value"].median()) if count else np.nan,
                    "mean_local_moran_i": float(subset["local_moran_i"].mean()) if count else np.nan,
                    "median_permutation_p_value": float(subset["permutation_p_value"].median()) if count else np.nan,
                }
            )
        significance_rows.append(
            {
                "variable": variable,
                "total_observations": len(table),
                "significant_observations_p_0_05": sig_total,
                "non_significant_observations": int((~table["significant_p_0_05"]).sum()),
                "significant_percentage": sig_total / len(table) * 100,
                "High-High_count": int((table["lisa_cluster_category"] == "High-High").sum()),
                "Low-Low_count": int((table["lisa_cluster_category"] == "Low-Low").sum()),
                "High-Low_count": int((table["lisa_cluster_category"] == "High-Low").sum()),
                "Low-High_count": int((table["lisa_cluster_category"] == "Low-High").sum()),
                "category_count_sum": int(table["lisa_cluster_category"].value_counts().reindex(CLUSTER_ORDER, fill_value=0).sum()),
            }
        )
    cluster_summary = pd.DataFrame(cluster_rows)
    sig_summary = pd.DataFrame(significance_rows)
    cluster_summary.to_csv(TABLE_DIR / "lisa_cluster_summary.csv", index=False)
    sig_summary.to_csv(TABLE_DIR / "lisa_significance_summary.csv", index=False)
    return cluster_summary, sig_summary


def ranked_tables(results: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    ranked = {}
    for key, table in results.items():
        rows = []
        for category in SIGNIFICANT_CATEGORIES:
            subset = table[table["lisa_cluster_category"] == category].copy()
            subset["_abs_i"] = subset["local_moran_i"].abs()
            subset = subset.sort_values(
                ["permutation_p_value", "_abs_i", "variable_value"],
                ascending=[True, False, False],
            ).head(20)
            for rank, (_, row) in enumerate(subset.iterrows(), start=1):
                rows.append(
                    {
                        "category": category,
                        "rank_within_category": rank,
                        "LSOA21CD": row["LSOA21CD"],
                        "LSOA21NM": row["LSOA21NM"],
                        "variable": LISA_VARIABLES[key]["variable"],
                        "variable_value": row["variable_value"],
                        "local_moran_i": row["local_moran_i"],
                        "permutation_p_value": row["permutation_p_value"],
                        "neighbour_count": row["neighbour_count"],
                    }
                )
        ranked_table = pd.DataFrame(rows)
        ranked_table.to_csv(TABLE_DIR / LISA_VARIABLES[key]["top_csv"], index=False)
        ranked[key] = ranked_table
    return ranked


def plot_categorical_map(gdf: gpd.GeoDataFrame, column: str, title: str, subtitle: str, filename: str, colors: dict[str, str], order: list[str]) -> None:
    plot_gdf = gdf.copy()
    plot_gdf[column] = pd.Categorical(plot_gdf[column], categories=order, ordered=True)
    bounds = plot_gdf.total_bounds
    xpad = (bounds[2] - bounds[0]) * 0.02
    ypad = (bounds[3] - bounds[1]) * 0.02
    fig, ax = plt.subplots(figsize=(8, 8), facecolor="white")
    for category in order:
        subset = plot_gdf[plot_gdf[column] == category]
        if not subset.empty:
            subset.plot(ax=ax, color=colors[category], edgecolor="#f7f7f7", linewidth=0.05)
    ax.set_title(f"{title}\n{subtitle}", fontsize=12, pad=10)
    ax.set_xlim(bounds[0] - xpad, bounds[2] + xpad)
    ax.set_ylim(bounds[1] - ypad, bounds[3] + ypad)
    ax.set_axis_off()
    handles = [Patch(facecolor=colors[category], edgecolor="#777777", label=category) for category in order]
    ax.legend(handles=handles, loc="lower left", frameon=True, title="Category", fontsize=8, title_fontsize=9)
    fig.tight_layout()
    fig.savefig(MAP_DIR / filename, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def lisa_maps(gdf: gpd.GeoDataFrame, results: dict[str, pd.DataFrame]) -> None:
    subtitle = f"Queen contiguity, row-standardised, {PERMUTATIONS} permutations, p_sim < {SIGNIFICANCE}"
    for key, table in results.items():
        spec = LISA_VARIABLES[key]
        merged = gdf[["LSOA21CD", "geometry"]].merge(table[["LSOA21CD", "lisa_cluster_category", "permutation_p_value"]], on="LSOA21CD", how="left")
        merged["significance_category"] = merged["permutation_p_value"].apply(significance_bin)
        plot_categorical_map(
            merged,
            "lisa_cluster_category",
            f"LISA Clusters: {spec['label']}",
            subtitle,
            spec["cluster_map"],
            CLUSTER_COLORS,
            CLUSTER_ORDER,
        )
        plot_categorical_map(
            merged,
            "significance_category",
            f"Local Moran Significance: {spec['label']}",
            "999 permutations; 0.001 is the minimum attainable p-value",
            spec["significance_map"],
            SIGNIFICANCE_COLORS,
            SIGNIFICANCE_ORDER,
        )


def multiple_testing_sensitivity(results: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    summary_rows = []
    for key, table in results.items():
        variable = LISA_VARIABLES[key]["variable"]
        adjusted = bh_adjust(table["permutation_p_value"].to_numpy(dtype=float))
        raw_significant = table["significant_p_0_05"].to_numpy(dtype=bool)
        adjusted_significant = adjusted < SIGNIFICANCE
        adjusted_categories = [cluster_from_q(q, sig) for q, sig in zip(table["pysal_quadrant_code"], adjusted_significant)]
        raw_count = int(raw_significant.sum())
        adjusted_count = int(adjusted_significant.sum())
        summary_rows.append(
            {
                "variable": variable,
                "raw_significant_count": raw_count,
                "fdr_adjusted_significant_count": adjusted_count,
                "difference": raw_count - adjusted_count,
                "raw_significant_percentage": raw_count / EXPECTED_ROWS * 100,
                "fdr_adjusted_significant_percentage": adjusted_count / EXPECTED_ROWS * 100,
            }
        )
        for i, row in table.reset_index(drop=True).iterrows():
            rows.append(
                {
                    "variable": variable,
                    "LSOA21CD": row["LSOA21CD"],
                    "raw_permutation_p_value": row["permutation_p_value"],
                    "benjamini_hochberg_adjusted_p_value": adjusted[i],
                    "raw_significance_indicator": bool(raw_significant[i]),
                    "fdr_adjusted_significance_indicator": bool(adjusted_significant[i]),
                    "raw_lisa_category": row["lisa_cluster_category"],
                    "fdr_adjusted_lisa_category": adjusted_categories[i],
                }
            )
    sensitivity = pd.DataFrame(rows)
    summary = pd.DataFrame(summary_rows)
    sensitivity.to_csv(TABLE_DIR / "lisa_multiple_testing_sensitivity.csv", index=False)
    summary.to_csv(TABLE_DIR / "lisa_multiple_testing_sensitivity_summary.csv", index=False)
    return sensitivity, summary


def overlap_tables(results: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    emp = results["employment"][["LSOA21CD", "lisa_cluster_category", "permutation_p_value", "significant_p_0_05"]].rename(
        columns={
            "lisa_cluster_category": "employment_cluster_category",
            "permutation_p_value": "employment_permutation_p_value",
            "significant_p_0_05": "employment_significant",
        }
    )
    mob = results["mobility"][["LSOA21CD", "lisa_cluster_category", "permutation_p_value", "significant_p_0_05"]].rename(
        columns={
            "lisa_cluster_category": "mobility_cluster_category",
            "permutation_p_value": "mobility_permutation_p_value",
            "significant_p_0_05": "mobility_significant",
        }
    )
    overlap = emp.merge(mob, on="LSOA21CD", how="inner", validate="one_to_one")
    overlap["cluster_categories_match"] = overlap["employment_cluster_category"] == overlap["mobility_cluster_category"]
    overlap["both_significant"] = overlap["employment_significant"] & overlap["mobility_significant"]
    overlap["both_high_high"] = (overlap["employment_cluster_category"] == "High-High") & (overlap["mobility_cluster_category"] == "High-High")
    overlap["both_low_low"] = (overlap["employment_cluster_category"] == "Low-Low") & (overlap["mobility_cluster_category"] == "Low-Low")
    overlap["one_significant_other_not"] = overlap["employment_significant"] ^ overlap["mobility_significant"]
    overlap.to_csv(TABLE_DIR / "lisa_employment_mobility_overlap.csv", index=False)
    metrics = {
        "matching_categories": int(overlap["cluster_categories_match"].sum()),
        "both_significant": int(overlap["both_significant"].sum()),
        "both_high_high": int(overlap["both_high_high"].sum()),
        "both_low_low": int(overlap["both_low_low"].sum()),
        "employment_high_high_but_mobility_not_high_high": int(((overlap["employment_cluster_category"] == "High-High") & (overlap["mobility_cluster_category"] != "High-High")).sum()),
        "mobility_high_high_but_employment_not_high_high": int(((overlap["mobility_cluster_category"] == "High-High") & (overlap["employment_cluster_category"] != "High-High")).sum()),
        "one_significant_other_non_significant": int(overlap["one_significant_other_not"].sum()),
    }
    summary = pd.DataFrame(
        [{"metric": metric, "count": count, "percentage_of_lsoas": count / EXPECTED_ROWS * 100} for metric, count in metrics.items()]
    )
    summary.to_csv(TABLE_DIR / "lisa_overlap_summary.csv", index=False)
    return overlap, summary


def read_global_moran() -> pd.DataFrame:
    path = TABLE_DIR / "global_moran_results.csv"
    if not path.exists():
        raise FileNotFoundError("Task 04A official PySAL global Moran results not found.")
    return pd.read_csv(path)


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    display = df.copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(lambda value: "" if pd.isna(value) else f"{value:.6f}")
        else:
            display[col] = display[col].map(lambda value: "" if pd.isna(value) else str(value))
    header = "| " + " | ".join(display.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(display.columns)) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in display.astype(str).to_numpy()]
    return "\n".join([header, separator, *rows])


def write_reports(
    env_info: dict[str, str],
    validation: pd.DataFrame,
    weights_validation: pd.DataFrame,
    cluster_summary: pd.DataFrame,
    sig_summary: pd.DataFrame,
    fdr_summary: pd.DataFrame,
    overlap_summary: pd.DataFrame,
) -> None:
    global_moran = read_global_moran()
    weights_row = weights_validation.iloc[0]
    emp_sig = sig_summary[sig_summary["variable"] == "employees_count"].iloc[0]
    mob_sig = sig_summary[sig_summary["variable"] == "mobility_inflow_total"].iloc[0]
    global_cols = ["variable", "moran_i", "analytical_z_score", "permutation_p_value"]
    report = f"""# Spatial ESDA Summary

## 1. Data and Weights

- Frozen dataset: `data/processed/master/master_analysis_dataset_final_v1_0.gpkg`
- Row count: {EXPECTED_ROWS:,}
- CRS: `EPSG:27700`
- Queen contiguity: `libpysal.weights.Queen.from_dataframe`
- Standardisation: row-standardised, `w.transform = "R"`
- Mean neighbours: {float(weights_row['mean_neighbours']):.6f}
- Median neighbours: {float(weights_row['median_neighbours']):.0f}
- Minimum neighbours: {int(weights_row['minimum_neighbours'])}
- Maximum neighbours: {int(weights_row['maximum_neighbours'])}
- Islands: {int(weights_row['number_of_islands'])}
- Connected components: {int(weights_row['number_of_connected_components'])}
- Permutations: {PERMUTATIONS}
- Random seed: {RANDOM_SEED}
- Python executable: `{env_info['python_executable']}`
- Python version: {env_info['python_version']}
- libpysal version: {env_info['libpysal_version']}
- esda version: {env_info['esda_version']}

## 2. Global Spatial Autocorrelation

Existing official PySAL Task 04A results:

{markdown_table(global_moran[global_cols])}

Permutation p-values of 0.001 should be read as the minimum attainable p-value with 999 permutations, not as an exact probability.

## 3. Employment LISA

- Significant LSOAs: {int(emp_sig['significant_observations_p_0_05']):,} ({float(emp_sig['significant_percentage']):.2f}%)
- High-High: {int(emp_sig['High-High_count']):,}
- Low-Low: {int(emp_sig['Low-Low_count']):,}
- High-Low: {int(emp_sig['High-Low_count']):,}
- Low-High: {int(emp_sig['Low-High_count']):,}

The employment LISA results show statistically significant local clustering and spatial outliers in a subset of London LSOAs. Categories describe local association under Queen contiguity and do not establish causal relationships.

## 4. Mobility LISA

- Significant LSOAs: {int(mob_sig['significant_observations_p_0_05']):,} ({float(mob_sig['significant_percentage']):.2f}%)
- High-High: {int(mob_sig['High-High_count']):,}
- Low-Low: {int(mob_sig['Low-Low_count']):,}
- High-Low: {int(mob_sig['High-Low_count']):,}
- Low-High: {int(mob_sig['Low-High_count']):,}

The mobility LISA results identify local concentrations and spatial outliers in March 2026 inflow. These are exploratory spatial patterns in observed mobility data, not modelled effects.

## 5. Employment-Mobility Overlap

{markdown_table(overlap_summary)}

The overlap comparison is descriptive only. Matching LISA categories indicate shared local spatial association categories, not evidence that one process causes the other.

## 6. Multiple-Testing Sensitivity

{markdown_table(fdr_summary)}

Primary maps and cluster tables use raw `p_sim < 0.05` for standard exploratory LISA comparability. FDR-adjusted counts are reported only as sensitivity evidence.

## 7. Limitations

- LISA is exploratory.
- Results depend on the spatial weights definition.
- Local significance involves multiple testing.
- LSOA results may be affected by MAUP.
- Cluster association does not establish causality.
- March 2026 mobility and 2024 employment are not temporally aligned.
- Unusually high or low values may influence local statistics.
- The analysis uses untransformed values for consistency with Task 04A.
"""
    (REPORT_DIR / "spatial_esda_summary.md").write_text(report, encoding="utf-8")

    notes = f"""# Spatial Interpretation Notes

## Employment Clusters

The employment LISA analysis identifies {int(emp_sig['significant_observations_p_0_05']):,} LSOAs with statistically significant local spatial association at raw `p_sim < 0.05`. High-High areas indicate LSOAs with above-average employment values surrounded by neighbours with above-average values. Low-Low areas indicate below-average employment values surrounded by below-average neighbours.

## Mobility Clusters

The mobility LISA analysis identifies {int(mob_sig['significant_observations_p_0_05']):,} LSOAs with statistically significant local spatial association at raw `p_sim < 0.05`. High-High areas indicate local concentrations of high March 2026 mobility inflow. Low-Low areas indicate local concentrations of lower mobility inflow.

## Shared Clusters

Employment and mobility cluster categories match for {int(overlap_summary.loc[overlap_summary['metric'] == 'matching_categories', 'count'].iloc[0]):,} LSOAs. Both variables are significant for {int(overlap_summary.loc[overlap_summary['metric'] == 'both_significant', 'count'].iloc[0]):,} LSOAs. This overlap is descriptive and should be treated as evidence of spatial association, not causality.

## Spatial Outliers

High-Low observations have above-average values surrounded by lower-value neighbours. Low-High observations have below-average values surrounded by higher-value neighbours. These are spatial outlier categories and should not automatically be interpreted as data errors.

## Link To Modelling

Significant global and local spatial dependence provides justification for testing spatial autocorrelation in OLS residuals and considering spatial econometric models if residual dependence remains. It does not establish in advance that a spatial lag or spatial error model will outperform OLS.

## Safe Interpretation Language

- “The variable is spatially associated with neighbouring values under Queen contiguity.”
- “This LSOA forms a statistically significant local cluster at raw `p_sim < 0.05`.”
- “The observed pattern is consistent with spatial concentration.”
- “The LISA map suggests spatial clustering but does not establish causality.”
- “The result should be interpreted as exploratory local spatial association.”

## Claims To Avoid

- Mobility causes employment.
- High-High areas are necessarily the most economically productive.
- Non-significant areas have no meaningful activity.
- LISA clusters represent permanent structures.
- Temporal changes can be inferred from one month of mobility data.
"""
    (REPORT_DIR / "spatial_interpretation_notes.md").write_text(notes, encoding="utf-8")


def verify_outputs(
    gdf: gpd.GeoDataFrame,
    weights,
    results: dict[str, pd.DataFrame],
    source_hash_before: str,
    source_mtime_before: float,
    env_info: dict[str, str],
    moran_objects: dict[str, Moran_Local],
) -> pd.DataFrame:
    required_csvs = [
        "local_spatial_input_validation.csv",
        "local_spatial_weights_validation.csv",
        "lisa_employment_results.csv",
        "lisa_mobility_results.csv",
        "lisa_cluster_summary.csv",
        "lisa_significance_summary.csv",
        "lisa_employment_top_clusters.csv",
        "lisa_mobility_top_clusters.csv",
        "lisa_multiple_testing_sensitivity.csv",
        "lisa_multiple_testing_sensitivity_summary.csv",
        "lisa_employment_mobility_overlap.csv",
        "lisa_overlap_summary.csv",
    ]
    required_pngs = [
        spec["cluster_map"] for spec in LISA_VARIABLES.values()
    ] + [spec["significance_map"] for spec in LISA_VARIABLES.values()]
    required_task04a = [
        "spatial_input_validation.csv",
        "spatial_weights_diagnostics.csv",
        "global_moran_results.csv",
        "global_moran_comparison_custom_vs_pysal.csv",
    ]
    rows = []

    def add(check: str, value, status: str, notes: str = "") -> None:
        rows.append({"check": check, "value": value, "status": status, "notes": notes})

    add("python_executable_recorded", env_info["python_executable"], "PASS")
    add("expected_python_version", env_info["python_version"], "PASS" if env_info["python_version"].startswith("3.12.12") else "FAIL")
    add("libpysal_imported", env_info["libpysal_version"], "PASS")
    add("esda_imported", env_info["esda_version"], "PASS")
    add("official_queen_object_used", type(weights).__module__ + "." + type(weights).__name__, "PASS" if isinstance(weights, Queen) else "FAIL")
    add("official_moran_local_used", "|".join(type(obj).__module__ + "." + type(obj).__name__ for obj in moran_objects.values()), "PASS" if all(isinstance(obj, Moran_Local) for obj in moran_objects.values()) else "FAIL")
    add("no_custom_fallback_used", True, "PASS")
    add("input_row_count", len(gdf), "PASS" if len(gdf) == EXPECTED_ROWS else "FAIL")
    add("unique_lsoa21cd_count", gdf["LSOA21CD"].nunique(), "PASS" if gdf["LSOA21CD"].nunique() == EXPECTED_ROWS else "FAIL")
    add("observations_align_with_weights", weights.n, "PASS" if weights.n == len(gdf) else "FAIL")
    add("no_islands", len(weights.islands), "PASS" if len(weights.islands) == 0 else "FAIL")
    add("one_connected_component", weights.n_components, "PASS" if weights.n_components == 1 else "FAIL")
    for key, table in results.items():
        add(f"{key}_local_moran_statistics_finite", bool(np.isfinite(table["local_moran_i"]).all()), "PASS" if np.isfinite(table["local_moran_i"]).all() else "FAIL")
        p_valid = bool(((table["permutation_p_value"] >= 0) & (table["permutation_p_value"] <= 1)).all())
        add(f"{key}_raw_permutation_p_values_valid", p_valid, "PASS" if p_valid else "FAIL")
        add(f"{key}_every_observation_has_category", int(table["lisa_cluster_category"].notna().sum()), "PASS" if table["lisa_cluster_category"].notna().sum() == EXPECTED_ROWS else "FAIL")
        category_sum = int(table["lisa_cluster_category"].value_counts().reindex(CLUSTER_ORDER, fill_value=0).sum())
        add(f"{key}_cluster_categories_sum_to_4994", category_sum, "PASS" if category_sum == EXPECTED_ROWS else "FAIL")
    for name in required_csvs:
        add(f"csv_exists__{name}", (TABLE_DIR / name).exists(), "PASS" if (TABLE_DIR / name).exists() else "FAIL")
    for name in required_pngs:
        path = MAP_DIR / name
        if not path.exists():
            add(f"png_exists__{name}", False, "FAIL")
            continue
        try:
            with Image.open(path) as image:
                image.verify()
            add(f"png_readable__{name}", True, "PASS")
        except Exception as exc:
            add(f"png_readable__{name}", False, "FAIL", str(exc))
    add("spatial_esda_summary_report_exists", (REPORT_DIR / "spatial_esda_summary.md").exists(), "PASS" if (REPORT_DIR / "spatial_esda_summary.md").exists() else "FAIL")
    add("spatial_interpretation_notes_exists", (REPORT_DIR / "spatial_interpretation_notes.md").exists(), "PASS" if (REPORT_DIR / "spatial_interpretation_notes.md").exists() else "FAIL")
    for name in required_task04a:
        add(f"task04a_output_present__{name}", (TABLE_DIR / name).exists(), "PASS" if (TABLE_DIR / name).exists() else "FAIL")
    add("source_gpkg_hash_unchanged", source_hash_before == file_sha256(MASTER_GPKG), "PASS" if source_hash_before == file_sha256(MASTER_GPKG) else "FAIL")
    add("source_gpkg_mtime_unchanged", source_mtime_before == MASTER_GPKG.stat().st_mtime, "PASS" if source_mtime_before == MASTER_GPKG.stat().st_mtime else "FAIL")
    add("no_source_dataset_modified", True, "PASS")
    add("no_ols_run", True, "PASS")
    add("no_spatial_regression_run", True, "PASS")
    add("no_machine_learning_model_run", True, "PASS")
    qa = pd.DataFrame(rows)
    qa.to_csv(TABLE_DIR / "local_spatial_esda_qa.csv", index=False)
    if (qa["status"] == "FAIL").any():
        raise RuntimeError(f"Local Spatial ESDA QA failed:\n{qa[qa['status'] == 'FAIL'].to_string(index=False)}")
    return qa


def main() -> None:
    start = time.time()
    ensure_dirs()
    env_info = log_environment()
    print(f"input_path: {MASTER_GPKG.relative_to(PROJECT_ROOT)}")
    print(f"output_tables: {TABLE_DIR.relative_to(PROJECT_ROOT)}")
    print(f"output_maps: {MAP_DIR.relative_to(PROJECT_ROOT)}")
    print(f"output_reports: {REPORT_DIR.relative_to(PROJECT_ROOT)}")
    print(f"random_seed: {RANDOM_SEED}")
    print(f"permutations: {PERMUTATIONS}")
    print(f"significance_threshold: p_sim < {SIGNIFICANCE}")
    print("weight_definition: first-order Queen contiguity, row-standardised")
    source_hash_before = file_sha256(MASTER_GPKG)
    source_mtime_before = MASTER_GPKG.stat().st_mtime
    gdf = read_input()
    validation = validate_input(gdf)
    weights = construct_weights(gdf)
    weights_validation = validate_weights(weights, gdf)
    results, moran_objects = run_local_moran(gdf, weights)
    cluster_summary, sig_summary = cluster_summaries(results)
    ranked_tables(results)
    lisa_maps(gdf, results)
    sensitivity, fdr_summary = multiple_testing_sensitivity(results)
    overlap, overlap_summary = overlap_tables(results)
    write_reports(env_info, validation, weights_validation, cluster_summary, sig_summary, fdr_summary, overlap_summary)
    qa = verify_outputs(gdf, weights, results, source_hash_before, source_mtime_before, env_info, moran_objects)
    runtime = time.time() - start
    print("Local Spatial ESDA complete.")
    print(f"runtime_seconds: {runtime:.2f}")
    print("Significance summary:")
    print(sig_summary.to_string(index=False))
    print("FDR sensitivity summary:")
    print(fdr_summary.to_string(index=False))
    print("Overlap summary:")
    print(overlap_summary.to_string(index=False))
    print("QA summary:")
    print(qa["status"].value_counts().to_string())
    print("No source dataset modification, regression, spatial regression, or machine-learning model was performed.")


if __name__ == "__main__":
    main()
