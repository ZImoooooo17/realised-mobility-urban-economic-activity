"""Run ESDA outputs for the frozen Master Analysis Dataset.

This script is read-only with respect to the master dataset. It writes
tables, figures, and a summary report under outputs/esda/.

It intentionally does not transform variables, remove outliers, fit
regression models, compute Moran's I, or begin OLS.
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs/esda"
TABLE_DIR = OUTPUT_DIR / "tables"
FIGURE_DIR = OUTPUT_DIR / "figures"
MPLCONFIG_DIR = OUTPUT_DIR / ".mplconfig"

MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor


MASTER_CSV = PROJECT_ROOT / "data/processed/master/master_analysis_dataset_final_v1_0.csv"

CORE_VARIABLES = {
    "employees_count": "Employees count",
    "mobility_inflow_total": "Mobility inflow total",
    "flow_entropy": "Flow entropy",
    "weekday_weekend_ratio": "Weekday-weekend ratio",
    "ptal_ai_mean": "PTAL mean accessibility index",
    "population_density_km2": "Population density per km2",
    "amenity_poi_density_km2": "Amenity POI density per km2",
    "retail_poi_density_km2": "Retail POI density per km2",
}

CORRELATION_VARIABLES = [
    "employees_count",
    "mobility_inflow_total",
    "mobility_inflow_mean_daily",
    "flow_entropy",
    "weekday_weekend_ratio",
    "ptal_ai_mean",
    "population_density_km2",
    "amenity_poi_density_km2",
    "retail_poi_density_km2",
    "unique_origins",
    "mobility_outflow_total",
    "net_flow",
]

VIF_VARIABLES = [
    "mobility_inflow_mean_daily",
    "flow_entropy",
    "weekday_weekend_ratio",
    "ptal_ai_mean",
    "population_density_km2",
    "amenity_poi_density_km2",
    "retail_poi_density_km2",
    "unique_origins",
    "net_flow",
]

SKEW_THRESHOLD = 2.0
KURTOSIS_THRESHOLD = 10.0
OUTLIER_IQR_MULTIPLIER = 1.5
HIGH_CORRELATION_THRESHOLD = 0.8
VIF_WARNING_THRESHOLD = 5.0
VIF_HIGH_THRESHOLD = 10.0


def ensure_outputs() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def read_master() -> pd.DataFrame:
    if not MASTER_CSV.exists():
        raise FileNotFoundError(MASTER_CSV)
    return pd.read_csv(MASTER_CSV)


def validate_required_columns(df: pd.DataFrame) -> None:
    required = set(CORE_VARIABLES) | set(CORRELATION_VARIABLES) | set(VIF_VARIABLES)
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(f"Master dataset missing required ESDA columns: {missing}")


def descriptive_statistics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
    for col in numeric_cols:
        values = pd.to_numeric(df[col], errors="coerce")
        finite = values[np.isfinite(values.to_numpy(dtype=float, na_value=np.nan))]
        rows.append(
            {
                "variable": col,
                "count": int(finite.count()),
                "mean": float(finite.mean()) if len(finite) else np.nan,
                "median": float(finite.median()) if len(finite) else np.nan,
                "standard_deviation": float(finite.std()) if len(finite) else np.nan,
                "min": float(finite.min()) if len(finite) else np.nan,
                "max": float(finite.max()) if len(finite) else np.nan,
                "skewness": float(finite.skew()) if len(finite) else np.nan,
                "kurtosis": float(finite.kurtosis()) if len(finite) else np.nan,
                "zero_count": int((values == 0).sum()),
                "missing_count": int(values.isna().sum()),
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(TABLE_DIR / "descriptive_statistics.csv", index=False)
    return result


def safe_filename(name: str) -> str:
    return name.lower().replace(" ", "_").replace("/", "_")


def distribution_plots(df: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid")
    for col, label in CORE_VARIABLES.items():
        values = pd.to_numeric(df[col], errors="coerce").dropna()

        plt.figure(figsize=(8, 5))
        sns.histplot(values, bins=40, kde=False, color="#34699a", edgecolor="white")
        plt.title(f"Histogram: {label}")
        plt.xlabel(col)
        plt.ylabel("LSOA count")
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / f"{safe_filename(col)}_histogram.png", dpi=200)
        plt.close()

        plt.figure(figsize=(8, 3.8))
        sns.boxplot(x=values, color="#c97b63", fliersize=2)
        plt.title(f"Boxplot: {label}")
        plt.xlabel(col)
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / f"{safe_filename(col)}_boxplot.png", dpi=200)
        plt.close()


def correlations(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    corr_df = df[CORRELATION_VARIABLES].apply(pd.to_numeric, errors="coerce")
    pearson = corr_df.corr(method="pearson")
    spearman = corr_df.corr(method="spearman")

    long_rows = []
    for method, matrix in [("pearson", pearson), ("spearman", spearman)]:
        for row_var in matrix.index:
            for col_var in matrix.columns:
                long_rows.append(
                    {
                        "method": method,
                        "row_variable": row_var,
                        "column_variable": col_var,
                        "correlation": float(matrix.loc[row_var, col_var]),
                    }
                )
    corr_long = pd.DataFrame(long_rows)
    corr_long.to_csv(TABLE_DIR / "correlation_matrix.csv", index=False)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        pearson,
        cmap="vlag",
        vmin=-1,
        vmax=1,
        center=0,
        annot=True,
        fmt=".2f",
        square=True,
        linewidths=0.4,
        cbar_kws={"label": "Pearson correlation"},
    )
    plt.title("Pearson Correlation Heatmap")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "correlation_heatmap.png", dpi=220)
    plt.close()

    return pearson, spearman


def calculate_vif(df: pd.DataFrame) -> pd.DataFrame:
    vif_df = df[VIF_VARIABLES].apply(pd.to_numeric, errors="coerce").dropna().copy()
    constant_cols = [col for col in vif_df.columns if vif_df[col].nunique(dropna=True) <= 1]
    vif_df = vif_df.drop(columns=constant_cols)
    z = (vif_df - vif_df.mean()) / vif_df.std(ddof=0)
    z = z.replace([np.inf, -np.inf], np.nan).dropna()

    rows = []
    for idx, col in enumerate(z.columns):
        try:
            vif_value = float(variance_inflation_factor(z.to_numpy(dtype=float), idx))
        except Exception as exc:
            rows.append(
                {
                    "variable": col,
                    "vif": np.nan,
                    "status": "calculation_failed",
                    "notes": str(exc),
                }
            )
            continue

        if vif_value >= VIF_HIGH_THRESHOLD:
            status = "high"
        elif vif_value >= VIF_WARNING_THRESHOLD:
            status = "moderate"
        else:
            status = "low"
        rows.append({"variable": col, "vif": vif_value, "status": status, "notes": ""})

    for col in constant_cols:
        rows.append(
            {
                "variable": col,
                "vif": np.nan,
                "status": "excluded_constant",
                "notes": "Variable is constant and cannot be used in VIF calculation.",
            }
        )

    result = pd.DataFrame(rows).sort_values(["status", "vif"], ascending=[True, False])
    result.to_csv(TABLE_DIR / "vif.csv", index=False)
    return result


def iqr_outlier_count(values: pd.Series) -> int:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    q1 = clean.quantile(0.25)
    q3 = clean.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - OUTLIER_IQR_MULTIPLIER * iqr
    upper = q3 + OUTLIER_IQR_MULTIPLIER * iqr
    return int(((clean < lower) | (clean > upper)).sum())


def high_correlations(pearson: pd.DataFrame, spearman: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, matrix in [("pearson", pearson), ("spearman", spearman)]:
        cols = list(matrix.columns)
        for i, row_var in enumerate(cols):
            for col_var in cols[i + 1 :]:
                value = float(matrix.loc[row_var, col_var])
                if abs(value) >= HIGH_CORRELATION_THRESHOLD:
                    rows.append(
                        {
                            "method": method,
                            "variable_1": row_var,
                            "variable_2": col_var,
                            "correlation": value,
                        }
                    )
    return pd.DataFrame(rows).sort_values("correlation", key=lambda s: s.abs(), ascending=False) if rows else pd.DataFrame(columns=["method", "variable_1", "variable_2", "correlation"])


def write_report(
    df: pd.DataFrame,
    desc: pd.DataFrame,
    pearson: pd.DataFrame,
    spearman: pd.DataFrame,
    vif: pd.DataFrame,
) -> None:
    skewed = desc[
        desc["skewness"].abs().ge(SKEW_THRESHOLD) | desc["kurtosis"].ge(KURTOSIS_THRESHOLD)
    ].sort_values("skewness", key=lambda s: s.abs(), ascending=False)
    outlier_rows = []
    for col in CORE_VARIABLES:
        outlier_rows.append({"variable": col, "iqr_outlier_count": iqr_outlier_count(df[col])})
    outliers = pd.DataFrame(outlier_rows).sort_values("iqr_outlier_count", ascending=False)
    high_corr = high_correlations(pearson, spearman)
    vif_flagged = vif[vif["status"].isin(["moderate", "high", "calculation_failed"])]

    transform_recs = []
    for _, row in skewed.iterrows():
        var = row["variable"]
        if var in {"mobility_processed_days", "mobility_expected_days", "mobility_coverage_ratio", "mobility_weekday_count", "mobility_weekend_count"}:
            continue
        if row["skewness"] > SKEW_THRESHOLD and row["min"] >= 0:
            transform_recs.append((var, "Consider log1p or square-root sensitivity during modelling; do not apply before ESDA review."))
        elif abs(row["skewness"]) > SKEW_THRESHOLD:
            transform_recs.append((var, "Consider robust scaling or alternative modelling strategy because values include negatives or strong asymmetry."))
    transform_table = pd.DataFrame(transform_recs, columns=["variable", "recommendation"]).drop_duplicates()

    lines = []
    lines.append("# ESDA Summary")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append("- Input: `data/processed/master/master_analysis_dataset_final_v1_0.csv`")
    lines.append("- Outputs: `outputs/esda/tables/` and `outputs/esda/figures/`")
    lines.append("- Constraints honoured: no dataset modification, no transformations applied, no outlier removal, no regression models, no Moran's I, and no OLS.")
    lines.append("")
    lines.append("## Dataset")
    lines.append("")
    lines.append(f"- Rows: {len(df):,}")
    lines.append(f"- Columns: {df.shape[1]:,}")
    lines.append(f"- Numeric variables summarised: {len(desc):,}")
    lines.append("")
    lines.append("## Skewed Variables")
    lines.append("")
    if skewed.empty:
        lines.append("No variables exceeded the skewness/kurtosis screening thresholds.")
    else:
        lines.append(f"Screening thresholds: absolute skewness >= {SKEW_THRESHOLD} or kurtosis >= {KURTOSIS_THRESHOLD}.")
        lines.append("")
        lines.append(skewed[["variable", "skewness", "kurtosis", "min", "max"]].head(20).to_markdown(index=False))
    lines.append("")
    lines.append("## Potential Outliers")
    lines.append("")
    lines.append(f"Outliers are counted using the {OUTLIER_IQR_MULTIPLIER}x IQR rule for screening only. No observations were removed.")
    lines.append("")
    lines.append(outliers.to_markdown(index=False))
    lines.append("")
    lines.append("## Possible Transformations")
    lines.append("")
    if transform_table.empty:
        lines.append("No transformation recommendations were generated by the screening thresholds.")
    else:
        lines.append("These are recommendations for later modelling sensitivity only; the ESDA pipeline does not apply them.")
        lines.append("")
        lines.append(transform_table.head(25).to_markdown(index=False))
    lines.append("")
    lines.append("## Highly Correlated Variables")
    lines.append("")
    if high_corr.empty:
        lines.append(f"No Pearson or Spearman correlations exceeded |r| >= {HIGH_CORRELATION_THRESHOLD}.")
    else:
        lines.append(f"Pairs with |correlation| >= {HIGH_CORRELATION_THRESHOLD}:")
        lines.append("")
        lines.append(high_corr.head(30).to_markdown(index=False))
    lines.append("")
    lines.append("## VIF")
    lines.append("")
    lines.append(f"VIF status thresholds: moderate >= {VIF_WARNING_THRESHOLD}; high >= {VIF_HIGH_THRESHOLD}.")
    lines.append("")
    lines.append(vif.to_markdown(index=False))
    lines.append("")
    if vif_flagged.empty:
        lines.append("No VIF values reached the moderate/high threshold.")
    else:
        lines.append("Variables with moderate/high VIF should be reviewed before modelling; this is not a model fit.")
    lines.append("")
    lines.append("## Modelling Considerations")
    lines.append("")
    lines.append("- Choose either `mobility_inflow_total` or `mobility_inflow_mean_daily` in a baseline specification, not both.")
    lines.append("- Treat `net_flow` as a robustness variable because the OD-symmetry limitation remains.")
    lines.append("- Inspect skewed employment, mobility, POI-density, and origin-count variables during ESDA before deciding transformations.")
    lines.append("- Review high correlations and VIF before constructing baseline and robustness specifications.")
    lines.append("- Retain temporal alignment caveats: Census 2021, PTAL 2023, BRES 2024, OSM 2026 extract, Locomizer March 2026.")
    lines.append("")
    lines.append("## Generated Files")
    lines.append("")
    lines.append("- `outputs/esda/tables/descriptive_statistics.csv`")
    lines.append("- `outputs/esda/tables/correlation_matrix.csv`")
    lines.append("- `outputs/esda/tables/vif.csv`")
    lines.append("- `outputs/esda/figures/*_histogram.png`")
    lines.append("- `outputs/esda/figures/*_boxplot.png`")
    lines.append("- `outputs/esda/figures/correlation_heatmap.png`")
    lines.append("")

    (OUTPUT_DIR / "esda_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_outputs()
    df = read_master()
    validate_required_columns(df)
    desc = descriptive_statistics(df)
    distribution_plots(df)
    pearson, spearman = correlations(df)
    vif = calculate_vif(df)
    write_report(df, desc, pearson, spearman, vif)
    print(f"Wrote {TABLE_DIR / 'descriptive_statistics.csv'}")
    print(f"Wrote {TABLE_DIR / 'correlation_matrix.csv'}")
    print(f"Wrote {TABLE_DIR / 'vif.csv'}")
    print(f"Wrote figures to {FIGURE_DIR}")
    print(f"Wrote {OUTPUT_DIR / 'esda_summary.md'}")


if __name__ == "__main__":
    main()
