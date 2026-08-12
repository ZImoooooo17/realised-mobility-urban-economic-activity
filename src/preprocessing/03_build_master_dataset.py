"""Build final dissertation Master Analysis Dataset v1.0.

This script joins validated processed LSOA-level inputs onto the locked
Greater London LSOA 2021 geography using LSOA21CD as the sole join key.
It does not run ESDA, modelling, or create model-specific transformations.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

GEOGRAPHY_PATH = PROJECT_ROOT / "data/processed/boundaries/lsoa21_london_bgc.gpkg"
BRES_PATH = PROJECT_ROOT / "data/processed/bres/bres_lsoa_variables.csv"
PTAL_PATH = PROJECT_ROOT / "data/processed/ptal/ptal_lsoa_variables.csv"
CENSUS_PATH = PROJECT_ROOT / "data/processed/census/census_2021_lsoa_population.csv"
POI_PATH = PROJECT_ROOT / "data/processed/poi/poi_lsoa_variables.csv"
MOBILITY_PATH = PROJECT_ROOT / "data/processed/mobility/march_2026_lsoa_mobility_final_31days.csv"

PROCESSED_DIR = PROJECT_ROOT / "data/processed/master"
OUTPUT_DIR = PROJECT_ROOT / "outputs/master_dataset"

CSV_OUTPUT = PROCESSED_DIR / "master_analysis_dataset_final_v1_0.csv"
GPKG_OUTPUT = PROCESSED_DIR / "master_analysis_dataset_final_v1_0.gpkg"
REPORT_OUTPUT = OUTPUT_DIR / "master_dataset_final_v1_0_report.md"
QA_OUTPUT = OUTPUT_DIR / "master_dataset_final_v1_0_qa.csv"
JOIN_COVERAGE_OUTPUT = OUTPUT_DIR / "master_dataset_final_v1_0_join_coverage.csv"
DICTIONARY_OUTPUT = OUTPUT_DIR / "master_dataset_final_v1_0_dictionary.csv"
MANIFEST_OUTPUT = OUTPUT_DIR / "master_dataset_final_v1_0_manifest.md"
PROVISIONAL_COMPARISON_CSV = OUTPUT_DIR / "provisional_vs_final_master_comparison.csv"
PROVISIONAL_COMPARISON_MD = OUTPUT_DIR / "provisional_vs_final_master_comparison.md"

EXPECTED_LSOAS = 4994
LSOA_PATTERN = r"^E010\d{5}$"
TOLERANCE = 1e-6


SOURCES = {
    "locked_geography": GEOGRAPHY_PATH,
    "bres_2024": BRES_PATH,
    "ptal_2023": PTAL_PATH,
    "census_2021_population": CENSUS_PATH,
    "osm_poi": POI_PATH,
    "mobility_march_2026_final_31day": MOBILITY_PATH,
}


def rel(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT))


def standardise_key(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    if "LSOA21CD" not in result.columns:
        raise RuntimeError("Input missing required LSOA21CD column")
    result["LSOA21CD"] = result["LSOA21CD"].astype("string").str.strip()
    return result


def read_csv_source(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return standardise_key(pd.read_csv(path))


def read_geography() -> gpd.GeoDataFrame:
    if not GEOGRAPHY_PATH.exists():
        raise FileNotFoundError(GEOGRAPHY_PATH)
    gdf = gpd.read_file(GEOGRAPHY_PATH).to_crs("EPSG:27700")
    gdf = standardise_key(gdf)
    return gdf


def check_required_columns(name: str, df: pd.DataFrame, required: list[str]) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise RuntimeError(f"{name} missing required columns: {missing}")


def source_status(path: Path, default: str) -> str:
    text = str(path).lower()
    if "provisional" in text or "interim" in text:
        return "provisional" if "provisional" in text else "interim"
    return default


def join_coverage_rows(geography: pd.DataFrame, sources: dict[str, pd.DataFrame]) -> pd.DataFrame:
    london_codes = set(geography["LSOA21CD"])
    rows = []
    for source_name, df in sources.items():
        codes = set(df["LSOA21CD"].dropna())
        rows.append(
            {
                "source_name": source_name,
                "source_file": rel(SOURCES[source_name]),
                "source_row_count": len(df),
                "unique_lsoa21cd_count": df["LSOA21CD"].nunique(dropna=True),
                "matched_london_lsoas": len(london_codes & codes),
                "unmatched_london_lsoas": len(london_codes - codes),
                "extra_non_london_lsoas": len(codes - london_codes),
                "duplicate_lsoa21cd_count": int(df["LSOA21CD"].duplicated().sum()),
                "missing_lsoa21cd_count": int(df["LSOA21CD"].isna().sum()),
                "valid_lsoa21cd_format_count": int(df["LSOA21CD"].str.match(LSOA_PATTERN, na=False).sum()),
            }
        )
    return pd.DataFrame(rows)


def add_qa(rows: list[dict], metric: str, value, status: str, notes: str = "") -> None:
    rows.append({"metric": metric, "value": value, "status": status, "notes": notes})


def analytical_variable_columns(df: pd.DataFrame) -> list[str]:
    exclude = {"LSOA21CD", "LSOA21NM", "geometry", "mobility_missing_dates"}
    return [col for col in df.columns if col not in exclude]


def numeric_summary_qa(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows: list[dict] = []
    for col in columns:
        series = df[col]
        if not pd.api.types.is_numeric_dtype(series):
            add_qa(rows, f"{col}__dtype", str(series.dtype), "pass")
            add_qa(rows, f"{col}__non_null_count", int(series.notna().sum()), "pass")
            add_qa(rows, f"{col}__missing_count", int(series.isna().sum()), "pass" if series.isna().sum() == 0 else "fail")
            continue

        values = pd.to_numeric(series, errors="coerce")
        finite = np.isfinite(values.to_numpy(dtype=float, na_value=np.nan))
        inf_count = int((~finite & ~values.isna().to_numpy()).sum())
        neg_count = int((values < 0).sum())
        status = "warn" if col == "net_flow" and neg_count > 0 else ("pass" if neg_count == 0 else "fail")
        add_qa(rows, f"{col}__dtype", str(series.dtype), "pass")
        add_qa(rows, f"{col}__non_null_count", int(series.notna().sum()), "pass")
        add_qa(rows, f"{col}__missing_count", int(series.isna().sum()), "pass" if series.isna().sum() == 0 else "fail")
        add_qa(rows, f"{col}__infinite_count", inf_count, "pass" if inf_count == 0 else "fail")
        add_qa(rows, f"{col}__negative_count", neg_count, status)
        add_qa(rows, f"{col}__zero_count", int((values == 0).sum()), "pass")
        add_qa(rows, f"{col}__minimum", float(values.min()), "pass")
        add_qa(rows, f"{col}__maximum", float(values.max()), "pass")
        add_qa(rows, f"{col}__mean", float(values.mean()), "pass")
        add_qa(rows, f"{col}__median", float(values.median()), "pass")
        add_qa(rows, f"{col}__standard_deviation", float(values.std()), "pass")
    return pd.DataFrame(rows)


def build_dictionary() -> pd.DataFrame:
    rows = [
        ("LSOA21CD", "identifier", "Locked Greater London LSOA 2021 geography", "2021", "LSOA21CD", "Greater London LSOA 2021 code.", "identifier", "none", "sole join key", "none identified"),
        ("LSOA21NM", "metadata", "Locked Greater London LSOA 2021 geography", "2021", "LSOA21NM", "Greater London LSOA 2021 name.", "text", "none", "geography label", "not used as join key"),
        ("employees_count", "DV", "BRES LSOA variables", "2024", "employees_count", "Workplace employee count by LSOA. Source BRES extract is Employees, Count.", "employees", "source-provided processed variable", "dependent variable", "BRES values are rounded; working proprietors are excluded; known ONS 2015-2024 geographic misreporting caveat retained."),
        ("employee_density_km2", "DV sensitivity", "BRES LSOA variables", "2024", "employee_density_km2", "Workplace employee count per square kilometre.", "employees per km2", "source-provided: employees_count / LSOA area km2", "descriptive/sensitivity outcome", "mechanically related to employees_count and area."),
        ("log1p_employees_count", "DV sensitivity", "BRES LSOA variables", "2024", "log1p_employees_count", "Log-one-plus transformation of workplace employee count.", "log employees", "source-provided transformation", "sensitivity only; no new transformation created here", "do not mix casually with raw count in baseline."),
        ("log1p_employee_density_km2", "DV sensitivity", "BRES LSOA variables", "2024", "log1p_employee_density_km2", "Log-one-plus transformation of workplace employee density.", "log employees per km2", "source-provided transformation", "sensitivity only; no new transformation created here", "do not mix casually with raw density in baseline."),
        ("sqrt_employee_density_km2", "DV sensitivity", "BRES LSOA variables", "2024", "sqrt_employee_density_km2", "Square-root transformation of workplace employee density.", "sqrt employees per km2", "source-provided transformation", "sensitivity only; no new transformation created here", "do not mix casually with raw density in baseline."),
        ("ptal_ai_mean", "accessibility IV", "PTAL LSOA variables", "2023", "mean_AI", "Official TfL mean public transport accessibility index for each LSOA.", "accessibility index", "source-provided processed variable renamed in PTAL pipeline", "main accessibility variable", "temporal mismatch with 2024 BRES and 2026 mobility."),
        ("ptal_ai_median", "metadata", "PTAL LSOA variables", "2023", "ptal_ai_median", "Median PTAL accessibility index.", "accessibility index", "source-provided processed variable", "descriptive/sensitivity", "not baseline accessibility variable."),
        ("ptal_ai_min", "metadata", "PTAL LSOA variables", "2023", "ptal_ai_min", "Minimum PTAL accessibility index.", "accessibility index", "source-provided processed variable", "descriptive/sensitivity", "not baseline accessibility variable."),
        ("ptal_ai_max", "metadata", "PTAL LSOA variables", "2023", "ptal_ai_max", "Maximum PTAL accessibility index.", "accessibility index", "source-provided processed variable", "descriptive/sensitivity", "not baseline accessibility variable."),
        ("ptal_mean_category", "metadata", "PTAL LSOA variables", "2023", "ptal_mean_category", "Categorical PTAL band corresponding to mean accessibility.", "category", "source-provided processed variable", "descriptive only", "categorical PTAL is not the main model variable."),
        ("usual_resident_population_2021", "control", "Census 2021 TS001 population", "2021", "2021", "All usual residents in households and communal establishments.", "persons", "renamed from TS001 population column", "population count retained for QA/descriptive analysis", "Nomis disclosure control may swap/perturb small-area counts by small amounts."),
        ("lsoa_area_km2", "control", "Locked Greater London LSOA 2021 geometry", "2021", "geometry", "Area of locked LSOA polygon.", "km2", "geometry area in EPSG:27700 / 1,000,000", "density denominator and spatial control metadata", "area follows locked boundary geometry."),
        ("population_density_km2", "control", "Census 2021 TS001 population", "2021", "usual_resident_population_2021 / lsoa_area_km2", "Usual resident population per square kilometre.", "persons per km2", "source-provided Census pipeline calculation", "main population control", "2021 residential population, temporally earlier than BRES and mobility."),
        ("amenity_poi_count", "control", "OSM POI LSOA variables", "2026-07-06 extract", "amenity_poi_count", "Count of retained amenity POIs assigned to LSOA under frozen POI rules.", "POIs", "source-provided processed variable", "QA/descriptive retained count", "depends on OSM completeness and frozen category rules."),
        ("amenity_poi_density_km2", "control", "OSM POI LSOA variables", "2026-07-06 extract", "amenity_density", "Retained amenity POIs per square kilometre.", "POIs per km2", "renamed from source column amenity_density", "built-environment control", "depends on OSM completeness and frozen category rules."),
        ("retail_poi_count", "control", "OSM POI LSOA variables", "2026-07-06 extract", "retail_poi_count", "Count of retained retail POIs assigned to LSOA under frozen POI rules.", "POIs", "source-provided processed variable", "QA/descriptive retained count", "depends on OSM completeness and frozen category rules."),
        ("retail_poi_density_km2", "control", "OSM POI LSOA variables", "2026-07-06 extract", "retail_poi_density", "Retained retail POIs per square kilometre.", "POIs per km2", "renamed from source column retail_poi_density", "built-environment control", "depends on OSM completeness and frozen category rules."),
        ("mobility_inflow_total", "core IV", "Locomizer final March 2026 mobility", "March 2026", "mobility_inflow_total", "All-origin inflow to London destination LSOA across 31 March days.", "extrapolated users", "source-provided processed variable", "mobility intensity option", "do not use with mobility_inflow_mean_daily in same baseline because mechanically related."),
        ("mobility_inflow_mean_daily", "core IV", "Locomizer final March 2026 mobility", "March 2026", "mobility_inflow_mean_daily", "Mean daily all-origin inflow to London destination LSOA.", "extrapolated users per day", "source-provided: total / 31", "mobility intensity option", "do not use with mobility_inflow_total in same baseline because mechanically related."),
        ("flow_entropy", "core IV", "Locomizer final March 2026 mobility", "March 2026", "flow_entropy", "Shannon entropy of valid raw H3 origins for each destination LSOA.", "entropy index", "source-provided processed variable", "flow diversity variable", "invalid/non-H3 origins are excluded from entropy denominator."),
        ("weekday_inflow_total", "core IV", "Locomizer final March 2026 mobility", "March 2026", "weekday_inflow_total", "All-origin inflow across 22 weekdays.", "extrapolated users", "source-provided processed variable", "temporal mobility descriptor", "mechanically related to weekday_inflow_mean_daily."),
        ("weekend_inflow_total", "core IV", "Locomizer final March 2026 mobility", "March 2026", "weekend_inflow_total", "All-origin inflow across 9 weekend days.", "extrapolated users", "source-provided processed variable", "temporal mobility descriptor", "mechanically related to weekend_inflow_mean_daily."),
        ("weekday_inflow_mean_daily", "core IV", "Locomizer final March 2026 mobility", "March 2026", "weekday_mean_daily_inflow", "Mean daily weekday inflow.", "extrapolated users per weekday", "renamed from source column weekday_mean_daily_inflow", "weekday mobility intensity", "derived from weekday total and weekday count."),
        ("weekend_inflow_mean_daily", "core IV", "Locomizer final March 2026 mobility", "March 2026", "weekend_mean_daily_inflow", "Mean daily weekend inflow.", "extrapolated users per weekend day", "renamed from source column weekend_mean_daily_inflow", "weekend mobility intensity", "derived from weekend total and weekend count."),
        ("weekday_weekend_ratio", "core IV", "Locomizer final March 2026 mobility", "March 2026", "weekday_weekend_ratio", "Weekday mean daily inflow divided by weekend mean daily inflow.", "ratio", "source-provided processed variable", "temporal mobility behaviour", "finite because no zero weekend denominators in final output."),
        ("unique_origins", "robustness variable", "Locomizer final March 2026 mobility", "March 2026", "unique_origins", "Count of distinct valid raw H3 origins with positive flow to the destination LSOA.", "H3 origins", "source-provided processed variable", "robustness / descriptive origin reach", "excludes invalid/non-H3 origins such as other."),
        ("mobility_outflow_total", "robustness variable", "Locomizer final March 2026 mobility", "March 2026", "mobility_outflow_total", "Flow from London origin LSOA to all valid destinations represented in the destination-format data.", "extrapolated users", "source-provided processed variable", "robustness only", "OD symmetry cannot be independently verified from destination-format files alone."),
        ("net_flow", "robustness variable", "Locomizer final March 2026 mobility", "March 2026", "net_flow", "All-origin inflow minus London-origin outflow.", "extrapolated users", "source-provided processed variable", "robustness only", "negative values expected for many LSOAs; OD symmetry limitation retained."),
        ("london_origin_flow", "metadata", "Locomizer final March 2026 mobility", "March 2026", "london_origin_flow", "Destination inflow from origins assigned to London LSOAs.", "extrapolated users", "source-provided processed variable", "QA/provenance", "not a primary inflow measure."),
        ("valid_non_london_origin_flow", "metadata", "Locomizer final March 2026 mobility", "March 2026", "valid_non_london_origin_flow", "Destination inflow from valid H3 origins not assigned to London LSOAs.", "extrapolated users", "source-provided processed variable", "QA/provenance", "not a primary inflow measure."),
        ("other_origin_flow", "metadata", "Locomizer final March 2026 mobility", "March 2026", "other_origin_flow", "Destination inflow from invalid/non-H3 origins.", "extrapolated users", "source-provided processed variable", "QA/provenance", "included in total inflow but excluded from entropy."),
        ("other_origin_share", "metadata", "Locomizer final March 2026 mobility", "March 2026", "other_origin_share", "Share of total destination inflow from invalid/non-H3 origins.", "share", "source-provided processed variable", "QA/provenance", "documents entropy exclusion share."),
        ("valid_origin_flow_share", "metadata", "Locomizer final March 2026 mobility", "March 2026", "valid_origin_flow_share", "Share of total destination inflow included in entropy denominator.", "share", "source-provided processed variable", "QA/provenance", "entropy denominator share."),
        ("external_origin_share", "metadata", "Locomizer final March 2026 mobility", "March 2026", "external_origin_share", "Share of total destination inflow from valid non-London H3 origins.", "share", "source-provided processed variable", "QA/provenance", "external-origin share."),
        ("entropy_denominator", "metadata", "Locomizer final March 2026 mobility", "March 2026", "entropy_denominator", "Flow denominator used for origin entropy.", "extrapolated users", "source-provided processed variable", "QA/provenance", "valid raw H3 origins only."),
        ("mobility_processed_days", "metadata", "Locomizer final March 2026 mobility", "March 2026", "observed_days", "Number of processed daily mobility files.", "days", "renamed from observed_days", "coverage/provenance", "constant 31 in final dataset."),
        ("mobility_expected_days", "metadata", "Locomizer final March 2026 mobility", "March 2026", "expected_days", "Expected March day count.", "days", "renamed from expected_days", "coverage/provenance", "constant 31."),
        ("mobility_coverage_ratio", "metadata", "Locomizer final March 2026 mobility", "March 2026", "coverage_ratio", "Processed days divided by expected days.", "ratio", "renamed from coverage_ratio", "coverage/provenance", "constant 1.0 in final dataset."),
        ("mobility_weekday_count", "metadata", "Locomizer final March 2026 mobility", "March 2026", "observed_weekdays", "Number of processed weekday dates.", "days", "renamed from observed_weekdays", "coverage/provenance", "constant 22."),
        ("mobility_weekend_count", "metadata", "Locomizer final March 2026 mobility", "March 2026", "observed_weekend_days", "Number of processed weekend dates.", "days", "renamed from observed_weekend_days", "coverage/provenance", "constant 9."),
        ("mobility_missing_dates", "metadata", "Locomizer final March 2026 mobility", "March 2026", "missing_dates", "Pipe-separated missing mobility dates.", "text", "renamed from missing_dates; blanks retained as no missing dates", "coverage/provenance", "empty in final dataset."),
        ("mobility_provisional_flag", "metadata", "Locomizer final March 2026 mobility", "March 2026", "provisional_flag", "False when mobility source is final rather than provisional.", "boolean", "renamed from provisional_flag", "coverage/provenance", "constant false."),
        ("mobility_coverage_status", "metadata", "Final master build", "2026", "derived from mobility coverage fields", "Human-readable mobility coverage status.", "text", "derived metadata", "coverage/provenance", "constant final_31day_complete when QA passes."),
        ("weekday_weekend_zero_denominator", "metadata", "Locomizer final March 2026 mobility", "March 2026", "weekday_weekend_zero_denominator", "Flag for zero weekend denominator in weekday/weekend ratio.", "boolean", "source-provided processed variable", "QA/provenance", "all false in final dataset."),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "final_column_name",
            "conceptual_role",
            "source_dataset",
            "source_year",
            "original_source_column",
            "definition",
            "units",
            "transformation_status",
            "expected_use_in_modelling",
            "known_limitations",
        ],
    )


def compare_provisional(master: pd.DataFrame) -> tuple[bool, str]:
    candidates = sorted((PROJECT_ROOT / "data/processed/master").glob("*provisional*.csv"))
    candidates += sorted((PROJECT_ROOT / "outputs/master_dataset").glob("*provisional*.csv"))
    candidates = [p for p in candidates if p.name not in {PROVISIONAL_COMPARISON_CSV.name}]
    if not candidates:
        message = "No provisional master analysis dataset was found; comparison skipped."
        PROVISIONAL_COMPARISON_MD.write_text(
            "# Provisional vs Final Master Comparison\n\n" + message + "\n",
            encoding="utf-8",
        )
        pd.DataFrame([{"status": "skipped", "reason": message}]).to_csv(PROVISIONAL_COMPARISON_CSV, index=False)
        return False, message
    raise RuntimeError(f"Multiple/unexpected provisional master candidates found; refusing to guess: {candidates}")


def build_master() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    geography = read_geography()
    geography_base = geography[["LSOA21CD", "LSOA21NM", "geometry"]].copy()

    bres = read_csv_source(BRES_PATH)
    ptal = read_csv_source(PTAL_PATH)
    census = read_csv_source(CENSUS_PATH)
    poi = read_csv_source(POI_PATH)
    mobility = read_csv_source(MOBILITY_PATH)

    check_required_columns("BRES", bres, ["LSOA21CD", "employees_count"])
    check_required_columns("PTAL", ptal, ["LSOA21CD", "ptal_ai_mean"])
    check_required_columns("Census", census, ["LSOA21CD", "usual_resident_population_2021", "lsoa_area_km2", "population_density_km2"])
    check_required_columns("POI", poi, ["LSOA21CD", "amenity_poi_count", "amenity_density", "retail_poi_count", "retail_poi_density"])
    check_required_columns(
        "Mobility",
        mobility,
        [
            "LSOA21CD",
            "observed_days",
            "expected_days",
            "coverage_ratio",
            "observed_weekdays",
            "observed_weekend_days",
            "mobility_inflow_total",
            "mobility_inflow_mean_daily",
            "weekday_mean_daily_inflow",
            "weekend_mean_daily_inflow",
            "mobility_outflow_total",
            "net_flow",
        ],
    )

    sources = {
        "locked_geography": pd.DataFrame(geography.drop(columns="geometry")),
        "bres_2024": bres,
        "ptal_2023": ptal,
        "census_2021_population": census,
        "osm_poi": poi,
        "mobility_march_2026_final_31day": mobility,
    }
    coverage = join_coverage_rows(pd.DataFrame(geography_base.drop(columns="geometry")), sources)
    coverage.to_csv(JOIN_COVERAGE_OUTPUT, index=False)
    if (coverage[["unmatched_london_lsoas", "extra_non_london_lsoas", "duplicate_lsoa21cd_count", "missing_lsoa21cd_count"]] != 0).any().any():
        raise RuntimeError("Join coverage failed; see join coverage output.")

    poi_master = poi.rename(
        columns={
            "amenity_density": "amenity_poi_density_km2",
            "retail_poi_density": "retail_poi_density_km2",
        }
    )[["LSOA21CD", "amenity_poi_count", "amenity_poi_density_km2", "retail_poi_count", "retail_poi_density_km2"]]

    mobility_master = mobility.rename(
        columns={
            "observed_days": "mobility_processed_days",
            "expected_days": "mobility_expected_days",
            "coverage_ratio": "mobility_coverage_ratio",
            "observed_weekdays": "mobility_weekday_count",
            "observed_weekend_days": "mobility_weekend_count",
            "missing_dates": "mobility_missing_dates",
            "provisional_flag": "mobility_provisional_flag",
            "weekday_mean_daily_inflow": "weekday_inflow_mean_daily",
            "weekend_mean_daily_inflow": "weekend_inflow_mean_daily",
        }
    ).copy()
    mobility_master["mobility_missing_dates"] = mobility_master["mobility_missing_dates"].fillna("")
    mobility_master["mobility_coverage_status"] = np.where(
        (mobility_master["mobility_processed_days"] == 31)
        & (mobility_master["mobility_expected_days"] == 31)
        & (mobility_master["mobility_coverage_ratio"].round(12) == 1.0)
        & (~mobility_master["mobility_provisional_flag"].astype(bool)),
        "final_31day_complete",
        "not_final_complete",
    )

    mobility_cols = [
        "LSOA21CD",
        "mobility_processed_days",
        "mobility_expected_days",
        "mobility_coverage_ratio",
        "mobility_weekday_count",
        "mobility_weekend_count",
        "mobility_missing_dates",
        "mobility_provisional_flag",
        "mobility_coverage_status",
        "mobility_inflow_total",
        "mobility_inflow_mean_daily",
        "flow_entropy",
        "weekday_inflow_total",
        "weekend_inflow_total",
        "weekday_inflow_mean_daily",
        "weekend_inflow_mean_daily",
        "weekday_weekend_ratio",
        "weekday_weekend_zero_denominator",
        "unique_origins",
        "mobility_outflow_total",
        "net_flow",
        "london_origin_flow",
        "valid_non_london_origin_flow",
        "other_origin_flow",
        "other_origin_share",
        "valid_origin_flow_share",
        "external_origin_share",
        "entropy_denominator",
    ]

    master_gdf = geography_base.merge(bres, on="LSOA21CD", how="left", validate="one_to_one")
    master_gdf = master_gdf.merge(ptal, on="LSOA21CD", how="left", validate="one_to_one")
    master_gdf = master_gdf.merge(
        census[["LSOA21CD", "usual_resident_population_2021", "lsoa_area_km2", "population_density_km2"]],
        on="LSOA21CD",
        how="left",
        validate="one_to_one",
    )
    master_gdf = master_gdf.merge(poi_master, on="LSOA21CD", how="left", validate="one_to_one")
    master_gdf = master_gdf.merge(mobility_master[mobility_cols], on="LSOA21CD", how="left", validate="one_to_one")

    ordered_cols = [
        "LSOA21CD",
        "LSOA21NM",
        "employees_count",
        "employee_density_km2",
        "log1p_employees_count",
        "log1p_employee_density_km2",
        "sqrt_employee_density_km2",
        "ptal_ai_mean",
        "ptal_ai_median",
        "ptal_ai_min",
        "ptal_ai_max",
        "ptal_mean_category",
        "usual_resident_population_2021",
        "lsoa_area_km2",
        "population_density_km2",
        "amenity_poi_count",
        "amenity_poi_density_km2",
        "retail_poi_count",
        "retail_poi_density_km2",
        *[col for col in mobility_cols if col != "LSOA21CD"],
        "geometry",
    ]
    master_gdf = master_gdf[ordered_cols].copy()

    csv_df = pd.DataFrame(master_gdf.drop(columns="geometry"))
    csv_df.to_csv(CSV_OUTPUT, index=False)
    master_gdf.to_file(GPKG_OUTPUT, layer="master_analysis_dataset_final_v1_0", driver="GPKG")

    qa_rows: list[dict] = []
    add_qa(qa_rows, "output_row_count", len(csv_df), "pass" if len(csv_df) == EXPECTED_LSOAS else "fail")
    add_qa(qa_rows, "unique_lsoa21cd", csv_df["LSOA21CD"].nunique(), "pass" if csv_df["LSOA21CD"].nunique() == EXPECTED_LSOAS else "fail")
    add_qa(qa_rows, "duplicate_lsoa21cd", int(csv_df["LSOA21CD"].duplicated().sum()), "pass")
    add_qa(qa_rows, "missing_lsoa21cd", int(csv_df["LSOA21CD"].isna().sum()), "pass")
    valid_format_count = int(csv_df["LSOA21CD"].astype("string").str.match(LSOA_PATTERN, na=False).sum())
    add_qa(qa_rows, "valid_lsoa21cd_format_count", valid_format_count, "pass" if valid_format_count == EXPECTED_LSOAS else "fail")
    add_qa(qa_rows, "unmatched_locked_london_lsoas", int(csv_df[analytical_variable_columns(csv_df)].isna().all(axis=1).sum()), "pass")
    add_qa(qa_rows, "gpkg_crs", str(master_gdf.crs), "pass" if str(master_gdf.crs).upper() == "EPSG:27700" else "fail")
    add_qa(qa_rows, "geometry_count", len(master_gdf.geometry), "pass" if len(master_gdf.geometry) == EXPECTED_LSOAS else "fail")
    invalid_geoms = int((~master_gdf.geometry.is_valid).sum())
    empty_geoms = int(master_gdf.geometry.is_empty.sum())
    add_qa(qa_rows, "invalid_geometry_count", invalid_geoms, "pass" if invalid_geoms == 0 else "fail")
    add_qa(qa_rows, "empty_geometry_count", empty_geoms, "pass" if empty_geoms == 0 else "fail")

    logical_checks = {
        "mobility_inflow_total_nonnegative": (csv_df["mobility_inflow_total"] >= 0).all(),
        "mobility_inflow_mean_daily_nonnegative": (csv_df["mobility_inflow_mean_daily"] >= 0).all(),
        "flow_entropy_nonnegative": (csv_df["flow_entropy"] >= 0).all(),
        "weekday_inflow_total_nonnegative": (csv_df["weekday_inflow_total"] >= 0).all(),
        "weekend_inflow_total_nonnegative": (csv_df["weekend_inflow_total"] >= 0).all(),
        "unique_origins_nonnegative": (csv_df["unique_origins"] >= 0).all(),
        "amenity_poi_count_nonnegative": (csv_df["amenity_poi_count"] >= 0).all(),
        "retail_poi_count_nonnegative": (csv_df["retail_poi_count"] >= 0).all(),
        "amenity_poi_density_km2_nonnegative": (csv_df["amenity_poi_density_km2"] >= 0).all(),
        "retail_poi_density_km2_nonnegative": (csv_df["retail_poi_density_km2"] >= 0).all(),
        "population_density_km2_nonnegative": (csv_df["population_density_km2"] >= 0).all(),
        "employees_count_nonnegative": (csv_df["employees_count"] >= 0).all(),
        "weekday_weekend_ratio_finite": np.isfinite(csv_df["weekday_weekend_ratio"]).all(),
        "mobility_coverage_status_final": (csv_df["mobility_coverage_status"] == "final_31day_complete").all(),
    }
    for metric, ok in logical_checks.items():
        add_qa(qa_rows, metric, bool(ok), "pass" if ok else "fail")

    relationships = {
        "mobility_inflow_mean_daily_equals_total_div_31": (
            csv_df["mobility_inflow_mean_daily"] - csv_df["mobility_inflow_total"] / 31
        ).abs(),
        "weekday_inflow_mean_daily_equals_total_div_weekdays": (
            csv_df["weekday_inflow_mean_daily"] - csv_df["weekday_inflow_total"] / csv_df["mobility_weekday_count"]
        ).abs(),
        "weekend_inflow_mean_daily_equals_total_div_weekends": (
            csv_df["weekend_inflow_mean_daily"] - csv_df["weekend_inflow_total"] / csv_df["mobility_weekend_count"]
        ).abs(),
        "weekday_weekend_ratio_equals_weekday_over_weekend": (
            csv_df["weekday_weekend_ratio"] - csv_df["weekday_inflow_mean_daily"] / csv_df["weekend_inflow_mean_daily"]
        ).abs(),
        "population_density_km2_equals_population_over_area": (
            csv_df["population_density_km2"] - csv_df["usual_resident_population_2021"] / csv_df["lsoa_area_km2"]
        ).abs(),
    }
    for metric, diff in relationships.items():
        max_diff = float(diff.max())
        add_qa(qa_rows, metric, max_diff, "pass" if max_diff <= TOLERANCE else "fail", f"tolerance={TOLERANCE}")

    qa = pd.concat([pd.DataFrame(qa_rows), numeric_summary_qa(csv_df, analytical_variable_columns(csv_df))], ignore_index=True)
    qa.to_csv(QA_OUTPUT, index=False)

    dictionary = build_dictionary()
    missing_dict = sorted(set(csv_df.columns) - set(dictionary["final_column_name"]) - {"geometry"})
    if missing_dict:
        raise RuntimeError(f"Dictionary missing columns: {missing_dict}")
    dictionary.to_csv(DICTIONARY_OUTPUT, index=False)

    comparison_available, comparison_message = compare_provisional(csv_df)

    status_counts = qa["status"].value_counts().to_dict()
    failures = qa[qa["status"] == "fail"]
    warnings = qa[qa["status"] == "warn"]
    if not failures.empty:
        raise RuntimeError(f"Master QA failed: {failures['metric'].tolist()}")

    manifest = f"""# Master Analysis Dataset Final v1.0 Manifest

Generated: {datetime.now().isoformat(timespec="seconds")}

## Inputs

- Locked geography: `{rel(GEOGRAPHY_PATH)}`
- BRES 2024: `{rel(BRES_PATH)}`
- PTAL 2023: `{rel(PTAL_PATH)}`
- Census 2021 population: `{rel(CENSUS_PATH)}`
- OSM POI: `{rel(POI_PATH)}`
- Final March 2026 mobility: `{rel(MOBILITY_PATH)}`

## Outputs

- CSV: `{rel(CSV_OUTPUT)}`
- GPKG: `{rel(GPKG_OUTPUT)}`
- Report: `{rel(REPORT_OUTPUT)}`
- QA: `{rel(QA_OUTPUT)}`
- Join coverage: `{rel(JOIN_COVERAGE_OUTPUT)}`
- Dictionary: `{rel(DICTIONARY_OUTPUT)}`
- Provisional comparison: `{rel(PROVISIONAL_COMPARISON_MD)}`

## Build Rule

The locked Greater London LSOA 2021 geography was used as the left-hand base. All joins used `LSOA21CD` after string casting and whitespace trimming. The final output is cross-sectional and explanatory/associational, not causal or forecasting.
"""
    MANIFEST_OUTPUT.write_text(manifest, encoding="utf-8")

    grouped = dictionary.groupby("conceptual_role")["final_column_name"].apply(lambda x: ", ".join(x)).reset_index()
    grouped_md = "\n".join(f"- {row.conceptual_role}: `{row.final_column_name}`" for row in grouped.itertuples(index=False))
    coverage_md = coverage.to_markdown(index=False)
    report = f"""# Master Analysis Dataset Final v1.0 Report

## Status

The final Master Analysis Dataset v1.0 passed QA and is ready for ESDA. No ESDA, modelling, spatial statistics, machine learning, standardisation, winsorisation, or log transformations were run in this task.

## Inputs Used

- Locked Greater London LSOA 2021 geometry: `{rel(GEOGRAPHY_PATH)}`
- BRES 2024 employment: `{rel(BRES_PATH)}`
- PTAL 2023: `{rel(PTAL_PATH)}`
- Census 2021 population: `{rel(CENSUS_PATH)}`
- OSM POI controls: `{rel(POI_PATH)}`
- Final March 2026 mobility, 31-day: `{rel(MOBILITY_PATH)}`

## Outputs

- CSV: `{rel(CSV_OUTPUT)}`
- GPKG: `{rel(GPKG_OUTPUT)}`
- QA: `{rel(QA_OUTPUT)}`
- Join coverage: `{rel(JOIN_COVERAGE_OUTPUT)}`
- Dictionary: `{rel(DICTIONARY_OUTPUT)}`
- Manifest: `{rel(MANIFEST_OUTPUT)}`

## Join Coverage

{coverage_md}

## QA Summary

- Pass checks: {int(status_counts.get("pass", 0))}
- Warning checks: {int(status_counts.get("warn", 0))}
- Failed checks: {int(status_counts.get("fail", 0))}
- Output row count: {len(csv_df):,}
- Output column count, CSV: {csv_df.shape[1]:,}
- Unique `LSOA21CD`: {csv_df["LSOA21CD"].nunique():,}
- Duplicate `LSOA21CD`: {int(csv_df["LSOA21CD"].duplicated().sum()):,}
- Missing `LSOA21CD`: {int(csv_df["LSOA21CD"].isna().sum()):,}
- GPKG CRS: `{master_gdf.crs}`
- Geometry count: {len(master_gdf.geometry):,}
- Invalid geometries: {invalid_geoms:,}
- Empty geometries: {empty_geoms:,}

## Expected Warnings And Data Characteristics

- `net_flow` has negative values for {int((csv_df["net_flow"] < 0).sum()):,} LSOAs. This is expected and retained as a robustness-variable characteristic, not a QA failure.
- `employees_count` has {int((csv_df["employees_count"] == 0).sum()):,} zero values.
- Amenity POI count has {int((csv_df["amenity_poi_count"] == 0).sum()):,} zero values.
- Retail POI count has {int((csv_df["retail_poi_count"] == 0).sum()):,} zero values.
- Mobility and employment variables remain highly skewed; distributional handling belongs to ESDA/modelling, not this build.

## Retained Variables By Role

{grouped_md}

## Modelling Notes

- `employees_count` is the dependent variable identified from the current Variable Design Book and processed BRES output.
- `ptal_ai_mean` is the main continuous PTAL accessibility variable; `ptal_mean_category` is descriptive only.
- `population_density_km2`, `amenity_poi_density_km2`, and `retail_poi_density_km2` are the main density controls; counts are retained for QA/descriptive analysis.
- Preserve both `mobility_inflow_total` and `mobility_inflow_mean_daily`, but do not use them together in the same baseline model because they are mechanically related.
- `net_flow` is retained only as a robustness variable because OD symmetry cannot be independently verified from destination-format files alone.

## Temporal Alignment

- BRES employment: 2024
- PTAL: 2023
- Census population: 2021
- OSM POIs: 2026-07-06 extract, based on `greater-london-260706.osm.pbf`
- Locomizer mobility: March 2026

The dataset is cross-sectional and intended for explanatory/associational analysis. It is not a causal or forecasting design.

## Provisional Comparison

{comparison_message}

## Reproduction Command

```bash
python3 scripts/master/build_master_analysis_dataset.py
```
"""
    REPORT_OUTPUT.write_text(report, encoding="utf-8")

    print(f"Wrote CSV: {rel(CSV_OUTPUT)}")
    print(f"Wrote GPKG: {rel(GPKG_OUTPUT)}")
    print(f"Wrote report: {rel(REPORT_OUTPUT)}")
    print(f"QA status: {int(status_counts.get('pass', 0))} pass, {int(status_counts.get('warn', 0))} warn, {int(status_counts.get('fail', 0))} fail")


def main() -> None:
    build_master()


if __name__ == "__main__":
    main()
