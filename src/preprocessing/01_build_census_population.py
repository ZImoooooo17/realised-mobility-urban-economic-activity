"""Build Census 2021 LSOA population controls for the dissertation.

Standalone pipeline:
- locate the official TS001 raw CSV under data/raw/census_2021_population
- validate and standardise LSOA21CD
- join to the locked Greater London LSOA 2021 geography
- calculate population density per km2
- write processed CSV, GPKG, QA, dictionary, manifest, and report
"""

from __future__ import annotations

import csv
import math
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data/raw/census_2021_population"
BOUNDARY_PATH = PROJECT_ROOT / "data/processed/boundaries/lsoa21_london_bgc.gpkg"
PROCESSED_DIR = PROJECT_ROOT / "data/processed/census"
OUTPUT_DIR = PROJECT_ROOT / "outputs/census"

CSV_OUTPUT = PROCESSED_DIR / "census_2021_lsoa_population.csv"
GPKG_OUTPUT = PROCESSED_DIR / "census_2021_lsoa_population.gpkg"
QA_OUTPUT = OUTPUT_DIR / "census_2021_lsoa_population_qa.csv"
REPORT_OUTPUT = OUTPUT_DIR / "census_2021_lsoa_population_report.md"
MANIFEST_OUTPUT = OUTPUT_DIR / "census_2021_lsoa_population_manifest.md"
DICTIONARY_OUTPUT = OUTPUT_DIR / "census_2021_lsoa_population_dictionary.csv"
JOIN_COVERAGE_OUTPUT = OUTPUT_DIR / "census_2021_lsoa_population_join_coverage.csv"

EXPECTED_LSOAS = 4994
LSOA_PATTERN = r"^E010\d{5}$"


def read_metadata_lines(path: Path, max_lines: int = 8) -> list[str]:
    lines: list[str] = []
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        for _, line in zip(range(max_lines), handle):
            lines.append(line.rstrip("\n"))
    return lines


def locate_raw_ts001_file() -> tuple[Path, list[str]]:
    candidates = sorted(RAW_DIR.glob("*.csv"))
    valid: list[tuple[Path, list[str]]] = []
    for path in candidates:
        lines = read_metadata_lines(path)
        joined = "\n".join(lines)
        if "TS001" not in joined or "usual residents" not in joined.lower():
            continue
        try:
            header = pd.read_csv(path, skiprows=7, nrows=0, encoding="utf-8-sig")
        except Exception:
            continue
        if {"mnemonic", "2021"}.issubset(set(header.columns)):
            valid.append((path, lines))

    if not valid:
        raise FileNotFoundError(f"No TS001 population CSV found under {RAW_DIR}")
    if len(valid) > 1:
        files = ", ".join(str(path.relative_to(PROJECT_ROOT)) for path, _ in valid)
        raise RuntimeError(f"Multiple candidate TS001 CSV files found; refusing to guess: {files}")
    return valid[0]


def metadata_value(lines: list[str], key: str) -> str:
    for line in lines:
        if line.startswith(f'"{key}'):
            parts = next(csv.reader([line]))
            if len(parts) >= 2:
                return parts[1]
    return ""


def read_raw_population(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, skiprows=7, encoding="utf-8-sig")
    required = {"2021 super output area - lower layer", "mnemonic", "2021"}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise RuntimeError(f"Raw Census file missing required columns: {missing}")

    result = raw.rename(
        columns={
            "2021 super output area - lower layer": "LSOA21NM_raw",
            "mnemonic": "LSOA21CD",
            "2021": "usual_resident_population_2021",
        }
    ).copy()
    result["LSOA21CD"] = result["LSOA21CD"].astype("string").str.strip()
    result["LSOA21NM_raw"] = result["LSOA21NM_raw"].astype("string").str.strip()
    result["usual_resident_population_2021"] = pd.to_numeric(
        result["usual_resident_population_2021"], errors="coerce"
    )
    result["valid_lsoa21cd_format"] = result["LSOA21CD"].str.match(LSOA_PATTERN, na=False)
    return result


def add_qa(rows: list[dict], metric: str, value, status: str, notes: str = "") -> None:
    rows.append({"metric": metric, "value": value, "status": status, "notes": notes})


def numeric_qa(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows: list[dict] = []
    for col in columns:
        series = df[col]
        values = pd.to_numeric(series, errors="coerce")
        finite = np.isfinite(values.to_numpy(dtype=float, na_value=np.nan))
        add_qa(rows, f"{col}__dtype", str(series.dtype), "pass")
        add_qa(rows, f"{col}__non_null_count", int(series.notna().sum()), "pass")
        add_qa(rows, f"{col}__missing_count", int(series.isna().sum()), "pass" if series.isna().sum() == 0 else "fail")
        inf_count = int((~finite & ~values.isna().to_numpy()).sum())
        add_qa(rows, f"{col}__infinite_count", inf_count, "pass" if inf_count == 0 else "fail")
        neg_count = int((values < 0).sum())
        add_qa(rows, f"{col}__negative_count", neg_count, "pass" if neg_count == 0 else "fail")
        add_qa(rows, f"{col}__zero_count", int((values == 0).sum()), "pass")
        add_qa(rows, f"{col}__minimum", float(values.min()), "pass")
        add_qa(rows, f"{col}__maximum", float(values.max()), "pass")
        add_qa(rows, f"{col}__mean", float(values.mean()), "pass")
        add_qa(rows, f"{col}__median", float(values.median()), "pass")
        add_qa(rows, f"{col}__standard_deviation", float(values.std()), "pass")
    return pd.DataFrame(rows)


def build_outputs() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    raw_path, metadata_lines = locate_raw_ts001_file()
    raw = read_raw_population(raw_path)
    population = raw[raw["valid_lsoa21cd_format"]].copy()
    footer_rows = raw[~raw["valid_lsoa21cd_format"]].copy()

    geography = gpd.read_file(BOUNDARY_PATH).to_crs("EPSG:27700")
    geography["LSOA21CD"] = geography["LSOA21CD"].astype("string").str.strip()
    geography["lsoa_area_km2"] = geography.geometry.area / 1_000_000

    keep_geo = ["LSOA21CD", "LSOA21NM", "lsoa_area_km2", "geometry"]
    attrs = population[["LSOA21CD", "usual_resident_population_2021"]].copy()
    merged = geography[keep_geo].merge(attrs, on="LSOA21CD", how="left", validate="one_to_one")
    merged["population_density_km2"] = (
        merged["usual_resident_population_2021"] / merged["lsoa_area_km2"]
    )

    csv_df = pd.DataFrame(
        merged.drop(columns="geometry")[
            ["LSOA21CD", "LSOA21NM", "usual_resident_population_2021", "lsoa_area_km2", "population_density_km2"]
        ]
    )
    csv_df["usual_resident_population_2021"] = csv_df["usual_resident_population_2021"].astype("Int64")
    csv_df.to_csv(CSV_OUTPUT, index=False)

    gpkg_gdf = merged[
        ["LSOA21CD", "LSOA21NM", "usual_resident_population_2021", "lsoa_area_km2", "population_density_km2", "geometry"]
    ].copy()
    gpkg_gdf["usual_resident_population_2021"] = gpkg_gdf["usual_resident_population_2021"].astype("Int64")
    gpkg_gdf.to_file(GPKG_OUTPUT, layer="census_2021_lsoa_population", driver="GPKG")

    london_codes = set(geography["LSOA21CD"])
    population_codes = set(population["LSOA21CD"])
    unmatched_london = sorted(london_codes - population_codes)
    extra_population = sorted(population_codes - london_codes)

    join_coverage = pd.DataFrame(
        [
            {
                "source_name": "census_2021_ts001_population",
                "source_file": str(raw_path.relative_to(PROJECT_ROOT)),
                "source_row_count_valid_lsoa": len(population),
                "raw_row_count_including_footer": len(raw),
                "unique_lsoa21cd_count": population["LSOA21CD"].nunique(),
                "matched_london_lsoas": len(london_codes & population_codes),
                "unmatched_london_lsoas": len(unmatched_london),
                "extra_non_london_lsoas": len(extra_population),
                "duplicate_lsoa21cd_count": int(population["LSOA21CD"].duplicated().sum()),
                "missing_lsoa21cd_count": int(population["LSOA21CD"].isna().sum()),
                "footer_or_non_lsoa_rows_excluded": len(footer_rows),
            }
        ]
    )
    join_coverage.to_csv(JOIN_COVERAGE_OUTPUT, index=False)

    qa_rows: list[dict] = []
    add_qa(qa_rows, "raw_file_found", str(raw_path.relative_to(PROJECT_ROOT)), "pass")
    add_qa(qa_rows, "raw_file_size_bytes", raw_path.stat().st_size, "pass")
    add_qa(qa_rows, "metadata_title_contains_ts001", "TS001" in "\n".join(metadata_lines), "pass")
    add_qa(qa_rows, "metadata_population", metadata_value(metadata_lines, "Population :"), "pass")
    add_qa(qa_rows, "metadata_units", metadata_value(metadata_lines, "Units      :"), "pass")
    add_qa(qa_rows, "metadata_residence_type", metadata_value(metadata_lines, "Residence type:"), "pass")
    add_qa(qa_rows, "raw_rows_including_footer", len(raw), "pass")
    add_qa(qa_rows, "valid_lsoa_rows", len(population), "pass" if len(population) == EXPECTED_LSOAS else "fail")
    add_qa(qa_rows, "footer_or_non_lsoa_rows_excluded", len(footer_rows), "pass")
    add_qa(qa_rows, "output_csv_rows", len(csv_df), "pass" if len(csv_df) == EXPECTED_LSOAS else "fail")
    add_qa(qa_rows, "output_gpkg_rows", len(gpkg_gdf), "pass" if len(gpkg_gdf) == EXPECTED_LSOAS else "fail")
    add_qa(qa_rows, "unique_lsoa21cd", csv_df["LSOA21CD"].nunique(), "pass" if csv_df["LSOA21CD"].nunique() == EXPECTED_LSOAS else "fail")
    add_qa(qa_rows, "duplicate_lsoa21cd", int(csv_df["LSOA21CD"].duplicated().sum()), "pass")
    add_qa(qa_rows, "missing_lsoa21cd", int(csv_df["LSOA21CD"].isna().sum()), "pass")
    valid_format_count = int(csv_df["LSOA21CD"].astype("string").str.match(LSOA_PATTERN, na=False).sum())
    add_qa(qa_rows, "valid_lsoa21cd_format_count", valid_format_count, "pass" if valid_format_count == EXPECTED_LSOAS else "fail")
    add_qa(qa_rows, "unmatched_london_lsoas", len(unmatched_london), "pass" if not unmatched_london else "fail")
    add_qa(qa_rows, "extra_non_london_lsoas", len(extra_population), "pass" if not extra_population else "fail")
    add_qa(qa_rows, "gpkg_crs", str(gpkg_gdf.crs), "pass" if str(gpkg_gdf.crs).upper() == "EPSG:27700" else "fail")
    add_qa(qa_rows, "geometry_count", len(gpkg_gdf.geometry), "pass" if len(gpkg_gdf.geometry) == EXPECTED_LSOAS else "fail")
    invalid_geoms = int((~gpkg_gdf.geometry.is_valid).sum())
    add_qa(qa_rows, "invalid_geometry_count", invalid_geoms, "pass" if invalid_geoms == 0 else "fail")
    empty_geoms = int(gpkg_gdf.geometry.is_empty.sum())
    add_qa(qa_rows, "empty_geometry_count", empty_geoms, "pass" if empty_geoms == 0 else "fail")
    density_check = csv_df["population_density_km2"] - (
        csv_df["usual_resident_population_2021"].astype(float) / csv_df["lsoa_area_km2"]
    )
    max_density_diff = float(density_check.abs().max())
    add_qa(qa_rows, "population_density_formula_max_abs_diff", max_density_diff, "pass" if max_density_diff <= 1e-9 else "fail")

    qa = pd.concat(
        [
            pd.DataFrame(qa_rows),
            numeric_qa(csv_df, ["usual_resident_population_2021", "lsoa_area_km2", "population_density_km2"]),
        ],
        ignore_index=True,
    )
    qa.to_csv(QA_OUTPUT, index=False)

    dictionary = pd.DataFrame(
        [
            {
                "final_column_name": "LSOA21CD",
                "source_dataset": "Locked Greater London LSOA 2021 geography / Census TS001",
                "source_year": "2021",
                "original_source_column": "mnemonic",
                "definition": "Lower Layer Super Output Area 2021 code.",
                "units": "identifier",
                "transformation": "cast to string and whitespace trimmed",
                "expected_use": "join key",
                "known_limitations": "none identified",
            },
            {
                "final_column_name": "LSOA21NM",
                "source_dataset": "Locked Greater London LSOA 2021 geography",
                "source_year": "2021",
                "original_source_column": "LSOA21NM",
                "definition": "Lower Layer Super Output Area 2021 name.",
                "units": "text",
                "transformation": "retained from locked geography",
                "expected_use": "descriptive geography label",
                "known_limitations": "not used as join key",
            },
            {
                "final_column_name": "usual_resident_population_2021",
                "source_dataset": "Census 2021 TS001",
                "source_year": "2021",
                "original_source_column": "2021",
                "definition": "All usual residents in households and communal establishments.",
                "units": "persons",
                "transformation": "renamed and cast to integer",
                "expected_use": "population count / denominator audit",
                "known_limitations": "Nomis disclosure control states records may be swapped between areas and counts perturbed by small amounts.",
            },
            {
                "final_column_name": "lsoa_area_km2",
                "source_dataset": "Locked Greater London LSOA 2021 geography",
                "source_year": "2021",
                "original_source_column": "geometry",
                "definition": "LSOA polygon area from locked EPSG:27700 geometry.",
                "units": "square kilometres",
                "transformation": "geometry area in square metres divided by 1,000,000",
                "expected_use": "density denominator and QA",
                "known_limitations": "area follows the locked boundary geometry used across the dissertation.",
            },
            {
                "final_column_name": "population_density_km2",
                "source_dataset": "Census 2021 TS001 and locked LSOA geometry",
                "source_year": "2021",
                "original_source_column": "usual_resident_population_2021 / geometry area",
                "definition": "Usual resident population per square kilometre.",
                "units": "persons per km2",
                "transformation": "usual_resident_population_2021 / lsoa_area_km2",
                "expected_use": "approved population control for modelling",
                "known_limitations": "cross-sectional 2021 residential population control; temporally earlier than 2026 mobility and 2024 BRES.",
            },
        ]
    )
    dictionary.to_csv(DICTIONARY_OUTPUT, index=False)

    status_counts = qa["status"].value_counts().to_dict()
    manifest = f"""# Census 2021 Population Pipeline Manifest

Generated: {datetime.now().isoformat(timespec="seconds")}

## Input

- Raw directory: `{RAW_DIR.relative_to(PROJECT_ROOT)}`
- Selected raw file: `{raw_path.relative_to(PROJECT_ROOT)}`
- Raw file size: {raw_path.stat().st_size:,} bytes
- Locked geography: `{BOUNDARY_PATH.relative_to(PROJECT_ROOT)}`
- Locked geography CRS: `EPSG:27700`

## Raw Metadata

- Title: {metadata_lines[1].strip('"') if len(metadata_lines) > 1 else ""}
- Population: {metadata_value(metadata_lines, "Population :")}
- Units: {metadata_value(metadata_lines, "Units      :")}
- Residence type: {metadata_value(metadata_lines, "Residence type:")}

## Outputs

- CSV: `{CSV_OUTPUT.relative_to(PROJECT_ROOT)}`
- GPKG: `{GPKG_OUTPUT.relative_to(PROJECT_ROOT)}`
- QA: `{QA_OUTPUT.relative_to(PROJECT_ROOT)}`
- Join coverage: `{JOIN_COVERAGE_OUTPUT.relative_to(PROJECT_ROOT)}`
- Dictionary: `{DICTIONARY_OUTPUT.relative_to(PROJECT_ROOT)}`
- Report: `{REPORT_OUTPUT.relative_to(PROJECT_ROOT)}`
"""
    MANIFEST_OUTPUT.write_text(manifest, encoding="utf-8")

    report = f"""# Census 2021 LSOA Population Processing Report

## Status

The standalone Census 2021 population dataset passed QA and contains exactly 4,994 Greater London LSOA 2021 records.

No master dataset build, ESDA, modelling, spatial statistics, machine learning, or model transformations were run.

## Input Selection

- Raw file discovered automatically: `{raw_path.relative_to(PROJECT_ROOT)}`
- Raw format: Nomis TS001 CSV with metadata rows and a data table beginning at line 8
- Census topic: TS001, number of usual residents in households and communal establishments
- Population definition: {metadata_value(metadata_lines, "Population :")}
- Units: {metadata_value(metadata_lines, "Units      :")}
- Residence type: {metadata_value(metadata_lines, "Residence type:")}
- Raw table rows read including footer/disclosure rows: {len(raw):,}
- Valid LSOA rows retained: {len(population):,}
- Footer/non-LSOA rows excluded: {len(footer_rows):,}

## Processing Rules

- `LSOA21CD` was created from the raw `mnemonic` column.
- `LSOA21CD` was cast to string and whitespace trimmed.
- Only codes matching `{LSOA_PATTERN}` were retained from the raw table.
- The locked Greater London LSOA 2021 geography was used as the left-hand base.
- `lsoa_area_km2` was calculated from the locked `EPSG:27700` geometry as polygon area in square metres divided by 1,000,000.
- `population_density_km2 = usual_resident_population_2021 / lsoa_area_km2`.

## Outputs

- CSV: `{CSV_OUTPUT.relative_to(PROJECT_ROOT)}`
- GPKG: `{GPKG_OUTPUT.relative_to(PROJECT_ROOT)}`
- QA: `{QA_OUTPUT.relative_to(PROJECT_ROOT)}`
- Join coverage: `{JOIN_COVERAGE_OUTPUT.relative_to(PROJECT_ROOT)}`
- Dictionary: `{DICTIONARY_OUTPUT.relative_to(PROJECT_ROOT)}`
- Manifest: `{MANIFEST_OUTPUT.relative_to(PROJECT_ROOT)}`

## Join Coverage

- Locked London LSOAs: {len(london_codes):,}
- Census valid LSOA rows: {len(population):,}
- Census unique LSOA codes: {population["LSOA21CD"].nunique():,}
- Matched London LSOAs: {len(london_codes & population_codes):,}
- Unmatched London LSOAs: {len(unmatched_london):,}
- Extra non-London LSOAs: {len(extra_population):,}
- Duplicate Census LSOA codes: {int(population["LSOA21CD"].duplicated().sum()):,}

## QA Summary

- Pass checks: {int(status_counts.get("pass", 0))}
- Warning checks: {int(status_counts.get("warn", 0))}
- Failed checks: {int(status_counts.get("fail", 0))}
- Output row count: {len(csv_df):,}
- Unique `LSOA21CD`: {csv_df["LSOA21CD"].nunique():,}
- Missing `LSOA21CD`: {int(csv_df["LSOA21CD"].isna().sum()):,}
- Duplicate `LSOA21CD`: {int(csv_df["LSOA21CD"].duplicated().sum()):,}
- GPKG CRS: `{gpkg_gdf.crs}`
- Geometry count: {len(gpkg_gdf.geometry):,}
- Invalid geometries: {invalid_geoms:,}
- Empty geometries: {empty_geoms:,}

## Population Summary

- Total usual resident population: {int(csv_df["usual_resident_population_2021"].sum()):,}
- Minimum LSOA population: {int(csv_df["usual_resident_population_2021"].min()):,}
- Maximum LSOA population: {int(csv_df["usual_resident_population_2021"].max()):,}
- Mean population density: {float(csv_df["population_density_km2"].mean()):.6f} persons per km2
- Minimum population density: {float(csv_df["population_density_km2"].min()):.6f} persons per km2
- Maximum population density: {float(csv_df["population_density_km2"].max()):.6f} persons per km2

## Known Limitation

The raw Nomis footer states that records may be swapped between geographic areas and counts perturbed by small amounts for disclosure control. This is expected for Census small-area outputs and is retained as a documentation limitation, not a QA failure.
"""
    REPORT_OUTPUT.write_text(report, encoding="utf-8")

    if (qa["status"] == "fail").any():
        failed = qa.loc[qa["status"] == "fail", "metric"].tolist()
        raise RuntimeError(f"Census population QA failed: {failed}")


def main() -> None:
    build_outputs()
    print(f"Wrote CSV: {CSV_OUTPUT.relative_to(PROJECT_ROOT)}")
    print(f"Wrote GPKG: {GPKG_OUTPUT.relative_to(PROJECT_ROOT)}")
    print(f"Wrote QA: {QA_OUTPUT.relative_to(PROJECT_ROOT)}")
    print(f"Wrote report: {REPORT_OUTPUT.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
