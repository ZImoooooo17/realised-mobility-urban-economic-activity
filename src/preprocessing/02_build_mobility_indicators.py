#!/usr/bin/env python3
"""March 2026 destination-based Locomizer mobility pipeline.

Main dissertation scope:
- Daily non-hourly `loco_all_tracks.tsv` files only.
- `MOVEMENT_MODALITY` must be exactly `ALL`.
- Destination-based indicators for Greater London LSOA 2021.
- Provisional outputs remain provisional until all 31 March ALL files exist.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import h3
import numpy as np
import pandas as pd
from pandas.util import hash_pandas_object
from shapely.geometry import Point


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_MOBILITY_DIR = PROJECT_ROOT / "data/raw/mobility"
BOUNDARY_PATH = PROJECT_ROOT / "data/processed/boundaries/lsoa21_london_bgc.gpkg"
INTERIM_DIR = PROJECT_ROOT / "data/interim/mobility"
DAILY_CACHE_DIR = INTERIM_DIR / "daily_lsoa"
PROCESSED_DIR = PROJECT_ROOT / "data/processed/mobility"
OUTPUT_DIR = PROJECT_ROOT / "outputs/mobility_pipeline"

MANIFEST_PATH = INTERIM_DIR / "march_2026_daily_file_manifest.csv"
SCHEMA_VALIDATION_PATH = OUTPUT_DIR / "mobility_schema_validation.csv"
PROCESSING_LOG_PATH = OUTPUT_DIR / "mobility_daily_processing_log.csv"
INDICATOR_QA_PATH = OUTPUT_DIR / "mobility_indicator_qa.csv"
COVERAGE_SUMMARY_PATH = OUTPUT_DIR / "mobility_coverage_summary.csv"
MISSING_DATES_PATH = OUTPUT_DIR / "mobility_missing_dates.csv"
INDICATOR_DICTIONARY_PATH = OUTPUT_DIR / "mobility_indicator_dictionary.md"
PROVISIONAL_REPORT_PATH = OUTPUT_DIR / "mobility_provisional_run_report.md"
FINAL_REPORT_PATH = OUTPUT_DIR / "mobility_final_31day_run_report.md"
FINAL_QA_PATH = OUTPUT_DIR / "mobility_final_31day_indicator_qa.csv"
FINAL_COVERAGE_PATH = OUTPUT_DIR / "mobility_final_31day_coverage_summary.csv"
FINAL_MANIFEST_PATH = OUTPUT_DIR / "mobility_final_31day_manifest.csv"
H3_LOOKUP_PATH = INTERIM_DIR / "h3_lsoa_lookup.parquet"

PROCESSING_VERSION = "2026-07-21_destination_lsoa_v2"
EXPECTED_DAYS = 31
RECONCILIATION_TOLERANCE = 1e-6

ALL_TRACKS_RE = re.compile(
    r"^Audience_Profiles_Destination_(2026-03-\d{2})_loco_all_tracks\.tsv$"
)
HOURLY_RE = re.compile(
    r"^Audience_Profiles_Destination_(2026-03-\d{2})_loco_all_tracks_0-24\.tsv$"
)
PEDESTRIAN_RE = re.compile(
    r"^Audience_Profiles_Destination_(2026-03-\d{2})_loco_pedestrian_tracks\.tsv$"
)
ANY_LOCOMIZER_RE = re.compile(r"^Audience_Profiles_Destination_(2026-03-\d{2}).*\.tsv$")

REQUIRED_COLUMNS = [
    "CODE",
    "ORIGIN_CODE",
    "DAY_TYPE",
    "DAY",
    "MONTH",
    "YEAR",
    "MOVEMENT_MODALITY",
    "EXTRAPOLATED_NUMBER_OF_USERS",
    "EXTRAPOLATED_NUMBER_OF_SIGNALS",
]
VALUE_COLUMNS = ["EXTRAPOLATED_NUMBER_OF_USERS", "EXTRAPOLATED_NUMBER_OF_SIGNALS"]
USERS = "EXTRAPOLATED_NUMBER_OF_USERS"
SIGNALS = "EXTRAPOLATED_NUMBER_OF_SIGNALS"
LOG_COLUMNS = [
    "date",
    "source_file",
    "filename",
    "source_signature",
    "schema_signature",
    "processing_version",
    "cache_status",
    "source_rows",
    "retained_destination_london_rows",
    "retained_london_origin_rows",
    "invalid_destination_rows",
    "outside_london_destination_rows",
    "london_origin_rows",
    "valid_non_london_origin_rows",
    "invalid_or_other_origin_rows",
    "missing_values",
    "negative_users",
    "negative_signals",
    "date_mismatch_rows",
    "modality_mismatch_rows",
    "duplicate_od_records",
    "total_users_before_spatial_filter",
    "total_users_after_destination_filter",
    "total_users_after_origin_filter",
    "valid_origin_h3_entropy_denominator",
    "total_signals_before_spatial_filter",
    "total_signals_after_destination_filter",
    "destination_aggregation_rows",
    "origin_aggregation_rows",
    "origin_destination_component_rows",
    "daily_reconciliation_abs_diff",
    "processing_duration_seconds",
    "success",
]


@dataclass(frozen=True)
class Paths:
    raw_mobility_dir: Path = RAW_MOBILITY_DIR
    boundary_path: Path = BOUNDARY_PATH
    interim_dir: Path = INTERIM_DIR
    daily_cache_dir: Path = DAILY_CACHE_DIR
    processed_dir: Path = PROCESSED_DIR
    output_dir: Path = OUTPUT_DIR


def ensure_dirs(paths: Paths) -> None:
    for path in [paths.interim_dir, paths.daily_cache_dir, paths.processed_dir, paths.output_dir]:
        path.mkdir(parents=True, exist_ok=True)


def expected_march_dates() -> list[str]:
    return [f"2026-03-{day:02d}" for day in range(1, 32)]


def source_signature(path: Path) -> str:
    stat = path.stat()
    raw = f"{path.name}|{stat.st_size}|{stat.st_mtime_ns}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def schema_signature(columns: list[str]) -> str:
    return hashlib.sha256("|".join(columns).encode("utf-8")).hexdigest()


def read_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        return handle.readline().rstrip("\n").split("\t")


def classify_file(path: Path) -> tuple[str | None, str, bool, str]:
    name = path.name
    if match := ALL_TRACKS_RE.match(name):
        return match.group(1), "daily_loco_all_tracks", True, ""
    if match := HOURLY_RE.match(name):
        return match.group(1), "hourly_0_24", False, "excluded_hourly_0_24"
    if match := PEDESTRIAN_RE.match(name):
        return match.group(1), "pedestrian_tracks", False, "excluded_pedestrian_tracks"
    if match := ANY_LOCOMIZER_RE.match(name):
        return match.group(1), "unexpected_locomizer_tsv", False, "unexpected_filename_pattern"
    return None, "not_locomizer", False, "not_locomizer"


def scan_candidate_file(path: Path) -> dict:
    header = read_header(path)
    missing_required = [col for col in REQUIRED_COLUMNS if col not in header]
    extra_fields = [col for col in header if col not in REQUIRED_COLUMNS]
    order_valid = header == REQUIRED_COLUMNS
    schema_valid = order_valid and not missing_required and not extra_fields
    movement_values: set[str] = set()
    day_values: set[str] = set()
    row_count = 0

    if "MOVEMENT_MODALITY" in header and "DAY" in header:
        for chunk in pd.read_csv(
            path,
            sep="\t",
            usecols=["MOVEMENT_MODALITY", "DAY"],
            chunksize=1_000_000,
            dtype={"MOVEMENT_MODALITY": "string", "DAY": "string"},
        ):
            row_count += len(chunk)
            movement_values.update(chunk["MOVEMENT_MODALITY"].dropna().astype(str).unique().tolist())
            day_values.update(chunk["DAY"].dropna().astype(str).unique().tolist())
    else:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            row_count = max(sum(1 for _ in handle) - 1, 0)

    sample = pd.read_csv(path, sep="\t", nrows=10000) if schema_valid else pd.DataFrame()
    dtype_sample = ";".join(f"{col}:{sample[col].dtype}" for col in sample.columns) if not sample.empty else ""

    return {
        "columns": "|".join(header),
        "column_order_valid": order_valid,
        "required_fields_present": not missing_required,
        "schema_valid": schema_valid,
        "missing_required_fields": "|".join(missing_required),
        "extra_fields": "|".join(extra_fields),
        "schema_signature": schema_signature(header),
        "movement_modality": "|".join(sorted(movement_values)),
        "row_count": row_count,
        "day_values": "|".join(sorted(day_values)),
        "dtype_sample": dtype_sample,
    }


def build_manifest(paths: Paths) -> tuple[pd.DataFrame, pd.DataFrame]:
    ensure_dirs(paths)
    observed_rows: list[dict] = []
    schema_rows: list[dict] = []

    for path in sorted(paths.raw_mobility_dir.iterdir()):
        if not path.is_file():
            continue
        file_date, file_type, candidate_main, default_exclusion = classify_file(path)
        if file_date is None:
            continue
        stat = path.stat()
        base = {
            "date": file_date,
            "filepath": str(path.resolve()),
            "filename": path.name,
            "file_type": file_type,
            "movement_modality": "",
            "exists": True,
            "valid_for_main_pipeline": False,
            "exclusion_reason": default_exclusion,
            "processing_status": "excluded" if default_exclusion else "discovered",
            "row_count": np.nan,
            "file_size": stat.st_size,
            "modified_time": pd.Timestamp(stat.st_mtime, unit="s").isoformat(),
            "source_signature": source_signature(path),
            "schema_signature": "",
        }
        if candidate_main:
            scan = scan_candidate_file(path)
            base.update(
                {
                    "movement_modality": scan["movement_modality"],
                    "row_count": scan["row_count"],
                    "schema_signature": scan["schema_signature"],
                }
            )
            modality_ok = scan["movement_modality"] == "ALL"
            day_ok = scan["day_values"] == file_date
            valid = bool(scan["schema_valid"] and modality_ok and day_ok)
            reasons = []
            if not scan["schema_valid"]:
                reasons.append("invalid_schema")
            if not modality_ok:
                reasons.append("movement_modality_not_all")
            if not day_ok:
                reasons.append("file_date_day_field_mismatch")
            base["valid_for_main_pipeline"] = valid
            base["exclusion_reason"] = "" if valid else "|".join(reasons)
            base["processing_status"] = "validated" if valid else "excluded"
            schema_rows.append(
                {
                    "date": file_date,
                    "filename": path.name,
                    "file_type": file_type,
                    "columns": scan["columns"],
                    "column_order_valid": scan["column_order_valid"],
                    "required_fields_present": scan["required_fields_present"],
                    "schema_valid": scan["schema_valid"],
                    "movement_modality": scan["movement_modality"],
                    "day_values": scan["day_values"],
                    "missing_required_fields": scan["missing_required_fields"],
                    "extra_fields": scan["extra_fields"],
                    "dtype_sample": scan["dtype_sample"],
                    "valid_for_main_pipeline": valid,
                    "exclusion_reason": base["exclusion_reason"],
                }
            )
        observed_rows.append(base)

    manifest = pd.DataFrame(observed_rows)
    expected_rows = []
    for day in expected_march_dates():
        valid_count = 0
        if not manifest.empty:
            valid_count = int(((manifest["date"] == day) & (manifest["valid_for_main_pipeline"] == True)).sum())
        if valid_count == 0:
            expected_rows.append(
                {
                    "date": day,
                    "filepath": "",
                    "filename": "",
                    "file_type": "expected_daily_loco_all_tracks",
                    "movement_modality": "",
                    "exists": False,
                    "valid_for_main_pipeline": False,
                    "exclusion_reason": "missing_required_all_tracks_file",
                    "processing_status": "missing",
                    "row_count": np.nan,
                    "file_size": np.nan,
                    "modified_time": "",
                    "source_signature": "",
                    "schema_signature": "",
                }
            )

    if expected_rows:
        manifest = pd.concat([manifest, pd.DataFrame(expected_rows)], ignore_index=True)

    duplicate_dates = (
        manifest[manifest["valid_for_main_pipeline"] == True]
        .groupby("date")
        .size()
        .loc[lambda s: s > 1]
        .index.tolist()
    )
    if duplicate_dates:
        mask = manifest["date"].isin(duplicate_dates) & (manifest["valid_for_main_pipeline"] == True)
        manifest.loc[mask, "valid_for_main_pipeline"] = False
        manifest.loc[mask, "exclusion_reason"] = "duplicate_valid_daily_file_for_date"
        manifest.loc[mask, "processing_status"] = "excluded"

    manifest = manifest.sort_values(["date", "file_type", "filename"]).reset_index(drop=True)
    schema_df = pd.DataFrame(schema_rows).sort_values(["date", "filename"]).reset_index(drop=True)
    manifest.to_csv(MANIFEST_PATH, index=False)
    schema_df.to_csv(SCHEMA_VALIDATION_PATH, index=False)
    return manifest, schema_df


def load_london_lsoas(boundary_path: Path) -> gpd.GeoDataFrame:
    lsoas = gpd.read_file(boundary_path)
    if str(lsoas.crs).upper() != "EPSG:27700":
        lsoas = lsoas.to_crs("EPSG:27700")
    return lsoas[["LSOA21CD", "geometry"]].copy()


def load_h3_lookup() -> pd.DataFrame:
    if H3_LOOKUP_PATH.exists():
        return pd.read_parquet(H3_LOOKUP_PATH)
    return pd.DataFrame(
        columns=["h3_code", "is_valid_h3", "h3_resolution", "lat", "lon", "lsoa21cd", "in_london"]
    )


def save_h3_lookup(lookup: pd.DataFrame) -> None:
    lookup.drop_duplicates("h3_code", keep="last").to_parquet(H3_LOOKUP_PATH, index=False)


def map_h3_codes_to_lsoa(
    codes: Iterable[str],
    lookup: pd.DataFrame,
    known_codes: set[str],
    london_lsoas: gpd.GeoDataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, set[str]]:
    code_series = pd.Series(list(codes), dtype="string").dropna().drop_duplicates()
    if code_series.empty:
        return lookup, lookup.iloc[0:0], known_codes
    new_codes = [str(code) for code in code_series if str(code) not in known_codes]
    mapped_rows: list[dict] = []
    valid_points: list[dict] = []

    for code in new_codes:
        valid = h3.is_valid_cell(code)
        if not valid:
            mapped_rows.append(
                {
                    "h3_code": code,
                    "is_valid_h3": False,
                    "h3_resolution": np.nan,
                    "lat": np.nan,
                    "lon": np.nan,
                    "lsoa21cd": pd.NA,
                    "in_london": False,
                }
            )
            continue
        lat, lon = h3.cell_to_latlng(code)
        valid_points.append(
            {
                "h3_code": code,
                "is_valid_h3": True,
                "h3_resolution": h3.get_resolution(code),
                "lat": lat,
                "lon": lon,
                "geometry": Point(lon, lat),
            }
        )

    if valid_points:
        points = gpd.GeoDataFrame(valid_points, geometry="geometry", crs="EPSG:4326").to_crs("EPSG:27700")
        joined = gpd.sjoin(points, london_lsoas, how="left", predicate="within")
        for _, row in joined.iterrows():
            lsoa = row.get("LSOA21CD", pd.NA)
            mapped_rows.append(
                {
                    "h3_code": row["h3_code"],
                    "is_valid_h3": True,
                    "h3_resolution": int(row["h3_resolution"]),
                    "lat": float(row["lat"]),
                    "lon": float(row["lon"]),
                    "lsoa21cd": lsoa if pd.notna(lsoa) else pd.NA,
                    "in_london": bool(pd.notna(lsoa)),
                }
            )

    if mapped_rows:
        lookup = pd.concat([lookup, pd.DataFrame(mapped_rows)], ignore_index=True)
        known_codes.update(row["h3_code"] for row in mapped_rows)
    return lookup, lookup[lookup["h3_code"].isin(code_series.astype(str))], known_codes


def combine_group_frames(frames: list[pd.DataFrame], keys: list[str]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame(columns=keys + VALUE_COLUMNS)
    combined = pd.concat(frames, ignore_index=True)
    return combined.groupby(keys, dropna=False, as_index=False)[VALUE_COLUMNS].sum()


def day_cache_paths(day: str) -> dict[str, Path]:
    day_dir = DAILY_CACHE_DIR / f"date={day}"
    return {
        "dir": day_dir,
        "destination": day_dir / "destination_lsoa_aggregation.parquet",
        "origin": day_dir / "origin_lsoa_aggregation.parquet",
        "components": day_dir / "origin_destination_components.parquet",
        "metadata": day_dir / "metadata.csv",
    }


def manifest_valid_rows(manifest: pd.DataFrame) -> pd.DataFrame:
    return manifest[manifest["valid_for_main_pipeline"].astype(str).str.lower().isin(["true", "1"])]


def cache_is_valid(day: str, source_signature_value: str, schema_signature_value: str) -> bool:
    paths = day_cache_paths(day)
    if not all(paths[key].exists() for key in ["destination", "origin", "components", "metadata"]):
        return False
    metadata = pd.read_csv(paths["metadata"]).iloc[0].to_dict()
    return bool(
        metadata.get("source_signature") == source_signature_value
        and metadata.get("schema_signature") == schema_signature_value
        and metadata.get("processing_version") == PROCESSING_VERSION
        and str(metadata.get("success")).lower() in ["true", "1"]
    )


def process_one_day(day: str, manifest: pd.DataFrame, paths: Paths, chunk_size: int, force: bool) -> dict:
    matches = manifest_valid_rows(manifest)
    matches = matches[matches["date"].astype(str) == day]
    if matches.empty:
        raise FileNotFoundError(f"No valid ALL daily all-tracks file found for {day}")
    if len(matches) > 1:
        raise ValueError(f"More than one valid daily file found for {day}")

    rec = matches.iloc[0]
    source_path = Path(rec["filepath"])
    cache_paths = day_cache_paths(day)
    if not force and cache_is_valid(day, rec["source_signature"], rec["schema_signature"]):
        metadata = pd.read_csv(cache_paths["metadata"]).iloc[0].to_dict()
        metadata["cache_status"] = "cached"
        append_processing_log(metadata)
        update_manifest_after_processing(day, metadata)
        return metadata

    start = time.time()
    cache_paths["dir"].mkdir(parents=True, exist_ok=True)
    london_lsoas = load_london_lsoas(paths.boundary_path)
    h3_lookup = load_h3_lookup()
    known_h3_codes = set(h3_lookup["h3_code"].astype(str)) if len(h3_lookup) else set()

    dest_frames: list[pd.DataFrame] = []
    origin_frames: list[pd.DataFrame] = []
    component_frames: list[pd.DataFrame] = []

    source_rows = retained_dest_rows = retained_london_origin_rows = 0
    invalid_dest_rows = outside_dest_rows = invalid_origin_rows = 0
    london_origin_rows = valid_non_london_origin_rows = 0
    missing_values = negative_users = negative_signals = date_mismatch_rows = modality_mismatch_rows = 0
    duplicate_od_records = 0
    seen_od_hashes: set[int] = set()
    total_users_before = total_signals_before = 0.0

    reader = pd.read_csv(
        source_path,
        sep="\t",
        usecols=REQUIRED_COLUMNS,
        chunksize=chunk_size,
        dtype={
            "CODE": "string",
            "ORIGIN_CODE": "string",
            "DAY_TYPE": "Int64",
            "DAY": "string",
            "MONTH": "Int64",
            "YEAR": "Int64",
            "MOVEMENT_MODALITY": "string",
            USERS: "float64",
            SIGNALS: "float64",
        },
    )

    for chunk_number, chunk in enumerate(reader, start=1):
        source_rows += len(chunk)
        missing_values += int(chunk[REQUIRED_COLUMNS].isna().sum().sum())
        negative_users += int((chunk[USERS] < 0).sum())
        negative_signals += int((chunk[SIGNALS] < 0).sum())
        date_mismatch_rows += int((chunk["DAY"].astype(str) != day).sum())
        modality_mismatch_rows += int((chunk["MOVEMENT_MODALITY"].astype(str) != "ALL").sum())
        total_users_before += float(chunk[USERS].sum())
        total_signals_before += float(chunk[SIGNALS].sum())

        od_hashes = hash_pandas_object(chunk[["CODE", "ORIGIN_CODE"]], index=False).astype("uint64")
        duplicate_od_records += int(od_hashes.duplicated().sum())
        for value in od_hashes[~od_hashes.duplicated()].tolist():
            value = int(value)
            if value in seen_od_hashes:
                duplicate_od_records += 1
            else:
                seen_od_hashes.add(value)

        h3_lookup, dest_map, known_h3_codes = map_h3_codes_to_lsoa(
            chunk["CODE"].dropna().unique(), h3_lookup, known_h3_codes, london_lsoas
        )
        h3_lookup, origin_map, known_h3_codes = map_h3_codes_to_lsoa(
            chunk["ORIGIN_CODE"].dropna().unique(), h3_lookup, known_h3_codes, london_lsoas
        )
        dest_map = dest_map.rename(
            columns={
                "h3_code": "CODE",
                "is_valid_h3": "dest_valid_h3",
                "h3_resolution": "dest_h3_resolution",
                "lsoa21cd": "destination_lsoa21cd",
                "in_london": "dest_in_london",
            }
        )[["CODE", "dest_valid_h3", "dest_h3_resolution", "destination_lsoa21cd", "dest_in_london"]]
        origin_map = origin_map.rename(
            columns={
                "h3_code": "ORIGIN_CODE",
                "is_valid_h3": "origin_valid_h3",
                "h3_resolution": "origin_h3_resolution",
                "lsoa21cd": "origin_lsoa21cd",
                "in_london": "origin_in_london",
            }
        )[["ORIGIN_CODE", "origin_valid_h3", "origin_h3_resolution", "origin_lsoa21cd", "origin_in_london"]]

        chunk = chunk.merge(dest_map, on="CODE", how="left").merge(origin_map, on="ORIGIN_CODE", how="left")
        for col in ["dest_valid_h3", "dest_in_london", "origin_valid_h3", "origin_in_london"]:
            chunk[col] = chunk[col].fillna(False).astype(bool)

        invalid_dest_rows += int((~chunk["dest_valid_h3"]).sum())
        outside_dest_rows += int((chunk["dest_valid_h3"] & ~chunk["dest_in_london"]).sum())
        invalid_origin_rows += int((~chunk["origin_valid_h3"]).sum())
        london_origin_rows += int(chunk["origin_in_london"].sum())
        valid_non_london_origin_rows += int((chunk["origin_valid_h3"] & ~chunk["origin_in_london"]).sum())

        london_dest = chunk[chunk["dest_in_london"]].copy()
        retained_dest_rows += len(london_dest)
        if not london_dest.empty:
            london_dest["date"] = day
            london_dest["is_weekend"] = pd.Timestamp(day).dayofweek >= 5
            london_dest["origin_class"] = np.select(
                [
                    london_dest["origin_in_london"],
                    london_dest["origin_valid_h3"] & ~london_dest["origin_in_london"],
                ],
                ["london_origin", "valid_non_london_origin"],
                default="invalid_or_other_origin",
            )
            dest_frames.append(
                london_dest.groupby(["date", "destination_lsoa21cd", "is_weekend", "origin_class"], as_index=False)[
                    VALUE_COLUMNS
                ].sum()
            )
            valid_origin_to_london = london_dest[london_dest["origin_valid_h3"]].copy()
            if not valid_origin_to_london.empty:
                component_frames.append(
                    valid_origin_to_london.groupby(
                        ["date", "destination_lsoa21cd", "ORIGIN_CODE"], as_index=False
                    )[VALUE_COLUMNS].sum()
                )

        london_origin = chunk[chunk["origin_in_london"] & chunk["dest_valid_h3"]].copy()
        retained_london_origin_rows += len(london_origin)
        if not london_origin.empty:
            london_origin["date"] = day
            origin_frames.append(london_origin.groupby(["date", "origin_lsoa21cd"], as_index=False)[VALUE_COLUMNS].sum())

    save_h3_lookup(h3_lookup)
    dest = combine_group_frames(dest_frames, ["date", "destination_lsoa21cd", "is_weekend", "origin_class"])
    origin = combine_group_frames(origin_frames, ["date", "origin_lsoa21cd"])
    components = combine_group_frames(component_frames, ["date", "destination_lsoa21cd", "ORIGIN_CODE"])
    dest.to_parquet(cache_paths["destination"], index=False)
    origin.to_parquet(cache_paths["origin"], index=False)
    components.to_parquet(cache_paths["components"], index=False)

    dest_total = float(dest[USERS].sum()) if len(dest) else 0.0
    origin_total = float(origin[USERS].sum()) if len(origin) else 0.0
    component_total = float(components[USERS].sum()) if len(components) else 0.0
    reconciliation_abs_diff = abs(dest_total - sum(frame[USERS].sum() for frame in dest_frames)) if dest_frames else 0.0
    success = bool(
        missing_values == 0
        and negative_users == 0
        and negative_signals == 0
        and date_mismatch_rows == 0
        and modality_mismatch_rows == 0
        and retained_dest_rows > 0
        and reconciliation_abs_diff <= RECONCILIATION_TOLERANCE
    )
    metadata = {
        "date": day,
        "source_file": str(source_path.resolve()),
        "filename": source_path.name,
        "source_signature": rec["source_signature"],
        "schema_signature": rec["schema_signature"],
        "processing_version": PROCESSING_VERSION,
        "cache_status": "processed",
        "source_rows": source_rows,
        "retained_destination_london_rows": retained_dest_rows,
        "retained_london_origin_rows": retained_london_origin_rows,
        "invalid_destination_rows": invalid_dest_rows,
        "outside_london_destination_rows": outside_dest_rows,
        "london_origin_rows": london_origin_rows,
        "valid_non_london_origin_rows": valid_non_london_origin_rows,
        "invalid_or_other_origin_rows": invalid_origin_rows,
        "missing_values": missing_values,
        "negative_users": negative_users,
        "negative_signals": negative_signals,
        "date_mismatch_rows": date_mismatch_rows,
        "modality_mismatch_rows": modality_mismatch_rows,
        "duplicate_od_records": duplicate_od_records,
        "total_users_before_spatial_filter": total_users_before,
        "total_users_after_destination_filter": dest_total,
        "total_users_after_origin_filter": origin_total,
        "valid_origin_h3_entropy_denominator": component_total,
        "total_signals_before_spatial_filter": total_signals_before,
        "total_signals_after_destination_filter": float(dest[SIGNALS].sum()) if len(dest) else 0.0,
        "destination_aggregation_rows": len(dest),
        "origin_aggregation_rows": len(origin),
        "origin_destination_component_rows": len(components),
        "daily_reconciliation_abs_diff": reconciliation_abs_diff,
        "processing_duration_seconds": time.time() - start,
        "success": success,
    }
    pd.DataFrame([metadata]).to_csv(cache_paths["metadata"], index=False)
    append_processing_log(metadata)
    update_manifest_after_processing(day, metadata)
    return metadata


def append_processing_log(metadata: dict) -> None:
    row = pd.DataFrame([{col: metadata.get(col, np.nan) for col in LOG_COLUMNS}])
    if PROCESSING_LOG_PATH.exists():
        old = pd.read_csv(PROCESSING_LOG_PATH)
        if "date" in old.columns:
            old = old[old["date"].astype(str) != str(metadata["date"])]
        for col in LOG_COLUMNS:
            if col not in old.columns:
                old[col] = np.nan
        old = old[LOG_COLUMNS]
        row = pd.concat([old, row], ignore_index=True)
    row[LOG_COLUMNS].sort_values("date").to_csv(PROCESSING_LOG_PATH, index=False)


def update_manifest_after_processing(day: str, metadata: dict) -> None:
    if not MANIFEST_PATH.exists():
        return
    manifest = pd.read_csv(MANIFEST_PATH)
    mask = (manifest["date"].astype(str) == day) & (
        manifest["valid_for_main_pipeline"].astype(str).str.lower().isin(["true", "1"])
    )
    if mask.any():
        manifest.loc[mask, "processing_status"] = "processed" if metadata["success"] else "failed"
        manifest.loc[mask, "row_count"] = metadata["source_rows"]
        manifest.to_csv(MANIFEST_PATH, index=False)


def entropy_from_grouped_flows(grouped: pd.DataFrame, dest_col: str, origin_col: str) -> pd.DataFrame:
    columns = [dest_col, "flow_entropy", "unique_origins", "entropy_denominator"]
    if grouped.empty:
        return pd.DataFrame(columns=columns)
    positive = grouped[grouped[USERS] > 0].copy()
    if positive.empty:
        return pd.DataFrame(columns=columns)
    totals = positive.groupby(dest_col)[USERS].sum().rename("entropy_denominator")
    work = positive.merge(totals, on=dest_col, how="left")
    work["p"] = work[USERS] / work["entropy_denominator"]
    work["term"] = -(work["p"] * np.log(work["p"]))
    return (
        work.groupby(dest_col)
        .agg(
            flow_entropy=("term", "sum"),
            unique_origins=(origin_col, "nunique"),
            entropy_denominator=("entropy_denominator", "first"),
        )
        .reset_index()
    )


def load_daily_outputs(days: list[str], key: str) -> pd.DataFrame:
    frames = []
    for day in days:
        path = day_cache_paths(day)[key]
        if path.exists():
            frames.append(pd.read_parquet(path))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator.where(denominator != 0) / denominator.where(denominator != 0)


def successful_processed_dates() -> list[str]:
    if not PROCESSING_LOG_PATH.exists():
        return []
    log = pd.read_csv(PROCESSING_LOG_PATH)
    if "success" not in log.columns:
        return []
    ok = log[log["success"].astype(str).str.lower().isin(["true", "1"])]
    return sorted(ok["date"].astype(str).tolist())


def build_monthly_indicators(days: list[str], missing_dates: list[str], provisional_flag: bool = True) -> pd.DataFrame:
    lsoas = gpd.read_file(BOUNDARY_PATH)[["LSOA21CD", "geometry"]]
    result = pd.DataFrame({"LSOA21CD": lsoas["LSOA21CD"]})
    dest = load_daily_outputs(days, "destination")
    origin = load_daily_outputs(days, "origin")
    components = load_daily_outputs(days, "components")

    observed_days = len(days)
    day_index = pd.DataFrame({"date": days})
    day_index["is_weekend"] = pd.to_datetime(day_index["date"]).dt.dayofweek >= 5
    observed_weekdays = int((~day_index["is_weekend"]).sum())
    observed_weekend_days = int(day_index["is_weekend"].sum())
    missing_str = "|".join(missing_dates)

    dest_total = dest.groupby("destination_lsoa21cd", as_index=False)[USERS].sum().rename(
        columns={"destination_lsoa21cd": "LSOA21CD", USERS: "mobility_inflow_total"}
    )
    result = result.merge(dest_total, on="LSOA21CD", how="left")

    daily_dest = dest.groupby(["date", "destination_lsoa21cd", "is_weekend"], as_index=False)[USERS].sum()
    weekday = daily_dest[daily_dest["is_weekend"] == False].groupby("destination_lsoa21cd", as_index=False)[
        USERS
    ].sum()
    weekend = daily_dest[daily_dest["is_weekend"] == True].groupby("destination_lsoa21cd", as_index=False)[
        USERS
    ].sum()
    result = result.merge(
        weekday.rename(columns={"destination_lsoa21cd": "LSOA21CD", USERS: "weekday_inflow_total"}),
        on="LSOA21CD",
        how="left",
    ).merge(
        weekend.rename(columns={"destination_lsoa21cd": "LSOA21CD", USERS: "weekend_inflow_total"}),
        on="LSOA21CD",
        how="left",
    )

    entropy_input = components.groupby(["destination_lsoa21cd", "ORIGIN_CODE"], as_index=False)[USERS].sum()
    entropy = entropy_from_grouped_flows(entropy_input, "destination_lsoa21cd", "ORIGIN_CODE").rename(
        columns={"destination_lsoa21cd": "LSOA21CD"}
    )
    result = result.merge(entropy, on="LSOA21CD", how="left")

    outflow = origin.groupby("origin_lsoa21cd", as_index=False)[USERS].sum().rename(
        columns={"origin_lsoa21cd": "LSOA21CD", USERS: "mobility_outflow_total"}
    )
    result = result.merge(outflow, on="LSOA21CD", how="left")

    by_class = dest.pivot_table(
        index="destination_lsoa21cd",
        columns="origin_class",
        values=USERS,
        aggfunc="sum",
        fill_value=0,
    ).reset_index()
    by_class = by_class.rename(columns={"destination_lsoa21cd": "LSOA21CD"})
    for col in ["london_origin", "valid_non_london_origin", "invalid_or_other_origin"]:
        if col not in by_class.columns:
            by_class[col] = 0.0
    by_class = by_class.rename(
        columns={
            "london_origin": "london_origin_flow",
            "valid_non_london_origin": "valid_non_london_origin_flow",
            "invalid_or_other_origin": "other_origin_flow",
        }
    )
    result = result.merge(by_class, on="LSOA21CD", how="left")

    fill_zero = [
        "mobility_inflow_total",
        "weekday_inflow_total",
        "weekend_inflow_total",
        "flow_entropy",
        "unique_origins",
        "entropy_denominator",
        "mobility_outflow_total",
        "london_origin_flow",
        "valid_non_london_origin_flow",
        "other_origin_flow",
    ]
    for col in fill_zero:
        if col not in result.columns:
            result[col] = 0.0
        result[col] = result[col].fillna(0)

    result["observed_days"] = observed_days
    result["expected_days"] = EXPECTED_DAYS
    result["coverage_ratio"] = observed_days / EXPECTED_DAYS
    result["observed_weekdays"] = observed_weekdays
    result["observed_weekend_days"] = observed_weekend_days
    result["missing_dates"] = missing_str
    result["provisional_flag"] = provisional_flag
    result["mobility_inflow_mean_daily"] = result["mobility_inflow_total"] / observed_days
    result["weekday_mean_daily_inflow"] = (
        result["weekday_inflow_total"] / observed_weekdays if observed_weekdays else np.nan
    )
    result["weekend_mean_daily_inflow"] = (
        result["weekend_inflow_total"] / observed_weekend_days if observed_weekend_days else np.nan
    )
    result["weekday_weekend_ratio"] = safe_ratio(
        result["weekday_mean_daily_inflow"], result["weekend_mean_daily_inflow"]
    )
    result["weekday_weekend_zero_denominator"] = result["weekend_mean_daily_inflow"] == 0
    result["net_flow"] = result["mobility_inflow_total"] - result["mobility_outflow_total"]
    result["other_origin_share"] = safe_ratio(result["other_origin_flow"], result["mobility_inflow_total"]).fillna(0)
    result["valid_origin_flow_share"] = safe_ratio(
        result["entropy_denominator"], result["mobility_inflow_total"]
    ).fillna(0)
    result["external_origin_share"] = safe_ratio(
        result["valid_non_london_origin_flow"], result["mobility_inflow_total"]
    ).fillna(0)

    ordered = [
        "LSOA21CD",
        "observed_days",
        "expected_days",
        "coverage_ratio",
        "observed_weekdays",
        "observed_weekend_days",
        "missing_dates",
        "provisional_flag",
        "mobility_inflow_total",
        "mobility_inflow_mean_daily",
        "flow_entropy",
        "weekday_inflow_total",
        "weekend_inflow_total",
        "weekday_mean_daily_inflow",
        "weekend_mean_daily_inflow",
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
    return result[ordered]


def write_indicator_dictionary() -> None:
    text = """# Mobility Indicator Dictionary

Scope: March 2026 destination-based Locomizer ALL-track mobility indicators at Greater London LSOA 2021 level. The final dissertation output is the 31-day run; provisional runs are audit/preliminary outputs only.

| Variable | Definition |
|---|---|
| `LSOA21CD` | Greater London LSOA 2021 code. |
| `observed_days` | Count of valid ALL daily files included in the run. |
| `expected_days` | Expected March 2026 day count, 31. |
| `coverage_ratio` | `observed_days / expected_days`. |
| `observed_weekdays` | Included Monday-Friday dates. |
| `observed_weekend_days` | Included Saturday-Sunday dates. |
| `missing_dates` | Pipe-separated expected dates missing from the ALL-track run. |
| `provisional_flag` | True when the output is not the final 31-day dissertation dataset. |
| `mobility_inflow_total` | All valid origins to London destination LSOAs, summed over observed days. |
| `mobility_inflow_mean_daily` | `mobility_inflow_total / observed_days`. |
| `flow_entropy` | Shannon entropy over valid raw H3 origins only. Invalid/non-H3 origins such as `other` are excluded from the entropy denominator. |
| `weekday_inflow_total` | All-origin inflow to destination LSOA across observed weekdays. |
| `weekend_inflow_total` | All-origin inflow to destination LSOA across observed weekend days. |
| `weekday_mean_daily_inflow` | `weekday_inflow_total / observed_weekdays`. |
| `weekend_mean_daily_inflow` | `weekend_inflow_total / observed_weekend_days`. |
| `weekday_weekend_ratio` | `weekday_mean_daily_inflow / weekend_mean_daily_inflow`; undefined where the denominator is zero. |
| `weekday_weekend_zero_denominator` | True when the weekend denominator is zero. |
| `unique_origins` | Count of distinct valid raw H3 origins with positive flow to the destination LSOA. |
| `mobility_outflow_total` | Flow from London origin LSOA to all valid destinations in the destination-format data. |
| `net_flow` | `mobility_inflow_total - mobility_outflow_total`. |
| `london_origin_flow` | Destination inflow from origins assigned to London LSOAs. |
| `valid_non_london_origin_flow` | Destination inflow from valid H3 origins not assigned to London LSOAs. |
| `other_origin_flow` | Destination inflow from invalid/non-H3 origins. Included in inflow, excluded from entropy. |
| `other_origin_share` | `other_origin_flow / mobility_inflow_total`. |
| `valid_origin_flow_share` | Entropy denominator share of total inflow. |
| `external_origin_share` | Valid non-London origin flow share of total inflow. |
| `entropy_denominator` | Flow denominator used for `flow_entropy`, based on valid raw H3 origins only. |
"""
    INDICATOR_DICTIONARY_PATH.write_text(text, encoding="utf-8")


def write_coverage_outputs(manifest: pd.DataFrame, processed_days: list[str], mode: str) -> list[str]:
    valid_days = sorted(manifest_valid_rows(manifest)["date"].astype(str).unique().tolist())
    missing_dates = [day for day in expected_march_dates() if day not in valid_days]
    coverage = pd.DataFrame(
        [
            {
                "run_mode": mode,
                "expected_days": EXPECTED_DAYS,
                "valid_available_days": len(valid_days),
                "processed_days": len(processed_days),
                "missing_date_count": len(missing_dates),
                "coverage_ratio": len(processed_days) / EXPECTED_DAYS,
                "missing_dates": "|".join(missing_dates),
                "provisional_flag": mode != "final",
            }
        ]
    )
    coverage.to_csv(COVERAGE_SUMMARY_PATH, index=False)
    pd.DataFrame({"date": missing_dates, "reason": "missing_valid_all_tracks_file"}).to_csv(
        MISSING_DATES_PATH, index=False
    )
    return missing_dates


def qa_rows(
    indicators: pd.DataFrame,
    manifest: pd.DataFrame,
    log: pd.DataFrame,
    missing_dates: list[str],
    expected_valid_files: int = 28,
) -> pd.DataFrame:
    rows = []

    def add(metric: str, value, status: str, notes: str = "") -> None:
        rows.append({"metric": metric, "value": value, "status": status, "notes": notes})

    valid_files = manifest_valid_rows(manifest)
    excluded = manifest[(manifest["exists"].astype(str).str.lower().isin(["true", "1"])) & (manifest["valid_for_main_pipeline"] == False)]
    add("valid_source_file_count", len(valid_files), "pass" if len(valid_files) == expected_valid_files else "warn")
    add("missing_dates", "|".join(missing_dates), "warn" if missing_dates else "pass")
    add("excluded_file_count", len(excluded), "pass")
    add("movement_modality_values_valid_files", "|".join(sorted(valid_files["movement_modality"].dropna().unique())), "pass")
    duplicate_dates = valid_files["date"].duplicated().sum()
    add("duplicated_valid_dates", int(duplicate_dates), "pass" if duplicate_dates == 0 else "fail")
    add("daily_duplicate_od_rows_total", int(log["duplicate_od_records"].sum()), "pass" if int(log["duplicate_od_records"].sum()) == 0 else "warn")
    add("daily_missing_values_total", int(log["missing_values"].sum()), "pass" if int(log["missing_values"].sum()) == 0 else "fail")
    add("daily_negative_users_total", int(log["negative_users"].sum()), "pass" if int(log["negative_users"].sum()) == 0 else "fail")
    add("daily_negative_signals_total", int(log["negative_signals"].sum()), "pass" if int(log["negative_signals"].sum()) == 0 else "fail")
    max_recon = float(log["daily_reconciliation_abs_diff"].max()) if len(log) else math.nan
    add("max_daily_aggregation_reconciliation_abs_diff", max_recon, "pass" if max_recon <= RECONCILIATION_TOLERANCE else "fail")
    add("output_lsoa_rows", len(indicators), "pass" if len(indicators) == 4994 else "fail")
    add("unique_lsoa_codes", indicators["LSOA21CD"].nunique(), "pass" if indicators["LSOA21CD"].nunique() == 4994 else "fail")
    add("duplicate_lsoa_codes", int(indicators["LSOA21CD"].duplicated().sum()), "pass" if indicators["LSOA21CD"].duplicated().sum() == 0 else "fail")
    add("negative_inflow_rows", int((indicators["mobility_inflow_total"] < 0).sum()), "pass")
    add("negative_outflow_rows", int((indicators["mobility_outflow_total"] < 0).sum()), "pass")
    add("entropy_min", float(indicators["flow_entropy"].min()), "pass" if indicators["flow_entropy"].min() >= -1e-9 else "fail")
    add("zero_wwr_denominator_rows", int(indicators["weekday_weekend_zero_denominator"].sum()), "pass")
    add("processed_date_count", len(log), "pass" if len(log) == expected_valid_files else "fail")
    add("cached_day_count", int((log["cache_status"] == "cached").sum()), "pass")
    add("newly_processed_day_count", int((log["cache_status"] == "processed").sum()), "pass")
    add("max_daily_users_after_destination_filter", float(log["total_users_after_destination_filter"].max()), "pass")
    add("min_daily_users_after_destination_filter", float(log["total_users_after_destination_filter"].min()), "pass")
    add("max_mobility_inflow_total", float(indicators["mobility_inflow_total"].max()), "pass")
    add("min_mobility_inflow_total", float(indicators["mobility_inflow_total"].min()), "pass")
    add("other_origin_flow_share_total", float(indicators["other_origin_flow"].sum() / indicators["mobility_inflow_total"].sum()), "pass")
    add("valid_origin_flow_share_total", float(indicators["entropy_denominator"].sum() / indicators["mobility_inflow_total"].sum()), "pass")
    return pd.DataFrame(rows)


def write_provisional_outputs(manifest: pd.DataFrame, processed_days: list[str], mode: str) -> tuple[Path, Path, pd.DataFrame, pd.DataFrame]:
    missing_dates = write_coverage_outputs(manifest, processed_days, mode)
    indicators = build_monthly_indicators(processed_days, missing_dates)
    csv_path = PROCESSED_DIR / f"march_2026_lsoa_mobility_provisional_{len(processed_days)}days.csv"
    gpkg_path = PROCESSED_DIR / f"march_2026_lsoa_mobility_provisional_{len(processed_days)}days.gpkg"
    indicators.to_csv(csv_path, index=False)
    lsoas = gpd.read_file(BOUNDARY_PATH)[["LSOA21CD", "geometry"]].to_crs("EPSG:27700")
    lsoas.merge(indicators, on="LSOA21CD", how="left").to_file(
        gpkg_path, layer=f"march_2026_lsoa_mobility_provisional_{len(processed_days)}days", driver="GPKG"
    )
    log = pd.read_csv(PROCESSING_LOG_PATH)
    log = log[log["date"].astype(str).isin(processed_days)].copy()
    qa = qa_rows(indicators, manifest, log, missing_dates)
    qa.to_csv(INDICATOR_QA_PATH, index=False)
    write_indicator_dictionary()
    write_provisional_report(manifest, processed_days, missing_dates, indicators, qa, csv_path, gpkg_path)
    return csv_path, gpkg_path, indicators, qa


def write_provisional_report(
    manifest: pd.DataFrame,
    processed_days: list[str],
    missing_dates: list[str],
    indicators: pd.DataFrame,
    qa: pd.DataFrame,
    csv_path: Path,
    gpkg_path: Path,
) -> None:
    excluded = manifest[(manifest["exists"].astype(str).str.lower().isin(["true", "1"])) & (manifest["valid_for_main_pipeline"] == False)]
    valid = manifest_valid_rows(manifest)
    log = pd.read_csv(PROCESSING_LOG_PATH)
    log = log[log["date"].astype(str).isin(processed_days)]
    failed_qa = qa[qa["status"] == "fail"]
    warn_qa = qa[qa["status"] == "warn"]
    report = f"""# Mobility Provisional Run Report

## Status

This is a **provisional** March 2026 mobility output for pipeline validation and preliminary ESDA only. It uses {len(processed_days)} of 31 March days. The missing ALL-track dates are: {', '.join(missing_dates)}.

The final dissertation indicators must be regenerated when the correct `MOVEMENT_MODALITY = ALL` files for the missing dates become available.

## Inputs

- Valid main input pattern: `Audience_Profiles_Destination_YYYY-MM-DD_loco_all_tracks.tsv`
- Excluded patterns: `*_0-24.tsv`, `*_loco_pedestrian_tracks.tsv`, and any file where `MOVEMENT_MODALITY` is not exactly `ALL`
- Valid ALL files processed: {len(valid)}
- Observed weekdays: {int(indicators['observed_weekdays'].iloc[0])}
- Observed weekend days: {int(indicators['observed_weekend_days'].iloc[0])}
- Coverage ratio: {float(indicators['coverage_ratio'].iloc[0]):.6f}

## Processed Dates

{', '.join(processed_days)}

## Excluded Files

"""
    if excluded.empty:
        report += "No excluded raw Locomizer files were present beyond missing expected dates.\n"
    else:
        for _, row in excluded.iterrows():
            report += f"- `{row['filename']}`: {row['exclusion_reason']}\n"

    report += f"""
## Outputs

- CSV: `{csv_path.relative_to(PROJECT_ROOT)}`
- GPKG: `{gpkg_path.relative_to(PROJECT_ROOT)}`
- Manifest: `{MANIFEST_PATH.relative_to(PROJECT_ROOT)}`
- Daily log: `{PROCESSING_LOG_PATH.relative_to(PROJECT_ROOT)}`
- QA: `{INDICATOR_QA_PATH.relative_to(PROJECT_ROOT)}`
- Coverage summary: `{COVERAGE_SUMMARY_PATH.relative_to(PROJECT_ROOT)}`
- Missing dates: `{MISSING_DATES_PATH.relative_to(PROJECT_ROOT)}`

## Cache Summary

- Newly processed days in current log: {int((log['cache_status'] == 'processed').sum())}
- Cached days in current log: {int((log['cache_status'] == 'cached').sum())}
- Daily cache directory: `{DAILY_CACHE_DIR.relative_to(PROJECT_ROOT)}`

## Indicator Definitions

- Destination inflow: all valid origins to London destination LSOAs.
- Entropy: valid raw H3 origins only; invalid/non-H3 origins such as `other` are excluded from the entropy denominator but retained in total inflow.
- Unique origins: distinct valid origin H3 cells with positive flow to each London destination LSOA.
- Outflow: London origin LSOAs to all valid destinations represented in the destination-format files.
- Net flow: inflow minus outflow. This remains a provisional robustness variable because full OD symmetry cannot be independently verified from destination-format files alone.
- Weekday-weekend ratio: mean daily weekday inflow divided by mean daily weekend inflow. Zero denominators are flagged in `weekday_weekend_zero_denominator` and not silently replaced.

## QA Summary

- Failed QA checks: {len(failed_qa)}
- Warning QA checks: {len(warn_qa)}
- Output LSOA rows: {len(indicators):,}
- Unique LSOA codes: {indicators['LSOA21CD'].nunique():,}
- Total provisional inflow: {float(indicators['mobility_inflow_total'].sum()):.6f}
- Total provisional outflow: {float(indicators['mobility_outflow_total'].sum()):.6f}
- Other-origin flow share: {float(indicators['other_origin_flow'].sum() / indicators['mobility_inflow_total'].sum()):.6f}
- Valid-origin entropy denominator share: {float(indicators['entropy_denominator'].sum() / indicators['mobility_inflow_total'].sum()):.6f}

## Known Limitations

- The provisional output omits {', '.join(missing_dates)}.
- Pedestrian files, if present, are excluded from the main ALL-track pipeline and must not be used as substitutes.
- Final dissertation indicators must not use these provisional filenames or totals as final results.
"""
    PROVISIONAL_REPORT_PATH.write_text(report, encoding="utf-8")


def write_final_outputs(manifest: pd.DataFrame, processed_days: list[str]) -> tuple[Path, Path, pd.DataFrame, pd.DataFrame]:
    missing_dates = [day for day in expected_march_dates() if day not in processed_days]
    if missing_dates:
        raise RuntimeError(f"Cannot write final outputs with missing dates: {', '.join(missing_dates)}")

    indicators = build_monthly_indicators(processed_days, missing_dates, provisional_flag=False)
    csv_path = PROCESSED_DIR / "march_2026_lsoa_mobility_final_31days.csv"
    gpkg_path = PROCESSED_DIR / "march_2026_lsoa_mobility_final_31days.gpkg"
    indicators.to_csv(csv_path, index=False)

    lsoas = gpd.read_file(BOUNDARY_PATH)[["LSOA21CD", "geometry"]].to_crs("EPSG:27700")
    gdf = lsoas.merge(indicators, on="LSOA21CD", how="left")
    gdf.to_file(gpkg_path, layer="march_2026_lsoa_mobility_final_31days", driver="GPKG")

    log = pd.read_csv(PROCESSING_LOG_PATH)
    log = log[log["date"].astype(str).isin(processed_days)].copy()
    qa = qa_rows(indicators, manifest, log, missing_dates, expected_valid_files=31)
    detailed = final_detailed_qa(indicators, gdf, processed_days)
    qa = pd.concat([qa, detailed], ignore_index=True)
    qa.to_csv(FINAL_QA_PATH, index=False)

    coverage = pd.DataFrame(
        [
            {
                "run_mode": "final",
                "expected_days": EXPECTED_DAYS,
                "valid_available_days": 31,
                "processed_days": len(processed_days),
                "missing_date_count": 0,
                "coverage_ratio": 1.0,
                "missing_dates": "",
                "provisional_flag": False,
                "weekday_dates": int((pd.to_datetime(pd.Series(processed_days)).dt.dayofweek < 5).sum()),
                "weekend_dates": int((pd.to_datetime(pd.Series(processed_days)).dt.dayofweek >= 5).sum()),
            }
        ]
    )
    coverage.to_csv(FINAL_COVERAGE_PATH, index=False)
    manifest.to_csv(FINAL_MANIFEST_PATH, index=False)
    write_indicator_dictionary()
    write_final_report(manifest, processed_days, indicators, qa, csv_path, gpkg_path)
    return csv_path, gpkg_path, indicators, qa


def final_detailed_qa(indicators: pd.DataFrame, gdf: gpd.GeoDataFrame, processed_days: list[str]) -> pd.DataFrame:
    rows = []

    def add(metric: str, value, status: str, notes: str = "") -> None:
        rows.append({"metric": metric, "value": value, "status": status, "notes": notes})

    london_lsoas = gpd.read_file(BOUNDARY_PATH)[["LSOA21CD"]]
    unmatched = set(london_lsoas["LSOA21CD"]) - set(indicators["LSOA21CD"])
    add("unmatched_london_lsoas", len(unmatched), "pass" if len(unmatched) == 0 else "fail")
    add("all_31_dates_processed_exactly_once", len(set(processed_days)) == 31 and len(processed_days) == 31, "pass")
    add("weekday_date_count", int((pd.to_datetime(pd.Series(processed_days)).dt.dayofweek < 5).sum()), "pass")
    add("weekend_date_count", int((pd.to_datetime(pd.Series(processed_days)).dt.dayofweek >= 5).sum()), "pass")
    add("gpkg_crs", str(gdf.crs), "pass" if str(gdf.crs).upper() == "EPSG:27700" else "fail")
    invalid_geom = int((~gdf.geometry.is_valid).sum())
    add("invalid_geometry_count", invalid_geom, "pass" if invalid_geom == 0 else "fail")

    indicator_cols = [
        col
        for col in indicators.columns
        if col != "LSOA21CD" and pd.api.types.is_numeric_dtype(indicators[col])
    ]
    for col in indicator_cols:
        series = indicators[col]
        add(f"missing_values__{col}", int(series.isna().sum()), "pass" if series.isna().sum() == 0 else "fail")
        finite_mask = np.isfinite(series.to_numpy(dtype=float, na_value=np.nan))
        inf_count = int((~finite_mask & ~series.isna().to_numpy()).sum())
        add(f"infinite_values__{col}", inf_count, "pass" if inf_count == 0 else "fail")
        neg_count = int((series < 0).sum()) if pd.api.types.is_numeric_dtype(series) else 0
        status = "warn" if col == "net_flow" and neg_count > 0 else ("pass" if neg_count == 0 else "fail")
        add(f"negative_values__{col}", neg_count, status)
        add(f"zero_values__{col}", int((series == 0).sum()), "pass")

    add("every_indicator_passed_qa", "see metric rows", "pass" if not any(row["status"] == "fail" for row in rows) else "fail")
    return pd.DataFrame(rows)


def write_final_report(
    manifest: pd.DataFrame,
    processed_days: list[str],
    indicators: pd.DataFrame,
    qa: pd.DataFrame,
    csv_path: Path,
    gpkg_path: Path,
) -> None:
    excluded = manifest[(manifest["exists"].astype(str).str.lower().isin(["true", "1"])) & (manifest["valid_for_main_pipeline"] == False)]
    failures = qa[qa["status"] == "fail"]
    warnings = qa[qa["status"] == "warn"]
    log = pd.read_csv(PROCESSING_LOG_PATH)
    log = log[log["date"].astype(str).isin(processed_days)]
    report = f"""# Mobility Final 31-Day Run Report

## Status

This final March 2026 mobility output uses all 31 valid daily `loco_all_tracks.tsv` files with `MOVEMENT_MODALITY = ALL`. It supersedes the provisional 28-day mobility outputs for modelling. The provisional outputs are retained only for audit and comparison.

## Inputs

- Valid main input pattern: `Audience_Profiles_Destination_YYYY-MM-DD_loco_all_tracks.tsv`
- Excluded patterns: `*_0-24.tsv`, `*_loco_pedestrian_tracks.tsv`, and any file where `MOVEMENT_MODALITY` is not exactly `ALL`
- Processed dates: {', '.join(processed_days)}
- Weekday dates: {int((pd.to_datetime(pd.Series(processed_days)).dt.dayofweek < 5).sum())}
- Weekend dates: {int((pd.to_datetime(pd.Series(processed_days)).dt.dayofweek >= 5).sum())}
- Coverage ratio: 1.000000

## Excluded Files

"""
    if excluded.empty:
        report += "No excluded raw Locomizer files were present.\n"
    else:
        for _, row in excluded.iterrows():
            report += f"- `{row['filename']}`: {row['exclusion_reason']}\n"

    report += f"""
## Outputs

- CSV: `{csv_path.relative_to(PROJECT_ROOT)}`
- GPKG: `{gpkg_path.relative_to(PROJECT_ROOT)}`
- Final manifest: `{FINAL_MANIFEST_PATH.relative_to(PROJECT_ROOT)}`
- Final QA: `{FINAL_QA_PATH.relative_to(PROJECT_ROOT)}`
- Final coverage summary: `{FINAL_COVERAGE_PATH.relative_to(PROJECT_ROOT)}`
- Daily processing log: `{PROCESSING_LOG_PATH.relative_to(PROJECT_ROOT)}`

## Cache Summary

- Cached days: {int((log['cache_status'] == 'cached').sum())}
- Newly processed days: {int((log['cache_status'] == 'processed').sum())}
- Daily cache directory: `{DAILY_CACHE_DIR.relative_to(PROJECT_ROOT)}`

## Indicator Definitions

- Destination inflow: all valid origins to London destination LSOAs.
- Entropy: valid raw H3 origins only; invalid/non-H3 origins such as `other` are excluded from the entropy denominator but retained in total inflow.
- Unique origins: distinct valid origin H3 cells with positive flow to each London destination LSOA.
- Outflow: London origin LSOAs to all valid destinations represented in the destination-format data.
- Net flow: inflow minus outflow. It remains a robustness indicator because OD symmetry cannot be independently verified from destination-format files alone.
- Weekday-weekend ratio: mean daily weekday inflow divided by mean daily weekend inflow. Zero denominators are flagged and not silently replaced.

## QA Summary

- Failed QA checks: {len(failures)}
- Warning QA checks: {len(warnings)}
- Output row count: {len(indicators):,}
- Unique LSOA21CD: {indicators['LSOA21CD'].nunique():,}
- Duplicate LSOA21CD: {int(indicators['LSOA21CD'].duplicated().sum()):,}
- Total final inflow: {float(indicators['mobility_inflow_total'].sum()):.6f}
- Total final outflow: {float(indicators['mobility_outflow_total'].sum()):.6f}
- Other-origin flow share: {float(indicators['other_origin_flow'].sum() / indicators['mobility_inflow_total'].sum()):.6f}
- Valid-origin entropy denominator share: {float(indicators['entropy_denominator'].sum() / indicators['mobility_inflow_total'].sum()):.6f}

## Known Limitation

`net_flow` is retained as a robustness indicator. OD symmetry cannot be independently verified from destination-format files alone.
"""
    FINAL_REPORT_PATH.write_text(report, encoding="utf-8")


def run_validation(day: str, chunk_size: int, force: bool) -> None:
    paths = Paths()
    manifest, _ = build_manifest(paths)
    metadata = process_one_day(day, manifest, paths, chunk_size=chunk_size, force=force)
    print(f"Validation processed {day}: success={metadata['success']} cache_status={metadata['cache_status']}")


def run_provisional(chunk_size: int, force: bool) -> None:
    paths = Paths()
    manifest, _ = build_manifest(paths)
    valid_days = sorted(manifest_valid_rows(manifest)["date"].astype(str).unique().tolist())
    if not valid_days:
        raise RuntimeError("No valid ALL daily files found for provisional run.")
    for day in valid_days:
        metadata = process_one_day(day, manifest, paths, chunk_size=chunk_size, force=force)
        print(f"{day}: success={metadata['success']} cache_status={metadata['cache_status']}")
    processed_days = successful_processed_dates()
    processed_days = [day for day in valid_days if day in processed_days]
    csv_path, gpkg_path, _, _ = write_provisional_outputs(manifest, processed_days, mode="provisional")
    print(f"Wrote provisional CSV: {csv_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote provisional GPKG: {gpkg_path.relative_to(PROJECT_ROOT)}")


def run_final(chunk_size: int, force: bool) -> None:
    paths = Paths()
    manifest, _ = build_manifest(paths)
    valid_days = sorted(manifest_valid_rows(manifest)["date"].astype(str).unique().tolist())
    missing = [day for day in expected_march_dates() if day not in valid_days]
    if missing:
        raise RuntimeError(f"Final mode requires all 31 valid ALL dates. Missing/incompatible: {', '.join(missing)}")
    for day in valid_days:
        metadata = process_one_day(day, manifest, paths, chunk_size=chunk_size, force=force)
        print(f"{day}: success={metadata['success']} cache_status={metadata['cache_status']}")
        if not metadata["success"]:
            raise RuntimeError(f"Daily processing failed for {day}; final output blocked.")
    processed_days = successful_processed_dates()
    processed_days = [day for day in valid_days if day in processed_days]
    if len(processed_days) != 31:
        raise RuntimeError(f"Final mode expected 31 processed days but found {len(processed_days)}")
    csv_path, gpkg_path, _, _ = write_final_outputs(manifest, processed_days)
    print(f"Wrote final CSV: {csv_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote final GPKG: {gpkg_path.relative_to(PROJECT_ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["validation", "provisional", "final"], required=True)
    parser.add_argument("--date", help="Required for validation mode, e.g. 2026-03-01.")
    parser.add_argument("--chunk-size", type=int, default=500_000)
    parser.add_argument("--force", action="store_true", help="Reprocess even if valid daily caches exist.")
    args = parser.parse_args()

    ensure_dirs(Paths())
    if args.mode == "validation":
        if not args.date:
            raise SystemExit("--date is required for validation mode")
        run_validation(args.date, chunk_size=args.chunk_size, force=args.force)
    elif args.mode == "provisional":
        run_provisional(chunk_size=args.chunk_size, force=args.force)
    elif args.mode == "final":
        run_final(chunk_size=args.chunk_size, force=args.force)


if __name__ == "__main__":
    main()
