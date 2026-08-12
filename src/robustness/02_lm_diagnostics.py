#!/usr/bin/env python3
"""E3 LM diagnostics using spreg and the frozen OLS spatial specification."""

from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import libpysal
import numpy as np
import pandas as pd
import spreg


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data/processed/master/master_analysis_dataset_final_v1_0.csv"
GEOMETRY_PATH = ROOT / "data/processed/master/master_analysis_dataset_final_v1_0.gpkg"
SPEC_PATH = ROOT / "outputs/modelling/ols/model_specification.json"
WEIGHTS_DIAGNOSTICS = ROOT / "outputs/esda/tables/spatial_weights_diagnostics.csv"

KEY = "LSOA21CD"
RESPONSE = "log1p_employees_count"
PREDICTORS = [
    "mobility_inflow_total",
    "flow_entropy",
    "weekday_weekend_ratio",
    "ptal_ai_mean",
    "population_density_km2",
    "amenity_poi_density_km2",
    "retail_poi_density_km2",
]


def tuple_to_row(name: str, value: tuple[float, float]) -> dict[str, object]:
    return {"diagnostic": name, "statistic": float(value[0]), "p_value": float(value[1]), "df": 1}


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    gdf = gpd.read_file(GEOMETRY_PATH)
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))

    if list(df[KEY]) != list(gdf[KEY]):
        raise RuntimeError("CSV and GPKG LSOA order differ; stop before LM diagnostics.")
    if len(df) != 4994 or df[KEY].nunique() != 4994:
        raise RuntimeError("Unexpected sample size or duplicated LSOA codes.")
    if df[[RESPONSE, *PREDICTORS]].isna().any().any():
        raise RuntimeError("Missing values found in E3 model fields.")

    w = libpysal.weights.Queen.from_dataframe(gdf, ids=KEY, use_index=False)
    w.transform = "R"

    y = df[[RESPONSE]].to_numpy()
    x = df[PREDICTORS].to_numpy()
    model = spreg.OLS(
        y,
        x,
        w=w,
        nonspat_diag=True,
        spat_diag=True,
        moran=True,
        name_y=RESPONSE,
        name_x=PREDICTORS,
        name_w="Queen order-1 row-standardised",
        name_ds="master_analysis_dataset_final_v1_0",
    )

    diagnostics = pd.DataFrame(
        [
            tuple_to_row("LM-lag", model.lm_lag),
            tuple_to_row("LM-error", model.lm_error),
            tuple_to_row("Robust LM-lag", model.rlm_lag),
            tuple_to_row("Robust LM-error", model.rlm_error),
            tuple_to_row("LM-SARMA", model.lm_sarma),
            {
                "diagnostic": "Moran residual I",
                "statistic": float(model.moran_res[0]),
                "p_value": float(model.moran_res[2]),
                "df": np.nan,
            },
        ]
    )

    weights_info = {
        "object": "libpysal.weights.Queen",
        "constructor": "libpysal.weights.Queen.from_dataframe(gdf, ids=LSOA21CD, use_index=False)",
        "order": 1,
        "transform": w.transform,
        "standardisation": "row-standardised",
        "observations": int(w.n),
        "islands": len(w.islands),
        "connected_components": int(w.n_components),
        "mean_neighbours": float(np.mean([len(v) for v in w.neighbors.values()])),
        "minimum_neighbours": int(np.min([len(v) for v in w.neighbors.values()])),
        "maximum_neighbours": int(np.max([len(v) for v in w.neighbors.values()])),
        "id_order_first_5": list(w.id_order[:5]),
    }

    qa = {
        "uses_authoritative_dataset": str(DATA_PATH.relative_to(ROOT)),
        "uses_authoritative_geometry": str(GEOMETRY_PATH.relative_to(ROOT)),
        "uses_authoritative_ols_specification": str(SPEC_PATH.relative_to(ROOT)),
        "dataset_rows": len(df),
        "unique_lsoa21cd": int(df[KEY].nunique()),
        "duplicate_lsoa21cd": int(df[KEY].duplicated().sum()),
        "csv_order_matches_geometry": list(df[KEY]) == list(gdf[KEY]),
        "same_response_as_frozen_spec": spec.get("response_variable", {}).get("source_column") == RESPONSE,
        "same_predictors_as_frozen_spec": [p.get("source_column") for p in spec.get("predictors", [])] == PREDICTORS,
        "weights_observations_4994": w.n == 4994,
        "weights_islands_zero": len(w.islands) == 0,
        "weights_row_standardised": w.transform == "R",
        "spreg_used": True,
        "manual_lm_formula_used": False,
    }

    diagnostics.to_csv(OUT / "lm_diagnostics.csv", index=False)
    (OUT / "spreg_ols_summary.txt").write_text(model.summary, encoding="utf-8")
    (OUT / "spatial_weights_source.json").write_text(json.dumps(weights_info, indent=2), encoding="utf-8")
    (OUT / "source_manifest.json").write_text(
        json.dumps(
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "purpose": "E3 LM diagnostics remediation",
                "authoritative_inputs": {
                    "dataset": str(DATA_PATH.relative_to(ROOT)),
                    "geometry": str(GEOMETRY_PATH.relative_to(ROOT)),
                    "model_specification": str(SPEC_PATH.relative_to(ROOT)),
                    "prior_weights_diagnostics": str(WEIGHTS_DIAGNOSTICS.relative_to(ROOT)),
                },
                "response": RESPONSE,
                "predictors": PREDICTORS,
                "qa": qa,
                "environment": {
                    "python_executable": sys.executable,
                    "python_version": sys.version,
                    "platform": platform.platform(),
                    "pandas": pd.__version__,
                    "geopandas": gpd.__version__,
                    "libpysal": libpysal.__version__,
                    "spreg": spreg.__version__,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("E3 complete")
    print(diagnostics.to_string(index=False))


if __name__ == "__main__":
    main()
