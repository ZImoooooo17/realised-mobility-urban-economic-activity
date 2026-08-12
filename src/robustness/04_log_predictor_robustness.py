#!/usr/bin/env python3
"""E6 log-X functional-form robustness for the combined OLS specification."""

from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import esda
import geopandas as gpd
import libpysal
import numpy as np
import pandas as pd
import scipy
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.stats.diagnostic import het_breuschpagan, linear_reset


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data/processed/master/master_analysis_dataset_final_v1_0.csv"
GEOMETRY_PATH = ROOT / "data/processed/master/master_analysis_dataset_final_v1_0.gpkg"
ORIGINAL_METRICS = ROOT / "outputs/modelling/ols/ols_model_metrics.csv"
ORIGINAL_COEFFICIENTS = ROOT / "outputs/modelling/ols/table_4_3_ols_results_full_precision.csv"
ORIGINAL_DIAGNOSTICS = ROOT / "outputs/modelling/ols/table_A4_ols_diagnostics_full_precision.csv"

KEY = "LSOA21CD"
RESPONSE = "log1p_employees_count"
ORIGINAL_PREDICTORS = [
    "mobility_inflow_total",
    "flow_entropy",
    "weekday_weekend_ratio",
    "ptal_ai_mean",
    "population_density_km2",
    "amenity_poi_density_km2",
    "retail_poi_density_km2",
]
LOG_TRANSFORMS = {
    "mobility_inflow_total": "log1p_mobility_inflow_total",
    "population_density_km2": "log1p_population_density_km2",
    "amenity_poi_density_km2": "log1p_amenity_poi_density_km2",
    "retail_poi_density_km2": "log1p_retail_poi_density_km2",
}
LOGX_PREDICTORS = [
    "log1p_mobility_inflow_total",
    "flow_entropy",
    "weekday_weekend_ratio",
    "ptal_ai_mean",
    "log1p_population_density_km2",
    "log1p_amenity_poi_density_km2",
    "log1p_retail_poi_density_km2",
]


def rmse(y_true: pd.Series, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def standardised_betas(model, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    y_sd = float(y.std(ddof=0))
    return {
        col: float(model.params[col] * X[col].std(ddof=0) / y_sd)
        for col in X.columns
        if col != "const"
    }


def model_outputs(df: pd.DataFrame, predictors: list[str]) -> tuple[object, object, pd.DataFrame, dict[str, float]]:
    y = df[RESPONSE]
    X = sm.add_constant(df[predictors], has_constant="add")
    model = sm.OLS(y, X).fit()
    robust = model.get_robustcov_results(cov_type="HC3")
    coeffs = pd.DataFrame(
        {
            "variable": X.columns,
            "coefficient": model.params,
            "hc3_standard_error": robust.bse,
            "hc3_t": robust.tvalues,
            "hc3_p_value": robust.pvalues,
            "standardised_beta": [np.nan] + [standardised_betas(model, X, y)[c] for c in X.columns if c != "const"],
        }
    )
    pred = model.predict(X)
    metrics = {
        "n": int(model.nobs),
        "df_model": float(model.df_model),
        "df_resid": float(model.df_resid),
        "r_squared": float(model.rsquared),
        "adjusted_r_squared": float(model.rsquared_adj),
        "rmse": rmse(y, pred),
        "mae": float(mean_absolute_error(y, pred)),
        "aic": float(model.aic),
        "bic": float(model.bic),
        "log_likelihood": float(model.llf),
    }
    return model, robust, coeffs, metrics


def diagnostics(model, df: pd.DataFrame, gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    y = df[RESPONSE]
    X = model.model.exog
    bp = het_breuschpagan(model.resid, X)
    reset = linear_reset(model, power=2, use_f=True)
    w = libpysal.weights.Queen.from_dataframe(gdf, ids=KEY, use_index=False)
    w.transform = "R"
    np.random.seed(20260722)
    moran = esda.Moran(model.resid, w, permutations=999)
    return pd.DataFrame(
        [
            {"diagnostic": "Breusch-Pagan LM", "statistic": float(bp[0]), "p_value": float(bp[1]), "detail": "heteroskedasticity"},
            {"diagnostic": "Breusch-Pagan F", "statistic": float(bp[2]), "p_value": float(bp[3]), "detail": "heteroskedasticity"},
            {"diagnostic": "Ramsey RESET F", "statistic": float(reset.fvalue), "p_value": float(reset.pvalue), "detail": "functional form"},
            {"diagnostic": "Residual Moran's I", "statistic": float(moran.I), "p_value": float(moran.p_sim), "detail": f"z={moran.z_sim}; expected={moran.EI}"},
        ]
    )


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    gdf = gpd.read_file(GEOMETRY_PATH)
    if list(df[KEY]) != list(gdf[KEY]):
        raise RuntimeError("CSV/GPKG order mismatch.")
    if len(df) != 4994 or df[KEY].nunique() != 4994:
        raise RuntimeError("Unexpected sample.")
    if df[[RESPONSE, *ORIGINAL_PREDICTORS]].isna().any().any():
        raise RuntimeError("Missing model values.")

    for source, target in LOG_TRANSFORMS.items():
        if (df[source] < 0).any():
            raise RuntimeError(f"Cannot log1p negative predictor: {source}")
        df[target] = np.log1p(df[source])

    model, robust, coeffs, metrics = model_outputs(df, LOGX_PREDICTORS)
    diag = diagnostics(model, df, gdf)

    original_metrics = pd.read_csv(ORIGINAL_METRICS).iloc[0].to_dict()
    comparison = pd.DataFrame(
        [
            {"specification": "primary_combined_original", **{k: original_metrics[k] for k in ["r_squared", "adjusted_r_squared", "rmse", "mae", "aic", "bic", "log_likelihood"]}},
            {"specification": "E6_logX_combined", **metrics},
        ]
    )
    for col in ["r_squared", "adjusted_r_squared", "rmse", "mae", "aic", "bic", "log_likelihood"]:
        comparison[f"delta_{col}_vs_primary"] = comparison[col] - comparison.loc[0, col]

    original_coeffs = pd.read_csv(ORIGINAL_COEFFICIENTS)
    original_predictor_rows = original_coeffs[original_coeffs["source_column"].isin(ORIGINAL_PREDICTORS)].copy()
    direction_map = {
        "log1p_mobility_inflow_total": "mobility_inflow_total",
        "flow_entropy": "flow_entropy",
        "weekday_weekend_ratio": "weekday_weekend_ratio",
        "ptal_ai_mean": "ptal_ai_mean",
        "log1p_population_density_km2": "population_density_km2",
        "log1p_amenity_poi_density_km2": "amenity_poi_density_km2",
        "log1p_retail_poi_density_km2": "retail_poi_density_km2",
    }
    rows = []
    for _, row in coeffs[coeffs["variable"] != "const"].iterrows():
        original_var = direction_map[row["variable"]]
        old = original_predictor_rows[original_predictor_rows["source_column"] == original_var].iloc[0]
        rows.append(
            {
                "original_variable": original_var,
                "logX_variable": row["variable"],
                "original_coefficient": float(old["Coefficient"]),
                "logX_coefficient": float(row["coefficient"]),
                "original_direction": "positive" if old["Coefficient"] > 0 else "negative",
                "logX_direction": "positive" if row["coefficient"] > 0 else "negative",
                "direction_stable": np.sign(old["Coefficient"]) == np.sign(row["coefficient"]),
                "original_p_value": float(old["Robust p"]),
                "logX_hc3_p_value": float(row["hc3_p_value"]),
                "original_significant_0_05": bool(float(old["Robust p"]) < 0.05),
                "logX_significant_0_05": bool(float(row["hc3_p_value"]) < 0.05),
                "logX_standardised_beta": float(row["standardised_beta"]),
            }
        )
    coefficient_comparison = pd.DataFrame(rows)

    pd.DataFrame([metrics]).to_csv(OUT / "logX_ols_metrics.csv", index=False)
    coeffs.to_csv(OUT / "logX_ols_coefficients.csv", index=False)
    comparison.to_csv(OUT / "primary_vs_logX_model_comparison.csv", index=False)
    coefficient_comparison.to_csv(OUT / "primary_vs_logX_coefficient_comparison.csv", index=False)
    diag.to_csv(OUT / "logX_ols_diagnostics.csv", index=False)
    pd.read_csv(ORIGINAL_DIAGNOSTICS).to_csv(OUT / "primary_ols_diagnostics_reference.csv", index=False)
    pd.DataFrame({KEY: df[KEY], "observed": df[RESPONSE], "predicted_logX": model.predict(sm.add_constant(df[LOGX_PREDICTORS], has_constant="add")), "residual_logX": model.resid}).to_csv(OUT / "logX_ols_predictions.csv", index=False)
    (OUT / "source_manifest.json").write_text(
        json.dumps(
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "purpose": "E6 log-X functional-form robustness",
                "authoritative_inputs": {
                    "dataset": str(DATA_PATH.relative_to(ROOT)),
                    "geometry": str(GEOMETRY_PATH.relative_to(ROOT)),
                    "original_metrics": str(ORIGINAL_METRICS.relative_to(ROOT)),
                    "original_coefficients": str(ORIGINAL_COEFFICIENTS.relative_to(ROOT)),
                    "original_diagnostics": str(ORIGINAL_DIAGNOSTICS.relative_to(ROOT)),
                },
                "response": RESPONSE,
                "log_transforms": LOG_TRANSFORMS,
                "unchanged_predictors": ["flow_entropy", "weekday_weekend_ratio", "ptal_ai_mean"],
                "qa": {
                    "n_equals_4994": len(df) == 4994,
                    "duplicate_lsoa": int(df[KEY].duplicated().sum()),
                    "missing_model_values": int(df[[RESPONSE, *ORIGINAL_PREDICTORS]].isna().sum().sum()),
                    "all_log_sources_nonnegative": all(bool((df[src] >= 0).all()) for src in LOG_TRANSFORMS),
                    "hc3_inference_used": True,
                    "primary_model_not_modified": True,
                },
                "environment": {
                    "python_executable": sys.executable,
                    "python_version": sys.version,
                    "platform": platform.platform(),
                    "pandas": pd.__version__,
                    "numpy": np.__version__,
                    "statsmodels": sm.__version__,
                    "libpysal": libpysal.__version__,
                    "esda": esda.__version__,
                    "scipy": scipy.__version__,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("E6 complete")
    print(comparison.to_string(index=False))
    print(diag.to_string(index=False))


if __name__ == "__main__":
    main()
