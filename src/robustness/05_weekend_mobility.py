#!/usr/bin/env python3
"""E7 weekend-only mobility inflow robustness."""

from __future__ import annotations

import glob
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, mean_squared_error


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data/processed/master/master_analysis_dataset_final_v1_0.csv"
CACHE_GLOB = ROOT / "data/interim/mobility/daily_lsoa/date=*/destination_lsoa_aggregation.parquet"
BASELINE_PATH = ROOT / "outputs/modelling/ols/baseline_model_metrics.csv"
PRIMARY_PATH = ROOT / "outputs/modelling/ols/ols_model_metrics.csv"

KEY = "LSOA21CD"
RESPONSE = "log1p_employees_count"
PREDICTORS = [
    "weekend_mobility_inflow_total",
    "flow_entropy",
    "weekday_weekend_ratio",
    "ptal_ai_mean",
    "population_density_km2",
    "amenity_poi_density_km2",
    "retail_poi_density_km2",
]
BASELINE_PREDICTORS = [
    "ptal_ai_mean",
    "population_density_km2",
    "amenity_poi_density_km2",
    "retail_poi_density_km2",
]


def fit_ols(df: pd.DataFrame, predictors: list[str], model_name: str) -> tuple[pd.DataFrame, dict[str, float], object]:
    y = df[RESPONSE]
    X = sm.add_constant(df[predictors], has_constant="add")
    model = sm.OLS(y, X).fit()
    robust = model.get_robustcov_results(cov_type="HC3")
    pred = model.predict(X)
    y_sd = y.std(ddof=0)
    coeffs = pd.DataFrame(
        {
            "model": model_name,
            "variable": X.columns,
            "coefficient": model.params,
            "hc3_standard_error": robust.bse,
            "hc3_t": robust.tvalues,
            "hc3_p_value": robust.pvalues,
            "standardised_beta": [np.nan] + [float(model.params[col] * X[col].std(ddof=0) / y_sd) for col in X.columns if col != "const"],
        }
    )
    metrics = {
        "model": model_name,
        "n": int(model.nobs),
        "df_model": float(model.df_model),
        "df_resid": float(model.df_resid),
        "r_squared": float(model.rsquared),
        "adjusted_r_squared": float(model.rsquared_adj),
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "mae": float(mean_absolute_error(y, pred)),
        "aic": float(model.aic),
        "bic": float(model.bic),
        "log_likelihood": float(model.llf),
        "formula": f"{RESPONSE} ~ {' + '.join(predictors)}",
    }
    return coeffs, metrics, model


def build_weekend_inflow() -> tuple[pd.DataFrame, pd.DataFrame]:
    daily_rows = []
    lsoa_rows = []
    for path in sorted(glob.glob(str(CACHE_GLOB))):
        df = pd.read_parquet(path)
        date = str(df["date"].iloc[0])
        is_weekend = bool(df["is_weekend"].iloc[0])
        total_users = float(df["EXTRAPOLATED_NUMBER_OF_USERS"].sum())
        daily_rows.append(
            {
                "date": date,
                "is_weekend": is_weekend,
                "rows": len(df),
                "unique_destination_lsoa": int(df["destination_lsoa21cd"].nunique()),
                "total_users_all_origin_classes": total_users,
                "missing_destination_lsoa": int(4994 - df["destination_lsoa21cd"].nunique()),
            }
        )
        if is_weekend:
            agg = (
                df.groupby("destination_lsoa21cd", as_index=False)["EXTRAPOLATED_NUMBER_OF_USERS"]
                .sum()
                .rename(columns={"destination_lsoa21cd": KEY, "EXTRAPOLATED_NUMBER_OF_USERS": "weekend_mobility_inflow_total"})
            )
            lsoa_rows.append(agg)
    if not lsoa_rows:
        raise RuntimeError("No weekend cache rows found.")
    weekend = pd.concat(lsoa_rows, ignore_index=True).groupby(KEY, as_index=False)["weekend_mobility_inflow_total"].sum()
    return weekend, pd.DataFrame(daily_rows)


def main() -> None:
    master = pd.read_csv(DATA_PATH)
    weekend, daily_qa = build_weekend_inflow()
    df = master.merge(weekend, on=KEY, how="left", validate="one_to_one")
    if len(df) != 4994 or df[KEY].nunique() != 4994:
        raise RuntimeError("Unexpected merged sample.")
    if df["weekend_mobility_inflow_total"].isna().any():
        raise RuntimeError("Missing reconstructed weekend inflow values.")

    weekend_dates = daily_qa.loc[daily_qa["is_weekend"], "date"].tolist()
    if len(weekend_dates) != 9:
        raise RuntimeError(f"Unexpected weekend date count: {len(weekend_dates)}")

    qa = {
        "weekend_dates": weekend_dates,
        "weekend_date_count": len(weekend_dates),
        "all_daily_cache_count": len(daily_qa),
        "weekend_daily_rows": daily_qa[daily_qa["is_weekend"]].to_dict(orient="records"),
        "weekend_lsoa_coverage": int(weekend[KEY].nunique()),
        "missing_weekend_inflow_after_join": int(df["weekend_mobility_inflow_total"].isna().sum()),
        "correlation_with_master_weekend_inflow_total": float(df["weekend_mobility_inflow_total"].corr(df["weekend_inflow_total"])),
        "max_abs_difference_vs_master_weekend_inflow_total": float((df["weekend_mobility_inflow_total"] - df["weekend_inflow_total"]).abs().max()),
        "correlation_with_original_aggregate_inflow": float(df["weekend_mobility_inflow_total"].corr(df["mobility_inflow_total"])),
        "coverage_all_4994": int(weekend[KEY].nunique()) == 4994,
    }

    coeffs, metrics, model = fit_ols(df, PREDICTORS, "E7 weekend-inflow combined OLS")
    baseline = pd.read_csv(BASELINE_PATH)
    primary = pd.read_csv(PRIMARY_PATH).assign(model="Primary combined OLS")
    comparison_cols = ["model", "n", "df_model", "df_resid", "r_squared", "adjusted_r_squared", "rmse", "mae", "aic", "bic", "log_likelihood", "formula"]
    comparison = pd.concat(
        [
            baseline[comparison_cols],
            pd.DataFrame([metrics])[comparison_cols],
            primary[comparison_cols],
        ],
        ignore_index=True,
    )
    baseline_vals = comparison.iloc[0]
    for col in ["r_squared", "adjusted_r_squared", "rmse", "mae", "aic", "bic", "log_likelihood"]:
        comparison[f"delta_{col}_vs_baseline"] = comparison[col] - baseline_vals[col]

    daily_qa.to_csv(OUT / "weekend_daily_cache_qa.csv", index=False)
    weekend.to_csv(OUT / "weekend_inflow_reconstructed.csv", index=False)
    pd.DataFrame([qa]).drop(columns=["weekend_daily_rows"]).to_csv(OUT / "weekend_inflow_qa_summary.csv", index=False)
    coeffs.to_csv(OUT / "weekend_inflow_ols_coefficients.csv", index=False)
    pd.DataFrame([metrics]).to_csv(OUT / "weekend_inflow_ols_metrics.csv", index=False)
    comparison.to_csv(OUT / "baseline_weekend_primary_comparison.csv", index=False)
    pd.DataFrame({KEY: df[KEY], "observed": df[RESPONSE], "predicted_weekend_model": model.fittedvalues, "residual_weekend_model": model.resid}).to_csv(OUT / "weekend_inflow_ols_predictions.csv", index=False)
    (OUT / "source_manifest.json").write_text(
        json.dumps(
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "purpose": "E7 weekend-only mobility inflow robustness",
                "authoritative_inputs": {
                    "dataset": str(DATA_PATH.relative_to(ROOT)),
                    "daily_cache_glob": "data/interim/mobility/daily_lsoa/date=*/destination_lsoa_aggregation.parquet",
                    "baseline_metrics": str(BASELINE_PATH.relative_to(ROOT)),
                    "primary_metrics": str(PRIMARY_PATH.relative_to(ROOT)),
                },
                "qa": qa,
                "response": RESPONSE,
                "predictors": PREDICTORS,
                "environment": {
                    "python_executable": sys.executable,
                    "python_version": sys.version,
                    "platform": platform.platform(),
                    "pandas": pd.__version__,
                    "numpy": np.__version__,
                    "statsmodels": sm.__version__,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("E7 complete")
    print(pd.DataFrame([qa]).drop(columns=["weekend_daily_rows"]).to_string(index=False))
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
