#!/usr/bin/env python3
"""E1 mobility-only OLS remediation.

Reads frozen authoritative inputs and writes only inside E1_mobility_only_ols.
"""

from __future__ import annotations

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
BASELINE_PATH = ROOT / "outputs/modelling/ols/baseline_model_metrics.csv"
COMBINED_PATH = ROOT / "outputs/modelling/ols/ols_model_metrics.csv"
SPEC_PATH = ROOT / "outputs/modelling/ols/model_specification.json"

KEY = "LSOA21CD"
RESPONSE = "log1p_employees_count"
MOBILITY_PREDICTORS = [
    "mobility_inflow_total",
    "flow_entropy",
    "weekday_weekend_ratio",
]


def rmse(y_true: pd.Series, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))

    required = [KEY, RESPONSE, "employees_count", *MOBILITY_PREDICTORS]
    missing_cols = [col for col in required if col not in df.columns]
    if missing_cols:
        raise RuntimeError(f"Missing required columns: {missing_cols}")
    if len(df) != 4994:
        raise RuntimeError(f"Unexpected row count: {len(df)}")
    if df[KEY].duplicated().any():
        raise RuntimeError("Duplicate LSOA21CD values found in authoritative dataset.")
    if df[required].isna().any().any():
        raise RuntimeError("Missing values found in required E1 fields.")
    if not np.allclose(df[RESPONSE], np.log1p(df["employees_count"])):
        raise RuntimeError("Response does not match log1p(employees_count).")

    y = df[RESPONSE]
    X = sm.add_constant(df[MOBILITY_PREDICTORS], has_constant="add")
    model = sm.OLS(y, X).fit()
    robust = model.get_robustcov_results(cov_type="HC3")
    pred = model.predict(X)

    coefficients = pd.DataFrame(
        {
            "variable": X.columns,
            "coefficient": model.params,
            "standard_error_classical": model.bse,
            "t_classical": model.tvalues,
            "p_classical": model.pvalues,
            "hc3_standard_error": robust.bse,
            "hc3_t": robust.tvalues,
            "hc3_p_value": robust.pvalues,
            "hc3_ci_lower_95": robust.conf_int()[:, 0],
            "hc3_ci_upper_95": robust.conf_int()[:, 1],
        }
    )

    metrics = {
        "model": "Model E1: Mobility-only OLS",
        "formula": f"{RESPONSE} ~ {' + '.join(MOBILITY_PREDICTORS)}",
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
        "residual_standard_error": float(np.sqrt(model.scale)),
        "predictor_count": len(MOBILITY_PREDICTORS),
        "hc3_robust_inference_used": True,
        "response_variable": RESPONSE,
    }

    baseline = pd.read_csv(BASELINE_PATH)
    combined = pd.read_csv(COMBINED_PATH).assign(
        model="Model B: Combined OLS",
        predictor_count=7,
    )
    combined["formula"] = combined["formula"]
    common_cols = [
        "model",
        "formula",
        "n",
        "df_model",
        "df_resid",
        "r_squared",
        "adjusted_r_squared",
        "rmse",
        "mae",
        "aic",
        "bic",
        "log_likelihood",
        "residual_standard_error",
        "predictor_count",
        "hc3_robust_inference_used",
    ]
    comparison = pd.concat(
        [
            baseline[common_cols],
            pd.DataFrame([metrics])[common_cols],
            combined[common_cols],
        ],
        ignore_index=True,
    )
    baseline_vals = comparison.loc[comparison["model"].str.contains("Baseline"), :].iloc[0]
    for col in ["r_squared", "adjusted_r_squared", "rmse", "mae", "aic", "bic", "log_likelihood"]:
        comparison[f"delta_{col}_vs_baseline"] = comparison[col] - baseline_vals[col]

    predictions = pd.DataFrame(
        {
            KEY: df[KEY],
            "observed": y,
            "predicted_mobility_only": pred,
            "residual_mobility_only": y - pred,
        }
    )

    source_manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "E1 mobility-only OLS remediation",
        "authoritative_inputs": {
            "dataset": str(DATA_PATH.relative_to(ROOT)),
            "baseline_metrics": str(BASELINE_PATH.relative_to(ROOT)),
            "combined_metrics": str(COMBINED_PATH.relative_to(ROOT)),
            "model_specification": str(SPEC_PATH.relative_to(ROOT)),
        },
        "dataset_sha256_from_frozen_spec": spec.get("dataset_sha256"),
        "response": RESPONSE,
        "predictors": MOBILITY_PREDICTORS,
        "qa": {
            "n_equals_4994": len(df) == 4994,
            "duplicate_lsoa_count": int(df[KEY].duplicated().sum()),
            "missing_required_values": int(df[required].isna().sum().sum()),
            "response_matches_log1p_employees_count": bool(np.allclose(df[RESPONSE], np.log1p(df["employees_count"]))),
            "hc3_inference_used": True,
        },
        "environment": {
            "python_executable": sys.executable,
            "python_version": sys.version,
            "platform": platform.platform(),
            "pandas": pd.__version__,
            "statsmodels": sm.__version__,
        },
        "outputs": {
            "coefficients": "mobility_only_ols_coefficients.csv",
            "metrics": "mobility_only_ols_metrics.csv",
            "comparison": "ols_baseline_mobility_only_combined_comparison.csv",
            "predictions": "mobility_only_ols_predictions.csv",
            "source_manifest": "source_manifest.json",
        },
    }

    coefficients.to_csv(OUT / "mobility_only_ols_coefficients.csv", index=False)
    pd.DataFrame([metrics]).to_csv(OUT / "mobility_only_ols_metrics.csv", index=False)
    comparison.to_csv(OUT / "ols_baseline_mobility_only_combined_comparison.csv", index=False)
    predictions.to_csv(OUT / "mobility_only_ols_predictions.csv", index=False)
    (OUT / "source_manifest.json").write_text(json.dumps(source_manifest, indent=2), encoding="utf-8")

    print("E1 complete")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
