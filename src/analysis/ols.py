"""OLS model fitting helpers for public reconstruction work."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, mean_squared_error


@dataclass(frozen=True)
class OLSResult:
    """Container for a fitted OLS model and public-safe summaries."""

    model: Any
    robust_model: Any
    response: str
    predictors: tuple[str, ...]
    metrics: dict[str, float | int | str | bool]
    coefficients: pd.DataFrame
    predictions: pd.DataFrame


def complete_case_frame(frame: pd.DataFrame, response: str, predictors: list[str]) -> pd.DataFrame:
    """Return complete numeric cases for the requested OLS specification."""
    columns = [response, *predictors]
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"missing OLS columns: {missing}")
    numeric = frame[columns].apply(pd.to_numeric, errors="coerce")
    return numeric.replace([np.inf, -np.inf], np.nan).dropna()


def fit_ols(
    frame: pd.DataFrame,
    *,
    response: str,
    predictors: list[str],
    robust_covariance: str = "HC3",
) -> OLSResult:
    """Fit an intercept OLS model with optional robust covariance inference."""
    data = complete_case_frame(frame, response, predictors)
    y = data[response]
    X = sm.add_constant(data[predictors], has_constant="add")
    model = sm.OLS(y, X).fit()
    robust_model = model.get_robustcov_results(cov_type=robust_covariance)
    fitted = model.fittedvalues
    residual = model.resid
    rmse = float(np.sqrt(mean_squared_error(y, fitted)))
    mae = float(mean_absolute_error(y, fitted))
    metrics: dict[str, float | int | str | bool] = {
        "n": int(model.nobs),
        "df_model": float(model.df_model),
        "df_resid": float(model.df_resid),
        "r_squared": float(model.rsquared),
        "adjusted_r_squared": float(model.rsquared_adj),
        "rmse": rmse,
        "mae": mae,
        "aic": float(model.aic),
        "bic": float(model.bic),
        "log_likelihood": float(model.llf),
        "residual_standard_error": float(np.sqrt(model.ssr / model.df_resid)),
        "hc3_robust_inference_used": robust_covariance.upper() == "HC3",
        "response_variable": response,
        "formula": f"{response} ~ {' + '.join(predictors)}",
    }
    coefficients = _coefficient_frame(model, robust_model, robust_covariance)
    predictions = pd.DataFrame(
        {
            "observed": y.to_numpy(dtype=float),
            "fitted": np.asarray(fitted, dtype=float),
            "residual": np.asarray(residual, dtype=float),
        },
        index=data.index,
    )
    return OLSResult(
        model=model,
        robust_model=robust_model,
        response=response,
        predictors=tuple(predictors),
        metrics=metrics,
        coefficients=coefficients,
        predictions=predictions,
    )


def _coefficient_frame(model: Any, robust_model: Any, covariance_type: str) -> pd.DataFrame:
    """Build coefficient and HC3 robust inference table."""
    variables = list(model.params.index)
    return pd.DataFrame(
        {
            "variable": variables,
            "coefficient": np.asarray(model.params, dtype=float),
            "standard_error": np.asarray(model.bse, dtype=float),
            "t": np.asarray(model.tvalues, dtype=float),
            "p": np.asarray(model.pvalues, dtype=float),
            "robust_standard_error": np.asarray(robust_model.bse, dtype=float),
            "robust_t": np.asarray(robust_model.tvalues, dtype=float),
            "robust_p": np.asarray(robust_model.pvalues, dtype=float),
            "covariance_type": covariance_type.upper(),
        }
    )


def compare_ols_models(results: dict[str, OLSResult]) -> pd.DataFrame:
    """Return a compact model-comparison table for fitted OLS results."""
    rows = []
    for name, result in results.items():
        rows.append(
            {
                "model": name,
                "n": result.metrics["n"],
                "predictor_count": len(result.predictors),
                "adjusted_r_squared": result.metrics["adjusted_r_squared"],
                "rmse": result.metrics["rmse"],
                "mae": result.metrics["mae"],
                "aic": result.metrics["aic"],
                "bic": result.metrics["bic"],
                "log_likelihood": result.metrics["log_likelihood"],
            }
        )
    return pd.DataFrame(rows)
