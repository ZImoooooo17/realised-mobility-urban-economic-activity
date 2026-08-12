"""Concentrated-likelihood SAR and SEM helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy import sparse
from scipy import stats
from scipy.optimize import minimize_scalar
from scipy.sparse.linalg import splu
from sklearn.metrics import mean_absolute_error, mean_squared_error


@dataclass(frozen=True)
class SpatialModelResult:
    """Container for a fitted spatial econometric model."""

    model: Literal["SAR", "SEM"]
    spatial_parameter_name: Literal["rho", "lambda"]
    spatial_parameter: float
    coefficients: pd.Series
    inference: pd.DataFrame
    log_likelihood: float
    aic: float
    bic: float
    rmse: float
    mae: float
    sigma2_ml: float
    predictions: pd.Series
    residuals: pd.Series
    optimisation_converged: bool
    optimisation_message: str


def weights_to_sparse(weights) -> sparse.csr_matrix:
    """Convert a libpysal weights object to a CSR sparse matrix."""
    matrix = weights.sparse
    return matrix.tocsr() if sparse.issparse(matrix) else sparse.csr_matrix(matrix)


def design_matrix(frame: pd.DataFrame, response: str, predictors: list[str]) -> tuple[pd.Series, pd.DataFrame]:
    """Return complete-case response and intercept design matrix."""
    columns = [response, *predictors]
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"missing spatial model columns: {missing}")
    data = frame[columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    X = data[predictors].copy()
    X.insert(0, "const", 1.0)
    return data[response], X


def fit_sar(
    frame: pd.DataFrame,
    weights,
    *,
    response: str,
    predictors: list[str],
    bounds: tuple[float, float] = (-0.95, 0.95),
) -> SpatialModelResult:
    """Fit a spatial lag model by direct concentrated maximum likelihood."""
    y, X = design_matrix(frame, response, predictors)
    W = weights_to_sparse(weights)
    yv = y.to_numpy(dtype=float)
    Xv = X.to_numpy(dtype=float)
    I = sparse.identity(W.shape[0], format="csc")

    def objective(rho: float) -> float:
        A = I - rho * W
        y_rho = A @ yv
        beta = np.linalg.lstsq(Xv, y_rho, rcond=None)[0]
        residual = y_rho - Xv @ beta
        sigma2 = float(residual @ residual / len(yv))
        return -_log_likelihood(A, sigma2, len(yv))

    optimum = minimize_scalar(objective, bounds=bounds, method="bounded", options={"xatol": 1e-12})
    rho = float(optimum.x)
    A = I - rho * W
    beta = np.linalg.lstsq(Xv, A @ yv, rcond=None)[0]
    linear = Xv @ beta
    fitted = rho * (W @ yv) + linear
    structural_residual = yv - fitted
    sigma2 = float(structural_residual @ structural_residual / len(yv))
    ll = _log_likelihood(A, sigma2, len(yv))
    k = Xv.shape[1] + 2
    beta_se = np.sqrt(np.diag(sigma2 * np.linalg.inv(Xv.T @ Xv)))
    parameter_se = _spatial_parameter_standard_error(objective, rho, step=1e-5)
    return _result("SAR", "rho", rho, parameter_se, beta, beta_se, X.columns, y.index, fitted, structural_residual, ll, sigma2, k, bool(optimum.success), str(optimum.message), yv)


def fit_sem(
    frame: pd.DataFrame,
    weights,
    *,
    response: str,
    predictors: list[str],
    bounds: tuple[float, float] = (-0.95, 0.95),
) -> SpatialModelResult:
    """Fit a spatial error model by direct concentrated maximum likelihood."""
    y, X = design_matrix(frame, response, predictors)
    W = weights_to_sparse(weights)
    yv = y.to_numpy(dtype=float)
    Xv = X.to_numpy(dtype=float)
    I = sparse.identity(W.shape[0], format="csc")

    def objective(lam: float) -> float:
        A = I - lam * W
        y_lam = A @ yv
        X_lam = A @ Xv
        beta = np.linalg.lstsq(X_lam, y_lam, rcond=None)[0]
        innovation = y_lam - X_lam @ beta
        sigma2 = float(innovation @ innovation / len(yv))
        return -_log_likelihood(A, sigma2, len(yv))

    optimum = minimize_scalar(objective, bounds=bounds, method="bounded", options={"xatol": 1e-12})
    lam = float(optimum.x)
    A = I - lam * W
    y_lam = A @ yv
    X_lam = A @ Xv
    beta = np.linalg.lstsq(X_lam, y_lam, rcond=None)[0]
    fitted = Xv @ beta
    residual = yv - fitted
    innovation = y_lam - X_lam @ beta
    sigma2 = float(innovation @ innovation / len(yv))
    ll = _log_likelihood(A, sigma2, len(yv))
    k = Xv.shape[1] + 2
    beta_se = np.sqrt(np.diag(sigma2 * np.linalg.inv(X_lam.T @ X_lam)))
    parameter_se = _spatial_parameter_standard_error(objective, lam, step=1.1e-5)
    return _result("SEM", "lambda", lam, parameter_se, beta, beta_se, X.columns, y.index, fitted, residual, ll, sigma2, k, bool(optimum.success), str(optimum.message), yv)


def _log_likelihood(A: sparse.spmatrix, sigma2: float, n: int) -> float:
    """Evaluate Gaussian spatial concentrated log-likelihood."""
    if sigma2 <= 0:
        return -np.inf
    logdet = float(np.sum(np.log(np.abs(splu(A.tocsc()).U.diagonal()))))
    return logdet - (n / 2.0) * (np.log(2.0 * np.pi * sigma2) + 1.0)


def _spatial_parameter_standard_error(objective, optimum: float, *, step: float = 1e-5) -> float:
    """Estimate the spatial-parameter standard error from the likelihood Hessian."""
    hessian = (objective(optimum + step) - 2.0 * objective(optimum) + objective(optimum - step)) / (step * step)
    if hessian <= 0:
        raise ValueError("spatial-parameter likelihood Hessian is not positive")
    return float(np.sqrt(1.0 / hessian))


def _inference_frame(
    model: Literal["SAR", "SEM"],
    beta: np.ndarray,
    beta_standard_errors: np.ndarray,
    columns: pd.Index,
    parameter_name: Literal["rho", "lambda"],
    parameter: float,
    parameter_standard_error: float,
) -> pd.DataFrame:
    """Return coefficient estimates, standard errors, z statistics, and p values."""
    variables = list(columns) + [parameter_name]
    estimates = np.concatenate([beta, np.asarray([parameter], dtype=float)])
    standard_errors = np.concatenate([beta_standard_errors, np.asarray([parameter_standard_error], dtype=float)])
    z_statistics = estimates / standard_errors
    p_values = 2.0 * stats.norm.sf(np.abs(z_statistics))
    return pd.DataFrame(
        {
            "model": model,
            "variable": variables,
            "estimate": estimates,
            "standard_error": standard_errors,
            "z_statistic": z_statistics,
            "p_value": p_values,
        }
    )


def _result(
    model: Literal["SAR", "SEM"],
    parameter_name: Literal["rho", "lambda"],
    parameter: float,
    parameter_standard_error: float,
    beta: np.ndarray,
    beta_standard_errors: np.ndarray,
    columns: pd.Index,
    index: pd.Index,
    fitted: np.ndarray,
    residual: np.ndarray,
    log_likelihood: float,
    sigma2: float,
    k: int,
    converged: bool,
    message: str,
    observed: np.ndarray,
) -> SpatialModelResult:
    """Build a spatial result object."""
    return SpatialModelResult(
        model=model,
        spatial_parameter_name=parameter_name,
        spatial_parameter=float(parameter),
        coefficients=pd.Series(beta, index=columns, dtype=float),
        inference=_inference_frame(model, beta, beta_standard_errors, columns, parameter_name, parameter, parameter_standard_error),
        log_likelihood=float(log_likelihood),
        aic=float(2 * k - 2 * log_likelihood),
        bic=float(np.log(len(observed)) * k - 2 * log_likelihood),
        rmse=float(np.sqrt(mean_squared_error(observed, fitted))),
        mae=float(mean_absolute_error(observed, fitted)),
        sigma2_ml=float(sigma2),
        predictions=pd.Series(fitted, index=index, name="fitted"),
        residuals=pd.Series(residual, index=index, name="residual"),
        optimisation_converged=converged,
        optimisation_message=message,
    )


def spatial_model_comparison(results: list[SpatialModelResult], *, ols_metrics: dict[str, float] | None = None) -> pd.DataFrame:
    """Return compact OLS/SAR/SEM comparison rows."""
    rows = []
    if ols_metrics:
        rows.append({"model": "OLS", **ols_metrics})
    for result in results:
        rows.append(
            {
                "model": result.model,
                "log_likelihood": result.log_likelihood,
                "aic": result.aic,
                "bic": result.bic,
                "rmse": result.rmse,
                "mae": result.mae,
                "spatial_parameter": result.spatial_parameter_name,
                "spatial_parameter_estimate": result.spatial_parameter,
                "spatial_parameter_standard_error": float(
                    result.inference.loc[result.inference["variable"] == result.spatial_parameter_name, "standard_error"].iloc[0]
                ),
                "sigma2_ml": result.sigma2_ml,
                "optimisation_converged": result.optimisation_converged,
            }
        )
    return pd.DataFrame(rows)
