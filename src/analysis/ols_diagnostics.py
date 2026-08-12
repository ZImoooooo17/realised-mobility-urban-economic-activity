"""OLS diagnostic helpers for public reconstruction work."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan, linear_reset
from statsmodels.stats.outliers_influence import variance_inflation_factor


def standardise_numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Z-standardise numeric predictors using population standard deviation."""
    values = frame[columns].apply(pd.to_numeric, errors="coerce").dropna().copy()
    constant = [column for column in values if values[column].nunique(dropna=True) <= 1]
    values = values.drop(columns=constant)
    z = (values - values.mean()) / values.std(ddof=0)
    return z.replace([np.inf, -np.inf], np.nan).dropna()


def variance_inflation_factors(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Calculate VIF on z-standardised predictors without an intercept."""
    z = standardise_numeric(frame, columns)
    matrix = z.to_numpy(dtype=float)
    rows = []
    for index, column in enumerate(z.columns):
        vif = float(variance_inflation_factor(matrix, index))
        rows.append({"variable": column, "vif": vif, "tolerance": 1.0 / vif if vif else np.nan})
    return pd.DataFrame(rows)


def predictor_diagnostics(frame: pd.DataFrame, predictors: list[str]) -> dict[str, object]:
    """Calculate correlation, VIF, and condition diagnostics for predictors."""
    z = standardise_numeric(frame, predictors)
    corr = z.corr(method="pearson")
    vif = variance_inflation_factors(frame, predictors)
    condition_number = float(np.linalg.cond(z.to_numpy(dtype=float)))
    return {
        "correlation_matrix": corr,
        "vif": vif,
        "maximum_vif": float(vif["vif"].max()),
        "mean_vif": float(vif["vif"].mean()),
        "condition_number": condition_number,
    }


def ols_diagnostics(result, *, reset_power: int = 2) -> dict[str, object]:
    """Calculate JB, Breusch-Pagan, RESET, and residual-scale diagnostics."""
    residuals = np.asarray(result.model.resid, dtype=float)
    jb = stats.jarque_bera(residuals)
    bp_lm, bp_lm_p, bp_f, bp_f_p = het_breuschpagan(residuals, result.model.model.exog)
    reset = linear_reset(result.model, power=reset_power, test_type="fitted", use_f=True)
    return {
        "jarque_bera": {
            "statistic": float(jb.statistic),
            "p_value": float(jb.pvalue),
            "skewness": float(stats.skew(residuals)),
            "kurtosis": float(stats.kurtosis(residuals, fisher=False)),
        },
        "breusch_pagan": {
            "statistic": float(bp_lm),
            "p_value": float(bp_lm_p),
            "f_statistic": float(bp_f),
            "f_p_value": float(bp_f_p),
        },
        "ramsey_reset": {
            "statistic": float(reset.fvalue),
            "p_value": float(reset.pvalue),
            "degrees_of_freedom": [int(reset.df_num), int(reset.df_denom)],
            "test_type": "fitted",
            "power": reset_power,
        },
        "residual_standard_error": float(np.sqrt(result.model.ssr / result.model.df_resid)),
        "maximum_absolute_standardised_residual": float(np.max(np.abs(result.model.get_influence().resid_studentized_internal))),
    }


def residual_moran(
    residuals: pd.Series | np.ndarray,
    weights,
    *,
    permutations: int = 999,
    random_seed: int | None = 20260722,
) -> dict[str, float | int]:
    """Calculate Global Moran's I for model residuals."""
    if random_seed is not None:
        np.random.seed(random_seed)
    from esda import Moran

    values = np.asarray(residuals, dtype=float)
    moran = Moran(values, weights, transformation="r", permutations=permutations, two_tailed=True)
    return {
        "moran_i": float(moran.I),
        "expected_i": float(moran.EI),
        "z_score": float(moran.z_norm),
        "permutation_p_value": float(moran.p_sim),
        "permutations": int(permutations),
    }
