"""Synthetic tests for public OLS reconstruction helpers."""

from __future__ import annotations

import importlib

import pandas as pd


ols_models = importlib.import_module("src.analysis.ols")
ols_diagnostics = importlib.import_module("src.analysis.ols_diagnostics")


def test_fit_ols_recovers_simple_linear_relationship() -> None:
    frame = pd.DataFrame({"y": [1.0, 3.0, 5.0, 7.0], "x": [0.0, 1.0, 2.0, 3.0], "z": [1.0, 1.0, 2.0, 2.0]})
    result = ols_models.fit_ols(frame, response="y", predictors=["x", "z"])

    assert result.metrics["n"] == 4
    assert result.metrics["rmse"] < 1e-10
    assert result.coefficients.loc[result.coefficients["variable"] == "x", "coefficient"].iloc[0] > 0


def test_predictor_diagnostics_returns_vif_and_condition_number() -> None:
    frame = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0], "z": [0.0, 1.0, 1.0, 2.0]})
    diagnostics = ols_diagnostics.predictor_diagnostics(frame, ["x", "z"])

    assert diagnostics["maximum_vif"] >= 1.0
    assert diagnostics["condition_number"] >= 1.0
