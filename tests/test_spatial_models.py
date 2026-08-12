"""Synthetic tests for spatial model reconstruction helpers."""

from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
from scipy import sparse


spatial_models = importlib.import_module("src.spatial.spatial_models")


class DummyWeights:
    """Tiny row-standardised weights object for tests."""

    def __init__(self) -> None:
        self.sparse = sparse.csr_matrix(
            np.array(
                [
                    [0.0, 1.0, 0.0, 0.0],
                    [0.5, 0.0, 0.5, 0.0],
                    [0.0, 0.5, 0.0, 0.5],
                    [0.0, 0.0, 1.0, 0.0],
                ]
            )
        )


def test_spatial_models_return_finite_metrics() -> None:
    frame = pd.DataFrame({"y": [1.0, 2.0, 2.7, 4.2], "x": [0.0, 1.0, 2.0, 3.0]})
    weights = DummyWeights()

    sar = spatial_models.fit_sar(frame, weights, response="y", predictors=["x"], bounds=(-0.5, 0.5))
    sem = spatial_models.fit_sem(frame, weights, response="y", predictors=["x"], bounds=(-0.5, 0.5))

    assert np.isfinite(sar.aic)
    assert np.isfinite(sem.aic)
    assert {"estimate", "standard_error", "z_statistic", "p_value"}.issubset(sar.inference.columns)
    assert {"estimate", "standard_error", "z_statistic", "p_value"}.issubset(sem.inference.columns)
    assert sar.optimisation_converged
    assert sem.optimisation_converged
