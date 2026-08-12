"""Import checks for public modules."""

from __future__ import annotations

import importlib


MODULES = [
    "src.analysis.ols",
    "src.analysis.ols_diagnostics",
    "src.analysis.run_ols",
    "src.spatial.weights",
    "src.spatial.spatial_models",
    "src.spatial.run_spatial_models",
]


def test_public_modules_import() -> None:
    for module in MODULES:
        importlib.import_module(module)
