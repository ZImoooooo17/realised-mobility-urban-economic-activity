"""Synthetic tests for Queen weights."""

from __future__ import annotations

import importlib.util
import importlib

import pytest


pytestmark = pytest.mark.skipif(importlib.util.find_spec("libpysal") is None, reason="libpysal is not installed")


def test_queen_weights_preserve_observation_order() -> None:
    import geopandas as gpd
    from shapely.geometry import box

    weights_module = importlib.import_module("src.spatial.weights")
    gdf = gpd.GeoDataFrame(
        {"LSOA21CD": ["A", "B", "C"], "geometry": [box(0, 0, 1, 1), box(1, 0, 2, 1), box(2, 0, 3, 1)]},
        crs="EPSG:27700",
    )
    weights = weights_module.queen_weights(gdf)
    assert list(weights.id_order) == ["A", "B", "C"]
