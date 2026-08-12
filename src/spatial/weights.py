"""Spatial weights helpers for public ESDA."""

from __future__ import annotations

import pandas as pd


def queen_weights(gdf, *, key: str = "LSOA21CD", row_standardise: bool = True):
    """Construct Queen contiguity weights preserving input identifier order."""
    from libpysal.weights import Queen

    if key not in gdf.columns:
        raise ValueError(f"missing spatial key column {key!r}")
    if gdf[key].duplicated().any():
        raise ValueError(f"spatial key column {key!r} contains duplicates")
    weights = Queen.from_dataframe(gdf, ids=list(gdf[key]), use_index=False)
    if row_standardise:
        weights.transform = "R"
    return weights


def weights_diagnostics(weights) -> pd.DataFrame:
    """Summarise a libpysal weights object."""
    neighbours = pd.Series({key: len(value) for key, value in weights.neighbors.items()})
    return pd.DataFrame(
        [
            {
                "n": int(weights.n),
                "mean_neighbours": float(neighbours.mean()),
                "median_neighbours": float(neighbours.median()),
                "minimum_neighbours": int(neighbours.min()),
                "maximum_neighbours": int(neighbours.max()),
                "islands": int(len(weights.islands)),
                "connected_components": int(getattr(weights, "n_components", 1)),
                "transform": getattr(weights, "transform", ""),
            }
        ]
    )
