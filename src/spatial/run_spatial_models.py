"""CLI for public SAR and SEM reconstruction utilities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import pandas as pd

from .spatial_models import fit_sar, fit_sem, spatial_model_comparison
from .weights import queen_weights


def main(argv: list[str] | None = None) -> int:
    """Fit SAR and SEM from explicit public-safe inputs."""
    parser = argparse.ArgumentParser(description="Fit public SAR and SEM reconstructions.")
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--geometry-file", type=Path, required=True)
    parser.add_argument("--key", default="LSOA21CD")
    parser.add_argument("--response", required=True)
    parser.add_argument("--predictors", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    frame = pd.read_csv(args.input_csv)
    geometry = gpd.read_file(args.geometry_file)
    ordered = geometry[[args.key, "geometry"]].merge(frame, on=args.key, how="inner", validate="one_to_one")
    weights = queen_weights(ordered, key=args.key, row_standardise=True)
    sar = fit_sar(ordered, weights, response=args.response, predictors=args.predictors)
    sem = fit_sem(ordered, weights, response=args.response, predictors=args.predictors)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    spatial_model_comparison([sar, sem]).to_csv(args.output_dir / "spatial_model_comparison.csv", index=False)
    sar.inference.to_csv(args.output_dir / "sar_coefficients.csv", index=False)
    sem.inference.to_csv(args.output_dir / "sem_coefficients.csv", index=False)
    (args.output_dir / "spatial_metrics.json").write_text(
        json.dumps(
            {
                "sar": {"rho": sar.spatial_parameter, "log_likelihood": sar.log_likelihood, "aic": sar.aic, "rmse": sar.rmse},
                "sem": {"lambda": sem.spatial_parameter, "log_likelihood": sem.log_likelihood, "aic": sem.aic, "rmse": sem.rmse},
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
