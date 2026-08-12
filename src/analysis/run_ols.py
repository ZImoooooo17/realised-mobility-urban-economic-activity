"""CLI for public OLS reconstruction utilities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .ols import compare_ols_models, fit_ols
from .ols_diagnostics import ols_diagnostics, predictor_diagnostics


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """Fit baseline and enhanced OLS models from explicit CSV inputs."""
    parser = argparse.ArgumentParser(description="Fit public OLS baseline and enhanced models.")
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--response", required=True)
    parser.add_argument("--baseline-predictors", nargs="+", required=True)
    parser.add_argument("--enhanced-predictors", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    frame = pd.read_csv(args.input_csv)
    baseline = fit_ols(frame, response=args.response, predictors=args.baseline_predictors)
    enhanced = fit_ols(frame, response=args.response, predictors=args.enhanced_predictors)
    diagnostics = ols_diagnostics(enhanced)
    predictor_info = predictor_diagnostics(frame, args.enhanced_predictors)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    compare_ols_models({"baseline": baseline, "enhanced": enhanced}).to_csv(args.output_dir / "ols_model_comparison.csv", index=False)
    enhanced.coefficients.to_csv(args.output_dir / "enhanced_ols_coefficients.csv", index=False)
    predictor_info["correlation_matrix"].to_csv(args.output_dir / "predictor_correlation_matrix.csv")
    predictor_info["vif"].to_csv(args.output_dir / "variance_inflation_factors.csv", index=False)
    _write_json(
        args.output_dir / "ols_diagnostics.json",
        {
            "baseline_metrics": baseline.metrics,
            "enhanced_metrics": enhanced.metrics,
            "diagnostics": diagnostics,
            "maximum_vif": predictor_info["maximum_vif"],
            "mean_vif": predictor_info["mean_vif"],
            "condition_number": predictor_info["condition_number"],
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
