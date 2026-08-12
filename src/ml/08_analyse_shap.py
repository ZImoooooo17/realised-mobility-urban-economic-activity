#!/usr/bin/env python3
"""E5 SHAP nonlinear shape analysis without retraining XGBoost."""

from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data/processed/master/master_analysis_dataset_final_v1_0.csv"
SHAP_PATH = ROOT / "outputs/modelling/ml/shap_values.csv"
SHAP_QA_PATH = ROOT / "outputs/modelling/ml/shap_interaction_qa.json"

KEY = "LSOA21CD"
FEATURES = [
    "mobility_inflow_total",
    "flow_entropy",
    "weekday_weekend_ratio",
]


def classify_shape(x: np.ndarray, y: np.ndarray, binned: pd.DataFrame) -> tuple[str, str]:
    pearson = stats.pearsonr(x, y).statistic
    spearman = stats.spearmanr(x, y).statistic
    shap_range = float(np.nanpercentile(y, 95) - np.nanpercentile(y, 5))
    if shap_range < 0.30 or abs(spearman) < 0.30:
        return "weak / unclear", "Small rank association or narrow SHAP contribution spread."

    means = binned["mean_shap"].to_numpy()
    slopes = np.diff(means)
    tolerance = max(0.02, 0.08 * np.nanstd(means))
    meaningful = slopes[np.abs(slopes) > tolerance]
    sign_changes = int(np.sum(np.sign(meaningful[1:]) != np.sign(meaningful[:-1]))) if len(meaningful) > 1 else 0
    dominant_share = float(np.mean(np.sign(slopes) == np.sign(spearman))) if len(slopes) else 0.0

    if abs(spearman) >= 0.90 and dominant_share >= 0.80:
        return "monotonic nonlinear", "Very strong rank association with a mostly one-direction binned profile, but the slope changes across the feature range."

    crosses_zero_once = bool(np.nanmin(means) < 0 < np.nanmax(means))
    if crosses_zero_once and abs(spearman) >= 0.55:
        first_positive = np.where(means > 0)[0]
        if len(first_positive) and np.all(means[first_positive[0] :] > -tolerance):
            return "threshold", "Binned SHAP profile shifts from negative or near-zero contribution into sustained positive contribution."

    if abs(spearman) >= 0.75 and dominant_share >= 0.80:
        return "monotonic nonlinear", "Strong rank association with a mostly one-direction binned profile, but the slope changes across the feature range."

    if sign_changes >= 2:
        return "non-monotonic", "Binned SHAP profile changes direction repeatedly."

    same_direction = abs(spearman) >= 0.55 and abs(pearson) >= 0.50
    if same_direction and abs(abs(spearman) - abs(pearson)) < 0.15:
        return "approximately linear", "Pearson and Spearman associations are both strong and similar."

    if abs(spearman) >= 0.45 and sign_changes == 0:
        first_half = np.nanmean(np.abs(slopes[: max(1, len(slopes) // 2)]))
        second_half = np.nanmean(np.abs(slopes[max(1, len(slopes) // 2) :]))
        if second_half < 0.55 * first_half:
            return "saturation / diminishing returns", "Binned SHAP slope weakens in the upper feature range."
        return "monotonic nonlinear", "Rank association is clear but the binned profile is not close to linear."

    if sign_changes == 1 and shap_range >= 0.30:
        return "threshold", "Binned profile shows one main directional shift."

    return "weak / unclear", "Pattern does not meet conservative thresholds for a clearer shape label."


def analyse_feature(df: pd.DataFrame, feature: str) -> dict[str, object]:
    shap_col = f"shap_{feature}"
    x = df[feature].to_numpy()
    y = df[shap_col].to_numpy()

    work = df[[KEY, feature, shap_col]].copy()
    work["bin"] = pd.qcut(work[feature], q=20, duplicates="drop")
    binned = (
        work.groupby("bin", observed=True)
        .agg(
            n=(KEY, "size"),
            min_feature=(feature, "min"),
            max_feature=(feature, "max"),
            median_feature=(feature, "median"),
            mean_feature=(feature, "mean"),
            mean_shap=(shap_col, "mean"),
            median_shap=(shap_col, "median"),
            p25_shap=(shap_col, lambda s: s.quantile(0.25)),
            p75_shap=(shap_col, lambda s: s.quantile(0.75)),
        )
        .reset_index(drop=True)
    )
    label, rationale = classify_shape(x, y, binned)
    pearson = stats.pearsonr(x, y)
    spearman = stats.spearmanr(x, y)
    result = {
        "feature": feature,
        "shap_column": shap_col,
        "n": int(len(work)),
        "pearson_r": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
        "spearman_rho": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
        "feature_min": float(np.nanmin(x)),
        "feature_median": float(np.nanmedian(x)),
        "feature_max": float(np.nanmax(x)),
        "shap_p05": float(np.nanpercentile(y, 5)),
        "shap_p50": float(np.nanpercentile(y, 50)),
        "shap_p95": float(np.nanpercentile(y, 95)),
        "shape_classification": label,
        "rationale": rationale,
    }
    binned.to_csv(OUT / f"{feature}_shap_binned_profile.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.scatter(work[feature], work[shap_col], s=7, alpha=0.20, linewidths=0, color="#3b6ea8")
    ax.plot(binned["median_feature"], binned["mean_shap"], color="#b23a48", linewidth=2.2)
    ax.axhline(0, color="#555555", linewidth=0.8)
    ax.set_xlabel(feature)
    ax.set_ylabel(f"SHAP value: {feature}")
    ax.set_title(f"SHAP Dependence: {feature}")
    ax.text(
        0.02,
        0.98,
        label,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.9},
    )
    fig.tight_layout()
    fig.savefig(OUT / f"{feature}_shap_dependence.png", dpi=180)
    plt.close(fig)
    return result


def main() -> None:
    data = pd.read_csv(DATA_PATH)
    shap = pd.read_csv(SHAP_PATH)
    shap_qa = json.loads(SHAP_QA_PATH.read_text(encoding="utf-8"))

    required_shap_cols = [f"shap_{feature}" for feature in FEATURES]
    missing = [col for col in [KEY, *FEATURES] if col not in data.columns] + [
        col for col in [KEY, *required_shap_cols] if col not in shap.columns
    ]
    if missing:
        raise RuntimeError(f"Missing required columns: {missing}")
    if len(shap) != 4994 or shap[KEY].duplicated().any():
        raise RuntimeError("Unexpected SHAP row count or duplicated SHAP LSOA codes.")

    df = data[[KEY, *FEATURES]].merge(shap[[KEY, *required_shap_cols]], on=KEY, how="inner", validate="one_to_one")
    if len(df) != 4994:
        raise RuntimeError(f"Feature/SHAP join did not preserve 4,994 rows: {len(df)}")
    if df.isna().any().any():
        raise RuntimeError("Missing values after feature/SHAP join.")

    results = [analyse_feature(df, feature) for feature in FEATURES]
    summary = pd.DataFrame(results)
    summary.to_csv(OUT / "shap_nonlinear_shape_summary.csv", index=False)
    df.to_csv(OUT / "shap_feature_joined_data.csv", index=False)

    (OUT / "source_manifest.json").write_text(
        json.dumps(
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "purpose": "E5 SHAP nonlinear shape analysis without retraining XGBoost",
                "authoritative_inputs": {
                    "dataset": str(DATA_PATH.relative_to(ROOT)),
                    "shap_values": str(SHAP_PATH.relative_to(ROOT)),
                    "shap_interaction_qa": str(SHAP_QA_PATH.relative_to(ROOT)),
                },
                "features_analysed": FEATURES,
                "model_retrained": False,
                "true_interaction_values_available": shap_qa.get("true_interaction_values_generated"),
                "qa": {
                    "joined_rows": len(df),
                    "duplicate_lsoa_after_join": int(df[KEY].duplicated().sum()),
                    "missing_values_after_join": int(df.isna().sum().sum()),
                    "shap_rows_authoritative": len(shap),
                },
                "environment": {
                    "python_executable": sys.executable,
                    "python_version": sys.version,
                    "platform": platform.platform(),
                    "pandas": pd.__version__,
                    "numpy": np.__version__,
                    "matplotlib": matplotlib.__version__,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("E5 complete")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
