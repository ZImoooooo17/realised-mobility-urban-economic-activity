#!/usr/bin/env python3
"""E4 comparable held-out OLS benchmark using the existing ML test split."""

from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data/processed/master/master_analysis_dataset_final_v1_0.csv"
GEOMETRY_PATH = ROOT / "data/processed/master/master_analysis_dataset_final_v1_0.gpkg"
ML_SUMMARY_PATH = ROOT / "outputs/modelling/ml/ml_model_summary.json"
ML_PERF_PATH = ROOT / "outputs/modelling/ml/table_4_5_machine_learning_performance_full_precision.csv"

KEY = "LSOA21CD"
RESPONSE = "log1p_employees_count"
PREDICTORS = [
    "mobility_inflow_total",
    "flow_entropy",
    "weekday_weekend_ratio",
    "ptal_ai_mean",
    "population_density_km2",
    "amenity_poi_density_km2",
    "retail_poi_density_km2",
]
RANDOM_SEED = 20260722
N_SPATIAL_FOLDS = 5
TEST_FOLD = 0


def label_spatial_folds(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    centroids = gdf.geometry.centroid
    coords = np.column_stack([centroids.x.to_numpy(), centroids.y.to_numpy()])
    raw_labels = KMeans(n_clusters=N_SPATIAL_FOLDS, random_state=RANDOM_SEED, n_init=50).fit_predict(coords)
    centers = (
        pd.DataFrame(coords, columns=["x", "y"])
        .assign(raw_label=raw_labels)
        .groupby("raw_label")[["x", "y"]]
        .mean()
    )
    centers["sort_key"] = centers["x"] + centers["y"]
    label_map = {raw: i for i, raw in enumerate(centers.sort_values("sort_key").index)}
    labels = np.array([label_map[label] for label in raw_labels])
    return pd.DataFrame({KEY: gdf[KEY].to_numpy(), "spatial_fold": labels})


def metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "test_r_squared": float(r2_score(y_true, y_pred)),
        "test_rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "test_mae": float(mean_absolute_error(y_true, y_pred)),
    }


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    gdf = gpd.read_file(GEOMETRY_PATH)
    ml_summary = json.loads(ML_SUMMARY_PATH.read_text(encoding="utf-8"))
    ml_perf = pd.read_csv(ML_PERF_PATH)

    fold_df = label_spatial_folds(gdf[[KEY, "geometry"]])
    data = df.merge(fold_df, on=KEY, how="left", validate="one_to_one")

    expected_fold_counts = {int(k): int(v) for k, v in ml_summary["validation"]["fold_counts"].items()}
    actual_fold_counts = data["spatial_fold"].value_counts().sort_index().astype(int).to_dict()
    if actual_fold_counts != expected_fold_counts:
        raise RuntimeError(f"Reconstructed fold counts differ from ML summary: {actual_fold_counts} vs {expected_fold_counts}")
    if int(ml_summary["validation"]["held_out_test_fold"]) != TEST_FOLD:
        raise RuntimeError("ML summary held-out test fold differs from expected TEST_FOLD=0.")

    train_idx = data["spatial_fold"] != TEST_FOLD
    test_idx = data["spatial_fold"] == TEST_FOLD
    X_train = sm.add_constant(data.loc[train_idx, PREDICTORS], has_constant="add")
    y_train = data.loc[train_idx, RESPONSE]
    X_test = sm.add_constant(data.loc[test_idx, PREDICTORS], has_constant="add")
    y_test = data.loc[test_idx, RESPONSE]

    model = sm.OLS(y_train, X_train).fit()
    test_pred = model.predict(X_test)
    train_pred = model.predict(X_train)
    ols_test = metrics(y_test, test_pred)
    ols_train = metrics(y_train, train_pred)

    ols_row = {
        "Model": "OLS benchmark",
        "Spatial CV R2": np.nan,
        "Test R2": ols_test["test_r_squared"],
        "RMSE": ols_test["test_rmse"],
        "MAE": ols_test["test_mae"],
        "Best Model": np.nan,
        "evaluation_basis": "same held-out spatial test fold as existing ML",
    }
    ml_rows = ml_perf.copy()
    ml_rows["evaluation_basis"] = "existing held-out spatial test fold from authoritative ML output"
    comparison = pd.concat([pd.DataFrame([ols_row]), ml_rows], ignore_index=True)

    partition = data[[KEY, "spatial_fold"]].copy()
    partition["partition"] = np.where(partition["spatial_fold"] == TEST_FOLD, "test", "train")
    predictions = pd.DataFrame(
        {
            KEY: data.loc[test_idx, KEY].to_numpy(),
            "spatial_fold": data.loc[test_idx, "spatial_fold"].to_numpy(),
            "observed": y_test.to_numpy(),
            "predicted_ols": test_pred.to_numpy(),
            "residual_ols": y_test.to_numpy() - test_pred.to_numpy(),
        }
    )
    coefficient_table = pd.DataFrame(
        {
            "variable": X_train.columns,
            "coefficient": model.params,
            "standard_error": model.bse,
            "t_statistic": model.tvalues,
            "p_value": model.pvalues,
        }
    )

    sar_sem_feasibility = {
        "status": "not_run",
        "reason": "Hard constraint says only record feasibility; do not extend experiment.",
        "feasibility": "Possible in principle only if SAR/SEM are re-estimated on training rows or a frozen training-fit spatial model exists. Current authoritative SAR/SEM were fitted on all 4,994 rows, so applying the same held-out protocol would require new spatial model estimation and careful train/test weights design.",
    }

    qa = {
        "uses_existing_ml_split_logic": True,
        "no_new_random_split_created": True,
        "random_seed": RANDOM_SEED,
        "n_spatial_folds": N_SPATIAL_FOLDS,
        "test_fold": TEST_FOLD,
        "fold_counts_match_ml_summary": actual_fold_counts == expected_fold_counts,
        "train_n": int(train_idx.sum()),
        "test_n": int(test_idx.sum()),
        "expected_train_n": 3916,
        "expected_test_n": 1078,
        "train_test_counts_match_existing_ml": int(train_idx.sum()) == 3916 and int(test_idx.sum()) == 1078,
    }

    comparison.to_csv(OUT / "heldout_ols_rf_xgboost_comparison.csv", index=False)
    partition.to_csv(OUT / "ml_spatial_partition_reconstructed.csv", index=False)
    predictions.to_csv(OUT / "heldout_ols_predictions.csv", index=False)
    coefficient_table.to_csv(OUT / "heldout_ols_train_coefficients.csv", index=False)
    (OUT / "sar_sem_heldout_feasibility.json").write_text(json.dumps(sar_sem_feasibility, indent=2), encoding="utf-8")
    (OUT / "source_manifest.json").write_text(
        json.dumps(
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "purpose": "E4 comparable held-out predictive evaluation",
                "authoritative_inputs": {
                    "dataset": str(DATA_PATH.relative_to(ROOT)),
                    "geometry": str(GEOMETRY_PATH.relative_to(ROOT)),
                    "ml_summary": str(ML_SUMMARY_PATH.relative_to(ROOT)),
                    "ml_performance": str(ML_PERF_PATH.relative_to(ROOT)),
                },
                "response": RESPONSE,
                "predictors": PREDICTORS,
                "qa": qa,
                "ols_train_metrics": ols_train,
                "ols_test_metrics": ols_test,
                "environment": {
                    "python_executable": sys.executable,
                    "python_version": sys.version,
                    "platform": platform.platform(),
                    "pandas": pd.__version__,
                    "geopandas": gpd.__version__,
                    "statsmodels": sm.__version__,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("E4 complete")
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
