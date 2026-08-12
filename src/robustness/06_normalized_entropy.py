#!/usr/bin/env python3
"""E8 normalized entropy robustness for OLS, ML, and SHAP."""

from __future__ import annotations

import json
import pickle
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import statsmodels.api as sm
import xgboost
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data/processed/master/master_analysis_dataset_final_v1_0.csv"
GEOMETRY_PATH = ROOT / "data/processed/master/master_analysis_dataset_final_v1_0.gpkg"
ORIGINAL_OLS = ROOT / "outputs/modelling/ols/ols_model_metrics.csv"
ORIGINAL_COEFFICIENTS = ROOT / "outputs/modelling/ols/table_4_3_ols_results_full_precision.csv"
ML_SUMMARY_PATH = ROOT / "outputs/modelling/ml/ml_model_summary.json"
ML_PERF_PATH = ROOT / "outputs/modelling/ml/table_4_5_machine_learning_performance_full_precision.csv"
RAW_IMPORTANCE_PATH = ROOT / "outputs/modelling/ml/feature_importance.csv"
RAW_SHAP_PATH = ROOT / "outputs/modelling/ml/shap_values.csv"

KEY = "LSOA21CD"
RESPONSE = "log1p_employees_count"
RANDOM_SEED = 20260722
N_SPATIAL_FOLDS = 5
TEST_FOLD = 0
PERMUTATION_REPEATS = 30
PREDICTORS_NORM = [
    "mobility_inflow_total",
    "normalized_flow_entropy",
    "weekday_weekend_ratio",
    "ptal_ai_mean",
    "population_density_km2",
    "amenity_poi_density_km2",
    "retail_poi_density_km2",
]
PREDICTORS_RAW = [
    "mobility_inflow_total",
    "flow_entropy",
    "weekday_weekend_ratio",
    "ptal_ai_mean",
    "population_density_km2",
    "amenity_poi_density_km2",
    "retail_poi_density_km2",
]


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "r_squared": float(r2_score(y_true, y_pred)),
        "rmse": rmse(y_true, y_pred),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def label_spatial_folds(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    centroids = gdf.geometry.centroid
    coords = np.column_stack([centroids.x.to_numpy(), centroids.y.to_numpy()])
    raw_labels = KMeans(n_clusters=N_SPATIAL_FOLDS, random_state=RANDOM_SEED, n_init=50).fit_predict(coords)
    centers = pd.DataFrame(coords, columns=["x", "y"]).assign(raw_label=raw_labels).groupby("raw_label")[["x", "y"]].mean()
    centers["sort_key"] = centers["x"] + centers["y"]
    label_map = {raw: i for i, raw in enumerate(centers.sort_values("sort_key").index)}
    labels = np.array([label_map[label] for label in raw_labels])
    return pd.DataFrame({KEY: gdf[KEY].to_numpy(), "spatial_fold": labels})


def fit_models() -> dict[str, object]:
    return {
        "Random Forest": RandomForestRegressor(
            n_estimators=500,
            max_depth=None,
            min_samples_leaf=1,
            max_features=1.0,
            bootstrap=True,
            random_state=RANDOM_SEED,
            n_jobs=1,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=RANDOM_SEED,
            n_jobs=1,
            tree_method="hist",
        ),
    }


def spatial_cv_score(estimator, X: pd.DataFrame, y: pd.Series, fold_ids: pd.Series) -> dict[str, object]:
    rows = []
    for fold in sorted(fold_ids.unique()):
        train_idx = fold_ids != fold
        valid_idx = fold_ids == fold
        model = pickle.loads(pickle.dumps(estimator))
        model.fit(X.loc[train_idx, :], y.loc[train_idx])
        pred = model.predict(X.loc[valid_idx, :])
        row = metrics(y.loc[valid_idx].to_numpy(), pred)
        row["fold"] = int(fold)
        row["n_validation"] = int(valid_idx.sum())
        rows.append(row)
    result = pd.DataFrame(rows)
    return {
        "spatial_cv_r_squared": float(result["r_squared"].mean()),
        "spatial_cv_rmse": float(result["rmse"].mean()),
        "spatial_cv_mae": float(result["mae"].mean()),
        "fold_results": rows,
    }


def fit_ols(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    y = df[RESPONSE]
    X = sm.add_constant(df[PREDICTORS_NORM], has_constant="add")
    model = sm.OLS(y, X).fit()
    robust = model.get_robustcov_results(cov_type="HC3")
    pred = model.predict(X)
    y_sd = y.std(ddof=0)
    coeffs = pd.DataFrame(
        {
            "variable": X.columns,
            "coefficient": model.params,
            "hc3_standard_error": robust.bse,
            "hc3_t": robust.tvalues,
            "hc3_p_value": robust.pvalues,
            "standardised_beta": [np.nan] + [float(model.params[col] * X[col].std(ddof=0) / y_sd) for col in X.columns if col != "const"],
        }
    )
    metric = {
        "n": int(model.nobs),
        "r_squared": float(model.rsquared),
        "adjusted_r_squared": float(model.rsquared_adj),
        "rmse": rmse(y, pred),
        "mae": float(mean_absolute_error(y, pred)),
        "aic": float(model.aic),
        "bic": float(model.bic),
        "log_likelihood": float(model.llf),
    }
    return coeffs, metric


def classify_shape(x: np.ndarray, y: np.ndarray, binned: pd.DataFrame) -> tuple[str, str]:
    spearman = stats.spearmanr(x, y).statistic
    pearson = stats.pearsonr(x, y).statistic
    shap_range = float(np.nanpercentile(y, 95) - np.nanpercentile(y, 5))
    if shap_range < 0.30 or abs(spearman) < 0.30:
        return "weak / unclear", "Small rank association or narrow SHAP contribution spread."
    means = binned["mean_shap"].to_numpy()
    slopes = np.diff(means)
    dominant_share = float(np.mean(np.sign(slopes) == np.sign(spearman))) if len(slopes) else 0.0
    tolerance = max(0.02, 0.08 * np.nanstd(means))
    crosses_zero = bool(np.nanmin(means) < 0 < np.nanmax(means))
    if abs(spearman) >= 0.90 and dominant_share >= 0.80:
        return "monotonic nonlinear", "Very strong rank association with a mostly one-direction binned profile."
    if crosses_zero and abs(spearman) >= 0.55:
        first_positive = np.where(means > 0)[0]
        if len(first_positive) and np.all(means[first_positive[0] :] > -tolerance):
            return "threshold", "Binned SHAP profile shifts into sustained positive contribution."
    if abs(spearman) >= 0.55 and abs(abs(spearman) - abs(pearson)) < 0.15:
        return "approximately linear", "Pearson and Spearman associations are both strong and similar."
    if abs(spearman) >= 0.45:
        return "monotonic nonlinear", "Clear rank association but not close to linear."
    return "weak / unclear", "Pattern does not meet conservative thresholds."


def shap_analysis(data: pd.DataFrame, shap_values: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
    rows = []
    shap_df = pd.DataFrame(shap_values, columns=[f"shap_{c}" for c in feature_names])
    joined = pd.concat([data[[KEY, *feature_names]].reset_index(drop=True), shap_df], axis=1)
    joined.to_csv(OUT / "normalized_entropy_shap_feature_joined_data.csv", index=False)
    for feature in ["normalized_flow_entropy", "mobility_inflow_total", "weekday_weekend_ratio"]:
        shap_col = f"shap_{feature}"
        work = joined[[KEY, feature, shap_col]].copy()
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
            )
            .reset_index(drop=True)
        )
        label, rationale = classify_shape(work[feature].to_numpy(), work[shap_col].to_numpy(), binned)
        binned.to_csv(OUT / f"{feature}_normalized_model_shap_binned_profile.csv", index=False)
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        ax.scatter(work[feature], work[shap_col], s=7, alpha=0.2, linewidths=0, color="#366a96")
        ax.plot(binned["median_feature"], binned["mean_shap"], color="#b23a48", linewidth=2)
        ax.axhline(0, color="#555", linewidth=0.8)
        ax.set_xlabel(feature)
        ax.set_ylabel(f"SHAP value: {feature}")
        ax.set_title(f"Normalized-entropy model SHAP: {feature}")
        fig.tight_layout()
        fig.savefig(OUT / f"{feature}_normalized_model_shap_dependence.png", dpi=180)
        plt.close(fig)
        rows.append(
            {
                "feature": feature,
                "mean_abs_shap": float(np.abs(work[shap_col]).mean()),
                "spearman_rho": float(stats.spearmanr(work[feature], work[shap_col]).statistic),
                "pearson_r": float(stats.pearsonr(work[feature], work[shap_col]).statistic),
                "shap_p05": float(np.percentile(work[shap_col], 5)),
                "shap_p50": float(np.percentile(work[shap_col], 50)),
                "shap_p95": float(np.percentile(work[shap_col], 95)),
                "shape_classification": label,
                "rationale": rationale,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    gdf = gpd.read_file(GEOMETRY_PATH)
    if list(df[KEY]) != list(gdf[KEY]):
        raise RuntimeError("CSV/GPKG order mismatch.")
    if (df["unique_origins"] <= 1).any():
        df["normalized_flow_entropy"] = np.where(df["unique_origins"] > 1, df["flow_entropy"] / np.log(df["unique_origins"]), np.nan)
    else:
        df["normalized_flow_entropy"] = df["flow_entropy"] / np.log(df["unique_origins"])
    if df["normalized_flow_entropy"].isna().any():
        raise RuntimeError("Normalized entropy contains missing values.")

    qa = {
        "n": len(df),
        "unique_lsoa": int(df[KEY].nunique()),
        "k_min": float(df["unique_origins"].min()),
        "k_median": float(df["unique_origins"].median()),
        "k_max": float(df["unique_origins"].max()),
        "k_le_1_count": int((df["unique_origins"] <= 1).sum()),
        "normalized_entropy_min": float(df["normalized_flow_entropy"].min()),
        "normalized_entropy_median": float(df["normalized_flow_entropy"].median()),
        "normalized_entropy_max": float(df["normalized_flow_entropy"].max()),
        "raw_entropy_vs_normalized_correlation": float(df["flow_entropy"].corr(df["normalized_flow_entropy"])),
        "normalized_entropy_vs_mobility_inflow_correlation": float(df["normalized_flow_entropy"].corr(df["mobility_inflow_total"])),
        "raw_entropy_vs_mobility_inflow_correlation": float(df["flow_entropy"].corr(df["mobility_inflow_total"])),
    }
    pd.DataFrame([qa]).to_csv(OUT / "normalized_entropy_qa_summary.csv", index=False)
    df[[KEY, "flow_entropy", "unique_origins", "normalized_flow_entropy", "mobility_inflow_total"]].to_csv(OUT / "normalized_entropy_values.csv", index=False)

    coeffs, ols_metrics = fit_ols(df)
    coeffs.to_csv(OUT / "normalized_entropy_ols_coefficients.csv", index=False)
    pd.DataFrame([ols_metrics]).to_csv(OUT / "normalized_entropy_ols_metrics.csv", index=False)

    original_metrics = pd.read_csv(ORIGINAL_OLS).iloc[0]
    original_coeffs = pd.read_csv(ORIGINAL_COEFFICIENTS)
    raw_entropy_row = original_coeffs[original_coeffs["source_column"] == "flow_entropy"].iloc[0]
    norm_entropy_row = coeffs[coeffs["variable"] == "normalized_flow_entropy"].iloc[0]
    ols_comparison = pd.DataFrame(
        [
            {
                "specification": "primary_raw_entropy",
                "entropy_variable": "flow_entropy",
                "entropy_coefficient": float(raw_entropy_row["Coefficient"]),
                "entropy_p_value": float(raw_entropy_row["Robust p"]),
                "r_squared": float(original_metrics["r_squared"]),
                "adjusted_r_squared": float(original_metrics["adjusted_r_squared"]),
                "rmse": float(original_metrics["rmse"]),
                "mae": float(original_metrics["mae"]),
                "aic": float(original_metrics["aic"]),
            },
            {
                "specification": "E8_normalized_entropy",
                "entropy_variable": "normalized_flow_entropy",
                "entropy_coefficient": float(norm_entropy_row["coefficient"]),
                "entropy_p_value": float(norm_entropy_row["hc3_p_value"]),
                "entropy_standardised_beta": float(norm_entropy_row["standardised_beta"]),
                **{k: ols_metrics[k] for k in ["r_squared", "adjusted_r_squared", "rmse", "mae", "aic"]},
            },
        ]
    )
    ols_comparison.to_csv(OUT / "raw_vs_normalized_entropy_ols_comparison.csv", index=False)

    ml_summary = json.loads(ML_SUMMARY_PATH.read_text(encoding="utf-8"))
    fold_df = label_spatial_folds(gdf[[KEY, "geometry"]])
    data = df.merge(fold_df, on=KEY, how="left", validate="one_to_one")
    expected_fold_counts = {int(k): int(v) for k, v in ml_summary["validation"]["fold_counts"].items()}
    actual_fold_counts = data["spatial_fold"].value_counts().sort_index().astype(int).to_dict()
    if actual_fold_counts != expected_fold_counts:
        raise RuntimeError("Cannot guarantee identical ML fold protocol.")

    X = data[PREDICTORS_NORM]
    y = data[RESPONSE]
    folds = data["spatial_fold"].astype(int)
    train_idx = folds != TEST_FOLD
    test_idx = folds == TEST_FOLD
    model_results = {}
    fitted = {}
    importance_rows = []
    for model_name, estimator in fit_models().items():
        cv = spatial_cv_score(estimator, X, y, folds)
        model = pickle.loads(pickle.dumps(estimator))
        model.fit(X.loc[train_idx, :], y.loc[train_idx])
        pred = model.predict(X.loc[test_idx, :])
        test = metrics(y.loc[test_idx].to_numpy(), pred)
        model_results[model_name] = {
            **cv,
            "test_r_squared": test["r_squared"],
            "test_rmse": test["rmse"],
            "test_mae": test["mae"],
            "train_n": int(train_idx.sum()),
            "test_n": int(test_idx.sum()),
            "test_fold": TEST_FOLD,
        }
        fitted[model_name] = model
        perm = permutation_importance(model, X.loc[test_idx], y.loc[test_idx], n_repeats=PERMUTATION_REPEATS, random_state=RANDOM_SEED, n_jobs=1, scoring="r2")
        for variable, mean_imp, std_imp in zip(PREDICTORS_NORM, perm.importances_mean, perm.importances_std):
            importance_rows.append({"model": model_name, "variable": variable, "mean_importance": float(mean_imp), "std_importance": float(std_imp), "n_repeats": PERMUTATION_REPEATS, "random_seed": RANDOM_SEED, "scoring": "r2"})

    ml_perf = pd.DataFrame(
        [
            {
                "Model": model,
                "Spatial CV R2": values["spatial_cv_r_squared"],
                "Test R2": values["test_r_squared"],
                "RMSE": values["test_rmse"],
                "MAE": values["test_mae"],
                "Best Model": "Yes" if model == max(model_results, key=lambda name: model_results[name]["test_r_squared"]) else "",
            }
            for model, values in model_results.items()
        ]
    )
    ml_perf.to_csv(OUT / "normalized_entropy_ml_performance.csv", index=False)
    pd.DataFrame([r for values in model_results.values() for r in values["fold_results"]]).to_csv(OUT / "normalized_entropy_ml_spatial_cv_fold_results.csv", index=False)
    importance = pd.DataFrame(importance_rows)
    importance["rank_within_model"] = importance.groupby("model")["mean_importance"].rank(method="first", ascending=False).astype(int)
    importance.to_csv(OUT / "normalized_entropy_feature_importance.csv", index=False)

    raw_ml_perf = pd.read_csv(ML_PERF_PATH)
    raw_importance = pd.read_csv(RAW_IMPORTANCE_PATH)
    raw_shap = pd.read_csv(RAW_SHAP_PATH)
    contribution_values = fitted["XGBoost"].get_booster().predict(xgboost.DMatrix(X), pred_contribs=True)
    norm_shap_values = contribution_values[:, :-1]
    norm_shap = shap_analysis(data.reset_index(drop=True), norm_shap_values, PREDICTORS_NORM)
    norm_shap.to_csv(OUT / "normalized_entropy_shap_summary.csv", index=False)
    raw_entropy_mean_abs_shap = float(raw_shap["shap_flow_entropy"].abs().mean())
    norm_entropy_mean_abs_shap = float(norm_shap.loc[norm_shap["feature"] == "normalized_flow_entropy", "mean_abs_shap"].iloc[0])
    entropy_ml_comparison = pd.DataFrame(
        [
            {
                "basis": "raw_entropy_existing_XGBoost",
                "test_r2": float(raw_ml_perf.loc[raw_ml_perf["Model"] == "XGBoost", "Test R2"].iloc[0]),
                "rmse": float(raw_ml_perf.loc[raw_ml_perf["Model"] == "XGBoost", "RMSE"].iloc[0]),
                "entropy_importance": float(raw_importance[(raw_importance["model"] == "XGBoost") & (raw_importance["variable"] == "flow_entropy")]["mean_importance"].iloc[0]),
                "entropy_importance_rank": int(raw_importance[(raw_importance["model"] == "XGBoost") & (raw_importance["variable"] == "flow_entropy")]["rank_within_model"].iloc[0]),
                "entropy_mean_abs_shap": raw_entropy_mean_abs_shap,
                "entropy_shape": "weak / unclear",
            },
            {
                "basis": "normalized_entropy_E8_XGBoost",
                "test_r2": float(ml_perf.loc[ml_perf["Model"] == "XGBoost", "Test R2"].iloc[0]),
                "rmse": float(ml_perf.loc[ml_perf["Model"] == "XGBoost", "RMSE"].iloc[0]),
                "entropy_importance": float(importance[(importance["model"] == "XGBoost") & (importance["variable"] == "normalized_flow_entropy")]["mean_importance"].iloc[0]),
                "entropy_importance_rank": int(importance[(importance["model"] == "XGBoost") & (importance["variable"] == "normalized_flow_entropy")]["rank_within_model"].iloc[0]),
                "entropy_mean_abs_shap": norm_entropy_mean_abs_shap,
                "entropy_shape": str(norm_shap.loc[norm_shap["feature"] == "normalized_flow_entropy", "shape_classification"].iloc[0]),
            },
        ]
    )
    entropy_ml_comparison.to_csv(OUT / "raw_vs_normalized_entropy_ml_shap_comparison.csv", index=False)

    data[[KEY, "spatial_fold"]].assign(partition=np.where(data["spatial_fold"] == TEST_FOLD, "test", "train")).to_csv(OUT / "ml_spatial_partition_reconstructed.csv", index=False)
    (OUT / "source_manifest.json").write_text(
        json.dumps(
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "purpose": "E8 normalized entropy robustness",
                "authoritative_inputs": {
                    "dataset": str(DATA_PATH.relative_to(ROOT)),
                    "geometry": str(GEOMETRY_PATH.relative_to(ROOT)),
                    "ml_summary": str(ML_SUMMARY_PATH.relative_to(ROOT)),
                    "raw_ml_performance": str(ML_PERF_PATH.relative_to(ROOT)),
                    "raw_feature_importance": str(RAW_IMPORTANCE_PATH.relative_to(ROOT)),
                    "raw_shap_values": str(RAW_SHAP_PATH.relative_to(ROOT)),
                },
                "qa": {
                    **qa,
                    "fold_counts_match_existing_ml": actual_fold_counts == expected_fold_counts,
                    "same_random_seed": RANDOM_SEED,
                    "same_hyperparameters": True,
                    "same_train_test_lsoa_protocol": True,
                    "train_n": int(train_idx.sum()),
                    "test_n": int(test_idx.sum()),
                },
                "environment": {
                    "python_executable": sys.executable,
                    "python_version": sys.version,
                    "platform": platform.platform(),
                    "pandas": pd.__version__,
                    "numpy": np.__version__,
                    "statsmodels": sm.__version__,
                    "sklearn": sklearn.__version__,
                    "geopandas": gpd.__version__,
                    "xgboost": xgboost.__version__,
                    "matplotlib": matplotlib.__version__,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("E8 complete")
    print(pd.DataFrame([qa]).to_string(index=False))
    print(ols_comparison.to_string(index=False))
    print(ml_perf.to_string(index=False))
    print(entropy_ml_comparison.to_string(index=False))


if __name__ == "__main__":
    main()
