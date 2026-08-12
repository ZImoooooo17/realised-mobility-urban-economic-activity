#!/usr/bin/env python3
"""Sprint 11 machine-learning models for the frozen Chapter 4 specification."""

from __future__ import annotations

import json
import pickle
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

try:
    import shap
except ModuleNotFoundError:
    shap = None


ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data/processed/master/master_analysis_dataset_final_v1_0.csv"
GEOMETRY_PATH = ROOT / "data/processed/master/master_analysis_dataset_final_v1_0.gpkg"
OLS_METRICS_PATH = ROOT / "outputs/modelling/ols/ols_model_metrics.csv"
SEM_METRICS_PATH = ROOT / "outputs/modelling/spatial/sem_model_metrics.csv"
OUTPUT_DIR = ROOT / "outputs/modelling/ml"
WRITING_MAIN_DIR = ROOT / "writing/tables/main"
WRITING_APPENDIX_DIR = ROOT / "writing/tables/appendix"
MANIFEST_PATH = ROOT / "writing/manifests/table_manifest.csv"
RESULTS_BOOK_PATH = ROOT / "writing/chapter4/CHAPTER4_RESULTS_BOOK.md"

RANDOM_SEED = 20260722
EXPECTED_N = 4994
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
KEY = "LSOA21CD"
N_SPATIAL_FOLDS = 5
TEST_FOLD = 0
PERMUTATION_REPEATS = 30


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
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


def spatial_cv_score(estimator, X: pd.DataFrame, y: pd.Series, fold_ids: pd.Series) -> dict[str, float]:
    rows = []
    for fold in sorted(fold_ids.unique()):
        train_idx = fold_ids != fold
        valid_idx = fold_ids == fold
        model = pickle.loads(pickle.dumps(estimator))
        model.fit(X.loc[train_idx, :], y.loc[train_idx])
        pred = model.predict(X.loc[valid_idx, :])
        fold_metrics = metrics(y.loc[valid_idx].to_numpy(), pred)
        fold_metrics["fold"] = int(fold)
        fold_metrics["n_validation"] = int(valid_idx.sum())
        rows.append(fold_metrics)
    result = pd.DataFrame(rows)
    return {
        "spatial_cv_r_squared": float(result["r_squared"].mean()),
        "spatial_cv_rmse": float(result["rmse"].mean()),
        "spatial_cv_mae": float(result["mae"].mean()),
        "spatial_cv_fold_results": rows,
    }


def latex_table(df: pd.DataFrame, caption: str, label: str, path: Path) -> None:
    path.write_text(
        df.to_latex(index=False, escape=True, caption=caption, label=label, na_rep="--"),
        encoding="utf-8",
    )


def update_manifest() -> None:
    manifest = pd.read_csv(MANIFEST_PATH)
    updates = {
        "Table 4.5": {
            "title": "Machine learning performance comparison",
            "chapter_section": "4.6 Random Forest",
            "publication_status": "Main",
            "final_or_source_asset": "final",
            "main_or_appendix": "Main",
            "target_publication_asset": "",
            "source_path": "outputs/modelling/ml/table_4_5_machine_learning_performance_full_precision.csv",
            "publication_path": "writing/tables/main/table_4_5_machine_learning_performance.csv",
            "latex_path": "writing/tables/main/table_4_5_machine_learning_performance.tex",
            "notes": "Frozen main dissertation table generated during Sprint 11; no OLS or spatial models rerun.",
        },
        "Table A5": {
            "title": "Machine learning robustness checks",
            "chapter_section": "Appendix",
            "publication_status": "Appendix",
            "final_or_source_asset": "final",
            "main_or_appendix": "Appendix",
            "target_publication_asset": "",
            "source_path": "outputs/modelling/ml/table_A5_machine_learning_robustness.csv",
            "publication_path": "writing/tables/appendix/table_A5_machine_learning_robustness.csv",
            "latex_path": "writing/tables/appendix/table_A5_machine_learning_robustness.tex",
            "notes": "Frozen appendix table generated during Sprint 11; records ML validation settings and hyperparameters.",
        },
    }
    for number, values in updates.items():
        mask = manifest["dissertation_number"] == number
        if not mask.any():
            values = {"asset_id": number.lower().replace(" ", "_").replace(".", ""), "dissertation_number": number, **values}
            manifest = pd.concat([manifest, pd.DataFrame([values])], ignore_index=True)
            continue
        for col, val in values.items():
            manifest.loc[mask, col] = val
    manifest.to_csv(MANIFEST_PATH, index=False)


def update_results_book(summary: dict[str, object]) -> None:
    feature_lines = "\n".join(
        f"- {row['rank']}. `{row['variable']}`: {row['mean_importance']:.6f}"
        for row in summary["top_five_importance"]
    )
    text = f"""

## Sprint 11 — Machine Learning Models

- Frozen response: `{RESPONSE}`
- Frozen predictors: `{', '.join(PREDICTORS)}`
- Sample size: {EXPECTED_N}
- Spatial CV strategy: {N_SPATIAL_FOLDS}-fold KMeans spatial blocks from frozen LSOA geometry centroids; held-out test fold `{TEST_FOLD}`; random seed `{RANDOM_SEED}`
- Best ML model: {summary['best_model']}
- Random Forest: Spatial CV R2 {summary['models']['Random Forest']['spatial_cv_r_squared']:.6f}; Test R2 {summary['models']['Random Forest']['test_r_squared']:.6f}; RMSE {summary['models']['Random Forest']['test_rmse']:.6f}; MAE {summary['models']['Random Forest']['test_mae']:.6f}
- XGBoost: Spatial CV R2 {summary['models']['XGBoost']['spatial_cv_r_squared']:.6f}; Test R2 {summary['models']['XGBoost']['test_r_squared']:.6f}; RMSE {summary['models']['XGBoost']['test_rmse']:.6f}; MAE {summary['models']['XGBoost']['test_mae']:.6f}
- OLS comparison RMSE: {summary['benchmark_models']['OLS']['rmse']:.6f}
- SEM comparison RMSE: {summary['benchmark_models']['SEM']['rmse']:.6f}
- Feature importance summary for best model:
{feature_lines}
- SHAP values generated: {summary['shap_values_generated']}
- Table 4.5: `outputs/modelling/ml/table_4_5_machine_learning_performance.csv`
- Table A5: `outputs/modelling/ml/table_A5_machine_learning_robustness.csv`
- Chapter 4 analysis complete: YES
"""
    existing = RESULTS_BOOK_PATH.read_text(encoding="utf-8")
    marker = "\n## Sprint 11 — Machine Learning Models\n"
    if marker in existing:
        existing = existing.split(marker)[0].rstrip() + text
    else:
        existing = existing.rstrip() + text
    RESULTS_BOOK_PATH.write_text(existing.rstrip() + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    WRITING_MAIN_DIR.mkdir(parents=True, exist_ok=True)
    WRITING_APPENDIX_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    gdf = gpd.read_file(GEOMETRY_PATH)
    fold_df = label_spatial_folds(gdf[[KEY, "geometry"]])
    data = df.merge(fold_df, on=KEY, how="left", validate="one_to_one")
    data["log1p_employees_count_check"] = np.log1p(data["employees_count"])

    qa = {
        "same_response_variable": RESPONSE in data.columns,
        "same_predictors": all(col in data.columns for col in PREDICTORS),
        "same_sample_size": len(data) == EXPECTED_N,
        "no_missing_predictors": int(data[PREDICTORS].isna().sum().sum()) == 0,
        "no_missing_response": int(data[RESPONSE].isna().sum()) == 0,
        "no_duplicate_lsoas": int(data[KEY].duplicated().sum()) == 0,
        "response_matches_log1p_employees_count": bool(np.allclose(data[RESPONSE], data["log1p_employees_count_check"])),
        "spatial_folds_complete": int(data["spatial_fold"].isna().sum()) == 0,
        "reproducible_random_seed": RANDOM_SEED,
    }
    qa_pass = all(value is True for key, value in qa.items() if key != "reproducible_random_seed")
    if not qa_pass:
        raise RuntimeError(f"QA failed before modelling: {qa}")

    X = data[PREDICTORS]
    y = data[RESPONSE]
    folds = data["spatial_fold"].astype(int)
    train_idx = folds != TEST_FOLD
    test_idx = folds == TEST_FOLD

    model_results: dict[str, dict[str, object]] = {}
    fitted: dict[str, object] = {}
    importance_rows = []
    models = fit_models()
    for model_name, estimator in models.items():
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
            "hyperparameters": model.get_params(),
        }
        fitted[model_name] = model
        perm = permutation_importance(
            model,
            X.loc[test_idx, :],
            y.loc[test_idx],
            n_repeats=PERMUTATION_REPEATS,
            random_state=RANDOM_SEED,
            n_jobs=1,
            scoring="r2",
        )
        for variable, mean_imp, std_imp in zip(PREDICTORS, perm.importances_mean, perm.importances_std):
            importance_rows.append(
                {
                    "model": model_name,
                    "variable": variable,
                    "mean_importance": float(mean_imp),
                    "std_importance": float(std_imp),
                    "n_repeats": PERMUTATION_REPEATS,
                    "random_seed": RANDOM_SEED,
                    "scoring": "r2",
                }
            )

    best_model = max(model_results, key=lambda name: model_results[name]["test_r_squared"])
    table_full = pd.DataFrame(
        [
            {
                "Model": model,
                "Spatial CV R2": values["spatial_cv_r_squared"],
                "Test R2": values["test_r_squared"],
                "RMSE": values["test_rmse"],
                "MAE": values["test_mae"],
                "Best Model": "Yes" if model == best_model else "",
            }
            for model, values in model_results.items()
        ]
    )
    table_publication = table_full.copy()
    for col in ["Spatial CV R2", "Test R2", "RMSE", "MAE"]:
        table_publication[col] = table_publication[col].map(lambda value: f"{value:.3f}")

    robustness_rows = []
    for model, values in model_results.items():
        params = values["hyperparameters"]
        robustness_rows.append(
            {
                "Model": model,
                "Spatial CV R2": values["spatial_cv_r_squared"],
                "Test R2": values["test_r_squared"],
                "RMSE": values["test_rmse"],
                "MAE": values["test_mae"],
                "Random Seed": RANDOM_SEED,
                "Number of Trees": params.get("n_estimators"),
                "Learning Rate": "" if model == "Random Forest" else params.get("learning_rate", ""),
                "Maximum Tree Depth": "None" if params.get("max_depth", "") is None else params.get("max_depth", ""),
                "Spatial CV Strategy": f"{N_SPATIAL_FOLDS}-fold KMeans spatial blocks using frozen geometry centroids",
                "Held-out Test Fold": TEST_FOLD,
                "Train n": values["train_n"],
                "Test n": values["test_n"],
            }
        )
    robustness = pd.DataFrame(robustness_rows)
    robustness_publication = robustness.copy()
    for col in ["Spatial CV R2", "Test R2", "RMSE", "MAE"]:
        robustness_publication[col] = robustness_publication[col].map(lambda value: f"{value:.6f}")

    importance = pd.DataFrame(importance_rows)
    importance["rank_within_model"] = importance.groupby("model")["mean_importance"].rank(
        method="first", ascending=False
    ).astype(int)
    best_importance = (
        importance[importance["model"] == best_model]
        .sort_values("mean_importance", ascending=False)
        .head(5)
        .reset_index(drop=True)
    )
    top_five = [
        {
            "rank": int(i + 1),
            "variable": row["variable"],
            "mean_importance": float(row["mean_importance"]),
            "std_importance": float(row["std_importance"]),
        }
        for i, row in best_importance.iterrows()
    ]

    shap_method = "shap.TreeExplainer"
    shap_model = best_model
    if shap is not None:
        explainer = shap.TreeExplainer(fitted[best_model])
        shap_values = explainer.shap_values(X)
    elif best_model == "XGBoost":
        shap_method = "xgboost.Booster.predict(pred_contribs=True)"
        contribution_values = fitted["XGBoost"].get_booster().predict(xgboost.DMatrix(X), pred_contribs=True)
        shap_values = contribution_values[:, :-1]
    else:
        shap_model = "XGBoost"
        shap_method = "xgboost.Booster.predict(pred_contribs=True); fallback because shap package dependency numba is unavailable for Random Forest"
        contribution_values = fitted["XGBoost"].get_booster().predict(xgboost.DMatrix(X), pred_contribs=True)
        shap_values = contribution_values[:, :-1]
    shap_df = pd.DataFrame(shap_values, columns=[f"shap_{col}" for col in PREDICTORS])
    shap_df.insert(0, KEY, data[KEY])
    shap_df.insert(1, "model", shap_model)
    shap_csv_path = OUTPUT_DIR / "shap_values.csv"
    shap_parquet_path = OUTPUT_DIR / "shap_values.parquet"
    shap_df.to_csv(shap_csv_path, index=False)
    try:
        shap_df.to_parquet(shap_parquet_path, index=False)
    except ImportError:
        pass

    table_full.to_csv(OUTPUT_DIR / "table_4_5_machine_learning_performance_full_precision.csv", index=False)
    table_publication.to_csv(OUTPUT_DIR / "table_4_5_machine_learning_performance.csv", index=False)
    robustness_publication.to_csv(OUTPUT_DIR / "table_A5_machine_learning_robustness.csv", index=False)
    importance.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)

    latex_table(
        table_publication,
        "Machine Learning Performance Comparison",
        "tab:machine_learning_performance",
        OUTPUT_DIR / "table_4_5_machine_learning_performance.tex",
    )
    latex_table(
        robustness_publication,
        "Machine Learning Robustness Checks",
        "tab:machine_learning_robustness",
        OUTPUT_DIR / "table_A5_machine_learning_robustness.tex",
    )

    shutil.copy2(OUTPUT_DIR / "table_4_5_machine_learning_performance.csv", WRITING_MAIN_DIR / "table_4_5_machine_learning_performance.csv")
    shutil.copy2(OUTPUT_DIR / "table_4_5_machine_learning_performance.tex", WRITING_MAIN_DIR / "table_4_5_machine_learning_performance.tex")
    shutil.copy2(OUTPUT_DIR / "table_A5_machine_learning_robustness.csv", WRITING_APPENDIX_DIR / "table_A5_machine_learning_robustness.csv")
    shutil.copy2(OUTPUT_DIR / "table_A5_machine_learning_robustness.tex", WRITING_APPENDIX_DIR / "table_A5_machine_learning_robustness.tex")

    ols_metrics = pd.read_csv(OLS_METRICS_PATH).iloc[0].to_dict()
    sem_metrics = pd.read_csv(SEM_METRICS_PATH).iloc[0].to_dict()
    summary = {
        "sprint": "Sprint 11: Machine Learning Models",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(DATA_PATH.relative_to(ROOT)),
        "geometry_path": str(GEOMETRY_PATH.relative_to(ROOT)),
        "response": RESPONSE,
        "predictors": PREDICTORS,
        "sample_size": int(len(data)),
        "random_seed": RANDOM_SEED,
        "validation": {
            "spatial_cv_strategy": f"{N_SPATIAL_FOLDS}-fold KMeans spatial blocks using frozen geometry centroids",
            "n_spatial_folds": N_SPATIAL_FOLDS,
            "held_out_test_fold": TEST_FOLD,
            "fold_counts": folds.value_counts().sort_index().astype(int).to_dict(),
            "permutation_repeats": PERMUTATION_REPEATS,
        },
        "models": model_results,
        "benchmark_models": {
            "OLS": {
                "r_squared": float(ols_metrics["r_squared"]),
                "adjusted_r_squared": float(ols_metrics["adjusted_r_squared"]),
                "rmse": float(ols_metrics["rmse"]),
                "mae": float(ols_metrics["mae"]),
                "aic": float(ols_metrics["aic"]),
                "bic": float(ols_metrics["bic"]),
                "log_likelihood": float(ols_metrics["log_likelihood"]),
            },
            "SEM": {
                "pseudo_r_squared": float(sem_metrics["pseudo_r_squared"]),
                "rmse": float(sem_metrics["rmse"]),
                "mae": float(sem_metrics["mae"]),
                "aic": float(sem_metrics["aic"]),
                "bic": float(sem_metrics["bic"]),
                "log_likelihood": float(sem_metrics["log_likelihood"]),
            },
        },
        "best_model": best_model,
        "best_model_selection_rule": "Highest held-out test R2; spatial CV R2 reported as spatial generalisation check.",
        "top_five_importance": top_five,
        "shap_values_generated": True,
        "shap_values_model": shap_model,
        "shap_values_method": shap_method,
        "shap_values_rows": int(len(shap_df)),
        "shap_values_missing": int(shap_df.isna().sum().sum()),
        "qa": {
            **qa,
            "shap_values_no_missing": int(shap_df.isna().sum().sum()) == 0,
            "shap_values_parquet_exists": shap_parquet_path.exists(),
            "publication_main_csv_matches_output": (OUTPUT_DIR / "table_4_5_machine_learning_performance.csv").read_bytes()
            == (WRITING_MAIN_DIR / "table_4_5_machine_learning_performance.csv").read_bytes(),
            "publication_main_tex_matches_output": (OUTPUT_DIR / "table_4_5_machine_learning_performance.tex").read_bytes()
            == (WRITING_MAIN_DIR / "table_4_5_machine_learning_performance.tex").read_bytes(),
            "publication_appendix_csv_matches_output": (OUTPUT_DIR / "table_A5_machine_learning_robustness.csv").read_bytes()
            == (WRITING_APPENDIX_DIR / "table_A5_machine_learning_robustness.csv").read_bytes(),
            "publication_appendix_tex_matches_output": (OUTPUT_DIR / "table_A5_machine_learning_robustness.tex").read_bytes()
            == (WRITING_APPENDIX_DIR / "table_A5_machine_learning_robustness.tex").read_bytes(),
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "geopandas": gpd.__version__,
            "sklearn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "shap": shap.__version__ if shap is not None else "not importable; missing numba dependency",
        },
        "prohibited_models_run": False,
        "publication_figures_generated": False,
    }
    (OUTPUT_DIR / "ml_model_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    update_manifest()
    update_results_book(summary)

    print(json.dumps({"best_model": best_model, "summary": str(OUTPUT_DIR / "ml_model_summary.json")}, indent=2))


if __name__ == "__main__":
    main()
