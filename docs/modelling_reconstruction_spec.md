# Modelling Reconstruction Specification

This specification documents validated OLS, SAR, and SEM reproducibility implementations corresponding to the final dissertation specification. They were reconstructed from frozen specifications and aggregate result files and are not claimed to be verified original historical scripts. Settings are marked as CONFIRMED, INFERRED, or UNKNOWN.

## Core Specification

| Setting | Value | Status | Evidence |
| --- | --- | --- | --- |
| Observation unit | Greater London 2021 LSOA | CONFIRMED | `model_specification.json` |
| Sample size | 4,994 complete cases | CONFIRMED | `model_specification.json`, OLS/spatial summaries |
| Unique key | `LSOA21CD` | CONFIRMED | `model_specification.json` |
| Response | `log1p_employees_count` | CONFIRMED | `model_specification.json` |
| Response transformation | `log1p(employees_count)` | CONFIRMED | `model_specification.json` |
| Intercept | Included | CONFIRMED | Formula metadata and coefficient tables |
| Missing-value policy | Complete-case numeric modelling rows | CONFIRMED | `complete_case_n = 4994`; zero missing predictor values |
| Predictor standardisation before modelling | None | CONFIRMED | `standardisation_plan.before_modelling = false` |
| VIF/condition diagnostics | z-standardised predictors, no intercept | CONFIRMED | `predictor_diagnostics_summary.json` |
| Robust OLS inference | HC3 | CONFIRMED | `ols_regression_summary.json`, robust coefficient table |
| Queen weights | `Queen.from_dataframe(..., ids=LSOA21CD, use_index=False); transform = "R"` | CONFIRMED | `model_specification.json`, spatial summary |
| Moran permutations | 999 | CONFIRMED | `ols_diagnostic_summary.json` |
| Moran random seed | 20260722 | CONFIRMED | `ols_diagnostic_summary.json` |
| SAR/SEM estimation | Direct concentrated maximum likelihood with sparse LU log-determinants | CONFIRMED | `spatial_model_selection_summary.json` |
| SAR/SEM bounds | `[-0.95, 0.95]` | CONFIRMED | `spatial_model_selection_summary.json` |
| Spatial standard errors | Reconstructed for SAR/SEM coefficients and spatial parameters | CONFIRMED | `sar_coefficients.csv`, `sem_coefficients.csv`, Stage 4B.1 validation |

## Predictor Sets

Baseline accessibility-controls OLS predictors:

- `ptal_ai_mean`
- `population_density_km2`
- `amenity_poi_density_km2`
- `retail_poi_density_km2`

Enhanced OLS, SAR, and SEM predictors:

- `mobility_inflow_total`
- `flow_entropy`
- `weekday_weekend_ratio`
- `ptal_ai_mean`
- `population_density_km2`
- `amenity_poi_density_km2`
- `retail_poi_density_km2`

## Diagnostics

OLS diagnostics include HC3 robust inference, Pearson correlation matrix, VIF, z-standardised condition number, Jarque-Bera, Breusch-Pagan, RESET, residual standard error, maximum absolute standardised residual, and residual Moran's I.

Spatial validation includes SAR rho, SEM lambda, log-likelihood, AIC, RMSE, model comparison, and coefficient direction checks. SAR RMSE is calculated from structural SAR residuals, matching the frozen dissertation output convention.

## Reconstruction Boundary

The repository code is a validated final reconstruction from frozen specifications and aggregate outputs. It does not claim recovery of the original dissertation source code and does not bundle data, geometry, row-level predictions, residuals, SHAP arrays, or model binaries.
