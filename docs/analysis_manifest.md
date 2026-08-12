# Analysis manifest

This document records the minimum analytical sequence corresponding to the final dissertation specification. All datasets are external and excluded.

## Final specification

- Unit: Greater London 2021 LSOA.
- Key: `LSOA21CD`.
- Complete-case sample: 4,994.
- Response: `log1p_employees_count = log1p(employees_count)`.
- Predictors: `mobility_inflow_total`, `flow_entropy`, `weekday_weekend_ratio`, `ptal_ai_mean`, `population_density_km2`, `amenity_poi_density_km2`, and `retail_poi_density_km2`.
- Predictor standardisation before modelling: none.
- Spatial weights: first-order Queen contiguity, stable LSOA identifier order, row-standardised.
- Random seed where applicable: `20260722`.

## Execution order

| Stage | Module | Principal outputs | Provenance |
| --- | --- | --- | --- |
| 1. Census | `src/preprocessing/01_build_census_population.py` | Population/density variables and QA summaries | Final dissertation code |
| 2. Mobility | `src/preprocessing/02_build_mobility_indicators.py` | Final 31-day LSOA mobility indicators and QA | Final dissertation code |
| 3. Master table | `src/preprocessing/03_build_master_dataset.py` | Model-ready table and geometry | Final dissertation code |
| 4. Descriptive ESDA | `src/analysis/04_descriptive_esda.py` | Descriptives, distributions, correlations, VIF | Final dissertation code |
| 5. Global spatial ESDA | `src/spatial/05_global_spatial_esda.py` | Queen diagnostics, Global Moran results, maps | Final dissertation code |
| 6. Local spatial ESDA | `src/spatial/06_local_spatial_esda.py` | Local Moran/LISA aggregates, sensitivity, overlap, maps | Final dissertation code |
| 7. OLS | `src/analysis/run_ols.py`, `ols.py`, `ols_diagnostics.py` | Aggregate metrics, HC3 coefficients, diagnostics | Validated final reproducibility implementation |
| 8. SAR and SEM | `src/spatial/run_spatial_models.py`, `spatial_models.py`, `weights.py` | Aggregate SAR/SEM metrics and inference | Validated final reproducibility implementation |
| 9. ML and SHAP | `src/ml/07_run_machine_learning.py` | Spatial-CV metrics, importance, SHAP outputs | Final dissertation code |
| 10. SHAP interpretation | `src/ml/08_analyse_shap.py` | Aggregate shape summaries and figures | Final dissertation code |
| 11. Robustness | `src/robustness/01-06_*.py` | Mobility-only, LM, predictive, log-X, weekend, and entropy checks | Final dissertation remediation code |
| 12. Reporting | `src/reporting/*.py` and stage table writers | Selected final figures and aggregate tables | Final dissertation code |

## OLS and spatial provenance

The OLS, SAR, and SEM modules are validated reproducibility implementations corresponding to the final dissertation specification. Their response, predictors, sample, weights, transformations, likelihood conventions, and aggregate outputs were validated against frozen dissertation records. They must not be represented as verified original historical scripts.

See `docs/modelling_reconstruction_spec.md` for confirmed settings.

## Committed outputs

The repository includes five representative figures and five aggregate tables only. They are reference outputs, not a complete archive. All row-level predictions, residuals, partitions, SHAP arrays, model binaries, caches, and intermediate outputs are excluded.

## Reproduction checks

Before comparing results, verify final columns, sample size, key uniqueness, CRS, table/geometry order, Queen-weight diagnostics, complete-case policy, no pre-model standardisation, and random seeds. Differences in source versions or authorised data deliveries must be documented rather than silently reconciled.
