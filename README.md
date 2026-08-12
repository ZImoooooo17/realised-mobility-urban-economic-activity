# Realised Mobility and Urban Economic Activity

Academic reproducibility repository for the MSc Urban Spatial Science dissertation at UCL CASA.

## Research context

Urban economic activity is unevenly distributed across cities and is shaped by accessibility, land use, transport infrastructure, and realised patterns of movement. This repository supports an analysis of whether mobility-derived indicators help explain workplace employment across Greater London, alongside conventional accessibility and contextual measures.

The study uses 2021 Lower Layer Super Output Areas (LSOAs) as its spatial unit. The final response is `log1p_employees_count`. The final explanatory specification combines mobility inflow, flow entropy, weekday-weekend ratio, PTAL, population density, amenity POI density, and retail POI density.

## Research questions

1. To what extent do mobility-derived indicators explain the spatial distribution of urban economic activity across Greater London?
2. Do mobility-derived indicators improve explanatory performance beyond traditional accessibility measures?
3. Do machine learning models outperform spatial statistical models in modelling mobility-economy relationships?

## Analytical workflow

The repository covers census and mobility preprocessing, master-table construction, descriptive ESDA, Global Moran analysis, Local Moran/LISA, OLS, SAR, SEM, spatially structured machine learning, SHAP interpretation, robustness checks, and selected figure/table generation.

OLS, SAR, and SEM are implemented as **validated reproducibility implementations corresponding to the final dissertation specification**. They reproduce the frozen final specification and aggregate outputs within documented tolerances, but they are not claimed to be the verified historical scripts originally run during the dissertation.

See `docs/analysis_manifest.md` for execution order and `docs/modelling_reconstruction_spec.md` for the confirmed model specification.

## Repository structure

```text
config/              Example paths and final analytical settings
data/                Data availability and input-schema documentation only
docs/                Analysis, provenance, and reproducibility documentation
outputs/figures/     Five representative final dissertation figures
outputs/tables/      Five aggregate, machine-readable final tables
src/preprocessing/   Census, mobility, and master-table construction
src/analysis/        Descriptive ESDA and validated OLS implementation
src/spatial/         Global/local ESDA and validated SAR/SEM implementations
src/ml/              Final machine-learning and SHAP workflows
src/robustness/      Final dissertation robustness analyses
src/reporting/       Selected final figure-generation scripts
tests/               Targeted imports, hygiene, OLS, and spatial tests
```

## Data availability

No research data are distributed in this repository. This includes raw, interim, processed, geospatial, mobility, employment, accessibility, POI, census, and master analytical datasets. Authorised users must obtain each source independently and configure local paths according to `data/README.md` and `config/config.example.yaml`.

Generated row-level predictions, residuals, spatial partitions, SHAP arrays, model binaries, and caches must also remain outside version control.

## Software environment

The Python dependency baseline is in `requirements.txt`. Geospatial packages may require platform-specific GDAL, GEOS, and PROJ libraries.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The final dissertation environment used Python 3.13 with recent NumPy, pandas, GeoPandas, statsmodels, PySAL, scikit-learn, XGBoost, and SHAP releases. Exact portability should be checked on the intended operating system before empirical reproduction.

## Reproduction

1. Obtain lawful access to all required source data.
2. Copy `config/config.example.yaml` to a local ignored configuration file and replace placeholders with local paths.
3. Run preprocessing stages in the order documented in `docs/analysis_manifest.md`.
4. Run ESDA, OLS, SAR/SEM, ML/SHAP, and robustness stages using the same final specification.
5. Compare aggregate outputs with the selected tables and figures under `outputs/`.

The validated OLS and spatial CLIs accept explicit input and output paths:

```bash
python -m src.analysis.run_ols --help
python -m src.spatial.run_spatial_models --help
```

The copied dissertation scripts retain repository-relative defaults for clarity. Review the example configuration and module-level paths before running them in a local authorised environment. Do not commit configured paths or generated analytical records.

## Tests

The approved tests use in-memory toy inputs and repository text only; they do not require dissertation data:

```bash
python -m pytest -q
```

## Reproducibility boundary

Public tests and reusable analytical methods are supported. Reproducing the empirical results requires lawful access to the excluded model-ready data and geometry. See `docs/reproducibility.md` for the precise boundary.

## Citation and licence

Citation metadata are provided in `CITATION.cff`. Original repository code and documentation are licensed under the MIT License. The licence does not grant rights to third-party datasets, provider records, dissertation source materials, or external works.
