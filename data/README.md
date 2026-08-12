# Data availability and input contracts

No data are distributed with this repository. All source, intermediate, processed, geospatial, model-ready, and row-level analytical data must remain outside version control.

Users are responsible for obtaining lawful access, complying with provider terms, documenting the versions they use, and configuring local paths. The placeholders in `config/config.example.yaml` are intentionally non-functional.

## Study frame

- Spatial unit: Greater London 2021 Lower Layer Super Output Area (LSOA).
- Stable key: `LSOA21CD`.
- Coordinate reference system: British National Grid, `EPSG:27700`.
- Final complete-case model size: 4,994 LSOAs.
- Employment period: BRES 2024 workplace employment.
- Mobility period: March 2026, covering 31 days in the final workflow.

## Required sources

| Source | Access and restrictions | Minimum expected content | Consuming stage |
| --- | --- | --- | --- |
| Locomizer aggregated origin-destination mobility | Restricted/provider-authorised access only; never redistribute records or deliveries | Destination/origin spatial identifiers, date/time coverage, and measures needed for inflow, entropy, and weekday-weekend indicators | `src/preprocessing/02_build_mobility_indicators.py` |
| BRES workplace employment | Obtain through an authorised official release/access route | LSOA-compatible employee counts used to create the response | Master construction and models |
| PTAL | Obtain from the relevant Transport for London release | Accessibility values sufficient to derive `ptal_ai_mean` | Master construction |
| Census 2021 population | Obtain from the ONS/Nomis release | `LSOA21CD` and usual-resident population | Census preprocessing |
| OpenStreetMap POIs | Obtain under applicable attribution/licence terms | Amenity and retail POIs sufficient to derive per-km2 densities | Master construction |
| London LSOA 2021 boundaries | Obtain from the official boundary release | Unique `LSOA21CD`, valid polygon geometry, `EPSG:27700` | Preprocessing and spatial stages |

## Final model-ready schema

| Variable | Meaning |
| --- | --- |
| `LSOA21CD` | Unique 2021 LSOA identifier |
| `employees_count` | BRES workplace employee count |
| `log1p_employees_count` | `log1p(employees_count)` final response |
| `mobility_inflow_total` | Final 31-day mobility inflow indicator |
| `flow_entropy` | Shannon entropy of valid mobility origins |
| `weekday_weekend_ratio` | Weekday mean daily inflow divided by weekend mean daily inflow |
| `ptal_ai_mean` | Mean PTAL accessibility index |
| `population_density_km2` | Census usual-resident population per square kilometre |
| `amenity_poi_density_km2` | Amenity POIs per square kilometre |
| `retail_poi_density_km2` | Retail POIs per square kilometre |
| `geometry` | LSOA polygon geometry for spatial stages |

Rows must be unique by `LSOA21CD`; tabular and geometry order must be reconciled before constructing spatial weights. Final models use complete numeric cases without pre-model predictor standardisation.

The dissertation-derived scripts retain a conventional local layout such as `data/raw/`, `data/interim/`, and `data/processed/`. These paths are ignored in full and are examples only. Users may instead provide external paths through local configuration or adjust module-level constants without committing those changes.

Never commit provider filenames, source hashes, row-level extracts, reconstructed observations, predictions, residuals, spatial folds, SHAP arrays, model binaries, or local absolute paths.
