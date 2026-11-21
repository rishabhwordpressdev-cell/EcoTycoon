# Eco Tycoon Data Model

## Overview

The Eco Tycoon data model follows a dimensional modeling approach with dimension tables and fact tables, designed for analytical queries using DuckDB.

## Entity-Relationship Diagram

```
dim_project
    |
    +---- fact_environment ----> ML features
    |
    +---- fact_prediction <---- scenarios

dim_species ---- fact_occurrence
```

## Dimension Tables

### dim_project
| Column | Type | Description |
|--------|------|-------------|
| project_id | INTEGER | Primary key |
| project_name | VARCHAR | Project identifier |
| region | VARCHAR | Geographic region |
| country | VARCHAR | ISO country code |
| geometry_wkt | VARCHAR | Project boundary as WKT |
| created_at | TIMESTAMP | Record creation time |
| updated_at | TIMESTAMP | Last update time |

### dim_species
| Column | Type | Description |
|--------|------|-------------|
| species_id | INTEGER | Primary key |
| scientific_name | VARCHAR | Binomial nomenclature |
| common_name | VARCHAR | Common name |
| taxonomic_class | VARCHAR | Taxonomic class |
| conservation_status | VARCHAR | IUCN status (LC, NT, VU, EN, CR) |
| created_at | TIMESTAMP | Record creation time |

### dim_scenario
| Column | Type | Description |
|--------|------|-------------|
| scenario_id | INTEGER | Primary key |
| scenario_name | VARCHAR | Scenario name (BAU, Mitigation, Conservation) |
| ndvi_factor | DOUBLE | NDVI modifier |
| forest_cover_factor | DOUBLE | Forest cover modifier |
| development_intensity_factor | DOUBLE | Development intensity modifier |
| climate_exposure_factor | DOUBLE | Climate exposure modifier |
| description | VARCHAR | Scenario description |

## Fact Tables

### fact_environment
Environmental observations and ML features for project locations.

| Column | Type | Description |
|--------|------|-------------|
| observation_id | INTEGER | Primary key |
| project_id | INTEGER | FK → dim_project |
| timestamp | TIMESTAMP | Observation date |
| ndvi | DOUBLE | Normalized Difference Vegetation Index (0-1) |
| forest_cover | DOUBLE | Forest cover percentage (0-100) |
| elevation | DOUBLE | Elevation in meters |
| slope | DOUBLE | Slope in degrees |
| rainfall | DOUBLE | Annual rainfall in mm |
| temperature | DOUBLE | Mean annual temperature in °C |
| distance_road | DOUBLE | Distance to nearest road in km |
| distance_water | DOUBLE | Distance to nearest water body in km |
| development_intensity | DOUBLE | Development intensity index (0-1) |
| climate_exposure | DOUBLE | Climate exposure index (0-1) |
| species_vulnerability | DOUBLE | Species vulnerability index (0-1) |
| dependency_impact | DOUBLE | Dependency impact index (0-1) |

### fact_occurrence
Species occurrence records (from GBIF or field surveys).

| Column | Type | Description |
|--------|------|-------------|
| occurrence_id | INTEGER | Primary key |
| species_id | INTEGER | FK → dim_species |
| project_id | INTEGER | FK → dim_project |
| gbif_id | VARCHAR | Original GBIF record ID |
| decimal_latitude | DOUBLE | Latitude (WGS84) |
| decimal_longitude | DOUBLE | Longitude (WGS84) |
| event_date | DATE | Observation date |
| basis_of_record | VARCHAR | HUMAN_OBSERVATION, PRESERVED_SPECIMEN, etc. |
| coordinate_uncertainty | DOUBLE | Coordinate uncertainty in meters |
| cleaned | BOOLEAN | Whether record passed cleaning |
| spatially_thinned | BOOLEAN | Whether record passed spatial thinning |

### fact_prediction
Model predictions for scenarios.

| Column | Type | Description |
|--------|------|-------------|
| prediction_id | INTEGER | Primary key |
| project_id | INTEGER | FK → dim_project |
| scenario_id | INTEGER | FK → dim_scenario |
| years_horizon | INTEGER | Time horizon (0, 5, 10, 20) |
| baseline_habitat | DOUBLE | Baseline habitat suitability |
| future_habitat | DOUBLE | Future habitat suitability |
| habitat_loss_pct | DOUBLE | Habitat loss percentage |
| fragmentation_index | DOUBLE | Fragmentation index (0-1) |
| risk_score | DOUBLE | Biodiversity risk score (0-1) |
| risk_category | VARCHAR | Low, Moderate, High, Critical |
| model_version | VARCHAR | Model version identifier |
| predicted_at | TIMESTAMP | Prediction timestamp |

## Feature List (ML Features)

The 12 primary features used by Random Forest and XGBoost models:

1. `ndvi` - Normalized Difference Vegetation Index (0-1)
2. `forest_cover` - Forest cover percentage (0-100)
3. `elevation` - Elevation in meters
4. `slope` - Slope in degrees
5. `rainfall` - Annual rainfall in mm
6. `temperature` - Mean annual temperature in °C
7. `distance_road` - Distance to nearest road in km
8. `distance_water` - Distance to nearest water body in km
9. `development_intensity` - Development intensity index (0-1)
10. `climate_exposure` - Climate exposure index (0-1)
11. `species_vulnerability` - Species vulnerability index (0-1)
12. `dependency_impact` - Dependency impact index (0-1)

## Targets

- `habitat_target` - Habitat suitability (0-1) for Random Forest
- `risk_target` - Biodiversity risk (0-1) for XGBoost

## Storage Formats

- **Parquet/Arrow**: Columnar format for efficient analytical queries
- **CSV (chunked)**: Bounded-memory ingestion for large files
- **DuckDB**: Embedded analytical database with SQL interface

## Data Flow

```
Raw Data (GBIF, Dynamic World, GFW, Earth Engine)
    ↓
Preprocessing (cleaning, spatial thinning, alignment)
    ↓
Feature Engineering (12 ML features)
    ↓
Dimensional Model (dim/fact tables)
    ↓
Machine Learning (RF habitat, XGBoost risk)
    ↓
Scenario Engine (BAU/Mitigation/Conservation)
    ↓
Predictions (fact_prediction)
    ↓
Dashboard / Reports
```