-- Eco Tycoon DuckDB Analytical Schema
-- Run with: duckdb -c ".read sql/schema.sql"

-- ============================================
-- DIMENSION TABLES
-- ============================================

CREATE SEQUENCE IF NOT EXISTS dim_project_seq;
CREATE TABLE IF NOT EXISTS dim_project (
    project_id INTEGER PRIMARY KEY DEFAULT nextval('dim_project_seq'),
    project_name VARCHAR NOT NULL,
    region VARCHAR,
    country VARCHAR(3),
    geometry_wkt VARCHAR,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE SEQUENCE IF NOT EXISTS dim_species_seq;
CREATE TABLE IF NOT EXISTS dim_species (
    species_id INTEGER PRIMARY KEY DEFAULT nextval('dim_species_seq'),
    scientific_name VARCHAR NOT NULL,
    common_name VARCHAR,
    taxonomic_class VARCHAR,
    conservation_status VARCHAR(10), -- LC, NT, VU, EN, CR, DD
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE SEQUENCE IF NOT EXISTS dim_scenario_seq;
CREATE TABLE IF NOT EXISTS dim_scenario (
    scenario_id INTEGER PRIMARY KEY DEFAULT nextval('dim_scenario_seq'),
    scenario_name VARCHAR NOT NULL UNIQUE,
    ndvi_factor DOUBLE NOT NULL,
    forest_cover_factor DOUBLE NOT NULL,
    development_intensity_factor DOUBLE NOT NULL,
    climate_exposure_factor DOUBLE NOT NULL,
    description VARCHAR
);

-- Insert default scenarios
INSERT OR IGNORE INTO dim_scenario (scenario_name, ndvi_factor, forest_cover_factor, development_intensity_factor, climate_exposure_factor, description) VALUES
('Business as Usual', 0.85, 0.80, 1.30, 1.15, 'Continued development without additional mitigation'),
('Mitigation', 0.95, 0.95, 1.05, 1.05, 'Development with standard mitigation measures'),
('Conservation', 1.05, 1.05, 0.80, 0.90, 'Conservation-focused management');

-- ============================================
-- FACT TABLES
-- ============================================

CREATE SEQUENCE IF NOT EXISTS fact_environment_seq;
CREATE TABLE IF NOT EXISTS fact_environment (
    observation_id INTEGER PRIMARY KEY DEFAULT nextval('fact_environment_seq'),
    project_id INTEGER NOT NULL REFERENCES dim_project(project_id),
    timestamp TIMESTAMP NOT NULL,
    ndvi DOUBLE,
    forest_cover DOUBLE,
    elevation DOUBLE,
    slope DOUBLE,
    rainfall DOUBLE,
    temperature DOUBLE,
    distance_road DOUBLE,
    distance_water DOUBLE,
    development_intensity DOUBLE,
    climate_exposure DOUBLE,
    species_vulnerability DOUBLE,
    dependency_impact DOUBLE
);

CREATE INDEX IF NOT EXISTS idx_fact_environment_project ON fact_environment(project_id);
CREATE INDEX IF NOT EXISTS idx_fact_environment_timestamp ON fact_environment(timestamp);

CREATE SEQUENCE IF NOT EXISTS fact_occurrence_seq;
CREATE TABLE IF NOT EXISTS fact_occurrence (
    occurrence_id INTEGER PRIMARY KEY DEFAULT nextval('fact_occurrence_seq'),
    species_id INTEGER NOT NULL REFERENCES dim_species(species_id),
    project_id INTEGER NOT NULL REFERENCES dim_project(project_id),
    gbif_id VARCHAR,
    decimal_latitude DOUBLE NOT NULL,
    decimal_longitude DOUBLE NOT NULL,
    event_date DATE,
    basis_of_record VARCHAR,
    coordinate_uncertainty DOUBLE,
    cleaned BOOLEAN DEFAULT FALSE,
    spatially_thinned BOOLEAN DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_fact_occurrence_species ON fact_occurrence(species_id);
CREATE INDEX IF NOT EXISTS idx_fact_occurrence_project ON fact_occurrence(project_id);
CREATE INDEX IF NOT EXISTS idx_fact_occurrence_coords ON fact_occurrence(decimal_latitude, decimal_longitude);

CREATE SEQUENCE IF NOT EXISTS fact_prediction_seq;
CREATE TABLE IF NOT EXISTS fact_prediction (
    prediction_id INTEGER PRIMARY KEY DEFAULT nextval('fact_prediction_seq'),
    project_id INTEGER NOT NULL REFERENCES dim_project(project_id),
    scenario_id INTEGER NOT NULL REFERENCES dim_scenario(scenario_id),
    years_horizon INTEGER NOT NULL,
    baseline_habitat DOUBLE,
    future_habitat DOUBLE,
    habitat_loss_pct DOUBLE,
    fragmentation_index DOUBLE,
    risk_score DOUBLE,
    risk_category VARCHAR(20),
    model_version VARCHAR,
    predicted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_fact_prediction_project ON fact_prediction(project_id);
CREATE INDEX IF NOT EXISTS idx_fact_prediction_scenario ON fact_prediction(scenario_id);
CREATE INDEX IF NOT EXISTS idx_fact_prediction_horizon ON fact_prediction(years_horizon);

-- ============================================
-- ANALYTICAL VIEWS
-- ============================================

-- Project summary with latest environmental conditions
CREATE OR REPLACE VIEW v_project_summary AS
SELECT
    p.project_id,
    p.project_name,
    p.region,
    p.country,
    COUNT(DISTINCT e.observation_id) AS num_observations,
    MAX(e.timestamp) AS latest_observation,
    AVG(e.ndvi) AS avg_ndvi,
    AVG(e.forest_cover) AS avg_forest_cover,
    AVG(e.development_intensity) AS avg_development_intensity,
    COUNT(DISTINCT o.occurrence_id) AS num_occurrences,
    COUNT(DISTINCT o.species_id) AS num_species
FROM dim_project p
LEFT JOIN fact_environment e ON p.project_id = e.project_id
LEFT JOIN fact_occurrence o ON p.project_id = o.project_id
GROUP BY p.project_id, p.project_name, p.region, p.country;

-- Species occurrence summary per project
CREATE OR REPLACE VIEW v_species_project_summary AS
SELECT
    p.project_id,
    p.project_name,
    s.species_id,
    s.scientific_name,
    s.conservation_status,
    COUNT(o.occurrence_id) AS occurrence_count,
    MIN(o.event_date) AS first_record,
    MAX(o.event_date) AS last_record,
    AVG(o.coordinate_uncertainty) AS avg_uncertainty
FROM dim_project p
JOIN fact_occurrence o ON p.project_id = o.project_id
JOIN dim_species s ON o.species_id = s.species_id
WHERE o.cleaned = TRUE
GROUP BY p.project_id, p.project_name, s.species_id, s.scientific_name, s.conservation_status;

-- Scenario comparison per project
CREATE OR REPLACE VIEW v_scenario_comparison AS
SELECT
    pr.project_id,
    pr.project_name,
    sc.scenario_name,
    pr.years_horizon,
    pr.baseline_habitat,
    pr.future_habitat,
    pr.habitat_loss_pct,
    pr.fragmentation_index,
    pr.risk_score,
    pr.risk_category,
    pr.predicted_at
FROM fact_prediction pr
JOIN dim_project p ON pr.project_id = p.project_id
JOIN dim_scenario sc ON pr.scenario_id = sc.scenario_id
ORDER BY pr.project_id, sc.scenario_name, pr.years_horizon;

-- Model performance tracking
CREATE OR REPLACE VIEW v_model_performance AS
SELECT
    model_version,
    COUNT(*) AS num_predictions,
    AVG(risk_score) AS avg_risk_score,
    AVG(habitat_loss_pct) AS avg_habitat_loss,
    MIN(predicted_at) AS first_prediction,
    MAX(predicted_at) AS latest_prediction
FROM fact_prediction
WHERE model_version IS NOT NULL
GROUP BY model_version;

-- ============================================
-- UTILITY FUNCTIONS (DuckDB macros)
-- ============================================

-- Calculate habitat suitability from features (matches ML model logic)
CREATE OR REPLACE MACRO calculate_habitat_suitability(
    ndvi, forest_cover, development_intensity, distance_water,
    rainfall, climate_exposure, distance_road, slope
) AS (
    0.35 * ndvi
    + 0.25 * (forest_cover / 100)
    + 0.10 * (1 - development_intensity)
    + 0.08 * (1 - LEAST(distance_water / 20, 1))
    + 0.07 * (LEAST(rainfall / 4000, 1))
    + 0.05 * (1 - climate_exposure)
    + 0.05 * (1 - LEAST(distance_road / 50, 1))
    + 0.05 * (1 - LEAST(slope / 60, 1))
);

-- Calculate fragmentation index
CREATE OR REPLACE MACRO calculate_fragmentation_index(
    development_intensity, distance_road, forest_cover, slope
) AS (
    0.40 * development_intensity
    + 0.25 * (1 - LEAST(distance_road / 10, 1))
    + 0.20 * (1 - forest_cover / 100)
    + 0.15 * (slope / 60)
);

-- ============================================
-- EXAMPLE QUERIES
-- ============================================

-- Load Parquet files (example)
-- COPY dim_project FROM 'data/dim_project.parquet';
-- COPY fact_environment FROM 'data/fact_environment.parquet';
-- COPY fact_occurrence FROM 'data/fact_occurrence.parquet';
-- COPY fact_prediction FROM 'data/fact_prediction.parquet';

-- Export to Parquet (example)
-- COPY dim_project TO 'output/dim_project.parquet' (FORMAT PARQUET);
-- COPY fact_environment TO 'output/fact_environment.parquet' (FORMAT PARQUET);
-- COPY fact_occurrence TO 'output/fact_occurrence.parquet' (FORMAT PARQUET);
-- COPY fact_prediction TO 'output/fact_prediction.parquet' (FORMAT PARQUET);

-- Query: Get all predictions for a project with scenario names
-- SELECT * FROM v_scenario_comparison WHERE project_id = 1;

-- Query: Species richness per project
-- SELECT project_name, COUNT(DISTINCT species_id) AS species_richness
-- FROM v_species_project_summary
-- GROUP BY project_name;

-- Query: Average habitat loss by scenario
-- SELECT scenario_name, AVG(habitat_loss_pct) AS avg_habitat_loss
-- FROM v_scenario_comparison
-- GROUP BY scenario_name
-- ORDER BY avg_habitat_loss DESC;

-- ============================================
-- CHUNKED CSV LOADING (for large files)
-- ============================================

-- Example: Load large CSV in chunks
-- CREATE TABLE temp_env AS SELECT * FROM fact_environment LIMIT 0;
--
-- -- In Python:
-- import duckdb
-- con = duckdb.connect('ecotycoon.duckdb')
-- chunk_size = 100000
-- for chunk in pd.read_csv('large_file.csv', chunksize=chunk_size):
--     con.register('chunk_df', chunk)
--     con.execute("INSERT INTO fact_environment SELECT * FROM chunk_df")