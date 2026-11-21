# Eco Tycoon Data Sources

## Primary Data Sources

### 1. GBIF (Global Biodiversity Information Facility)
- **URL**: https://www.gbif.org/
- **API**: https://api.gbif.org/v1/
- **Data**: Species occurrence records with coordinates, dates, and metadata
- **Access**: Free, open access
- **Usage in Eco Tycoon**: 
  - `eco_tycoon.data_sources.GBIFClient` for occurrence search
  - `eco_tycoon.preprocess.clean_occurrences` for coordinate validation
  - `eco_tycoon.preprocess.spatial_thin_occurrences` for sampling bias reduction

**Key API Endpoints**:
- Occurrence search: `GET /occurrence/search`
- Species matching: `GET /species/match`
- Asynchronous download: `POST /occurrence/download/request`

### 2. Google Dynamic World V1
- **URL**: https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_DYNAMICWORLD_V1
- **Resolution**: 10 meters
- **Temporal**: Near real-time (2-5 day latency)
- **Classes**: Water, Trees, Grass, Flooded Vegetation, Crops, Shrub/Scrub, Built Area, Bare Ground, Snow/Ice
- **Usage in Eco Tycoon**: Land cover classification, forest cover estimation, NDVI proxy

### 3. Global Forest Watch (GFW)
- **URL**: https://data.globalforestwatch.org/
- **Data**: Tree cover extent, tree cover loss, tree cover gain, protected areas, biodiversity indicators
- **Formats**: GeoTIFF, Vector tiles, API
- **Usage in Eco Tycoon**: Forest change detection, habitat loss quantification, protected area overlap

### 4. Google Earth Engine (GEE)
- **URL**: https://earthengine.google.com/
- **Data Catalog**: https://developers.google.com/earth-engine/datasets/catalog
- **Key Datasets**:
  - MODIS: NDVI, EVI, LST, Land Cover
  - Landsat/Sentinel: Surface reflectance, indices
  - SRTM/ASTER: Elevation, slope
  - CHIRPS/ERA5: Precipitation, temperature
  - WorldPop: Human population density
- **Usage in Eco Tycoon**: Scalable feature extraction for large regions, time series analysis

## Derived Environmental Variables

| Variable | Source | Processing |
|----------|--------|------------|
| NDVI | MODIS/Sentinel-2 | Median composite, cloud masking |
| Forest Cover | Dynamic World / GFW | Tree cover > 30% threshold |
| Elevation | SRTM 30m / ASTER GDEM | Direct extraction |
| Slope | SRTM | Terrain analysis (degrees) |
| Rainfall | CHIRPS / ERA5 | Annual sum, mm |
| Temperature | ERA5 / MODIS LST | Annual mean, °C |
| Distance to Road | OpenStreetMap / GRIP | Euclidean distance, km |
| Distance to Water | HydroSHEDS / Dynamic World | Euclidean distance, km |
| Development Intensity | WorldPop / VIIRS Nightlights / Built Area | Composite index 0-1 |
| Climate Exposure | CMIP6 projections / ERA5 | Anomaly-based index 0-1 |
| Species Vulnerability | IUCN Red List / Literature | Expert-derived index 0-1 |
| Dependency Impact | Literature / Ecological networks | Network centrality-based 0-1 |

## Data Acquisition Workflow

### For Research Deployment

1. **Define Study Region**
   - Project boundary (polygon WKT/GeoJSON)
   - Buffer zone (e.g., 10-50 km)

2. **Acquire Occurrence Data**
   ```python
   from eco_tycoon.data_sources import GBIFClient, GBIFQuery
   
   client = GBIFClient()
   query = GBIFQuery(scientific_name="Panthera tigris", country="IN", limit=20000)
   occurrences = client.search(query)
   ```

3. **Clean Occurrences**
   ```python
   from eco_tycoon.preprocess import clean_occurrences, spatial_thin_occurrences
   
   cleaned = clean_occurrences(occurrences)
   thinned = spatial_thin_occurrences(cleaned, min_distance_km=1.0)
   ```

4. **Extract Environmental Features (via GEE)**
   - Upload project geometry to GEE
   - Sample environmental layers at occurrence locations
   - Generate background/pseudo-absence points
   - Compute 12 ML features for each point

5. **Build Training Dataset**
   - Combine presence and background points
   - Calculate habitat suitability targets
   - Calculate biodiversity risk targets
   - Split spatially (not randomly) for validation

6. **Train Models**
   - Random Forest for habitat suitability
   - XGBoost for biodiversity risk
   - Optional: MaxEnt for presence-only modeling
   - Optional: LSTM for temporal forecasting

## Sample Data Files

The `data/` directory contains synthetic sample data for demonstration:

| File | Source | Purpose |
|------|--------|---------|
| `sample_environment.csv` | Manual entry template | Project Analysis tab |
| `environmental_monitoring_1200.csv` | Synthetic | Model training demo |
| `species_occurrences_1500.csv` | Synthetic (GBIF-style) | Data & GBIF tab demo |
| `species_dependencies_1000.csv` | Synthetic (ecological) | Dependencies tab demo |
| `habitat_suitability_grid.csv` | Synthetic (50x50 grid) | Spatial Metrics tab demo |
| `maxent_training_data.csv` | Synthetic (presence/background) | MaxEnt tab demo |

## Data Quality Considerations

### Occurrence Data
- **Coordinate uncertainty**: Filter records > 10km uncertainty
- **Spatial bias**: Apply spatial thinning (1km grid recommended)
- **Temporal relevance**: Use recent records (last 20-30 years)
- **Taxonomic accuracy**: Verify species names via GBIF backbone

### Environmental Layers
- **Spatial alignment**: All layers must share CRS and resolution
- **Temporal consistency**: Match observation period with environmental data
- **Scale appropriateness**: 10-30m for site-level, 1km for regional

### Model Training
- **Spatial validation**: Use spatial block cross-validation, not random
- **Class imbalance**: Address presence/background ratio for MaxEnt
- **Feature correlation**: Check VIF, consider dimensionality reduction
- **Uncertainty quantification**: Ensemble spread, prediction intervals

## Licensing and Attribution

| Source | License | Attribution Required |
|--------|---------|---------------------|
| GBIF | CC0 / CC-BY 4.0 | Yes (varies by dataset) |
| Dynamic World | CC-BY 4.0 | Yes |
| Global Forest Watch | CC-BY 4.0 | Yes |
| Google Earth Engine | Various | Check each dataset |
| OpenStreetMap | ODbL | Yes |
| SRTM | Public Domain | No |
| CHIRPS | CC-BY 4.0 | Yes |
| ERA5 | Copernicus | Yes |

## Sensitive Species Data

**Important**: Some species occurrence data may be sensitive (e.g., endangered species locations). Before publishing or sharing:
- Apply coordinate generalization (e.g., 10km grid)
- Remove exact coordinates for CR/EN species
- Follow GBIF sensitive data guidelines
- Consult national wildlife authorities

## Recommended Citation

When using Eco Tycoon with real data, cite:
- Original data sources (GBIF, Dynamic World, GFW, GEE datasets)
- Eco Tycoon platform: Patil (2025). Eco Tycoon: AI-Based Predictive Biodiversity Impact Assessment Platform.
- Machine learning methods: Breiman (2001) Random Forests; Chen & Guestrin (2016) XGBoost; Phillips et al. (2006) MaxEnt