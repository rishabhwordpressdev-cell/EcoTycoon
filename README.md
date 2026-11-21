# Eco Tycoon

**AI-Based Predictive Biodiversity Impact Assessment Platform**

Eco Tycoon is a modular Streamlit application for biodiversity impact assessment. It integrates environmental observations, species occurrences, development scenarios, and machine learning models to predict habitat suitability, quantify habitat loss and fragmentation, estimate biodiversity risk, and produce actionable recommendations.

## Features

- **Random Forest** model for habitat suitability prediction
- **XGBoost** model for biodiversity risk prediction
- **Scenario comparison** for Business as Usual, Mitigation, and Conservation at 0, 5, 10, and 20 years
- **GBIF occurrence retrieval** and cleaning tools
- **Species dependency network** analysis
- **Raster fragmentation/connectivity** metrics
- **MaxEnt** species distribution modeling (optional, requires `requirements-advanced.txt`)
- **LSTM** temporal forecasting (optional, requires `requirements-advanced.txt`)
- **Large-data support** using DuckDB, Parquet/Arrow, and chunked CSV ingestion

## Quick Start

### Windows (Easy Method)

1. Extract the ZIP file to a folder (e.g., `C:\EcoTycoon`)
2. Double-click `install_windows.bat` to create virtual environment and install dependencies
3. Double-click `run_app.bat` to start the application
4. Open http://localhost:8501 in your browser

### Windows (Manual Method)

```cmd
# Create virtual environment
python -m venv .venv

# Activate it
.venv\Scripts\activate

# Install core dependencies
pip install -r requirements.txt

# Start the application
python -m streamlit run app.py
```

### Linux/macOS

```bash
# Create virtual environment
python3 -m venv .venv

# Activate it
source .venv/bin/activate

# Install core dependencies
pip install -r requirements.txt

# Start the application
python -m streamlit run app.py
```

## Advanced Features

For MaxEnt, LSTM, Earth Engine, and optional GIS features:

```bash
pip install -r requirements-advanced.txt
```

## Command-Line Demo

Run a quick demo without the Streamlit UI:

```bash
python run_demo.py

# With custom configuration
set ECO_DEMO_ROWS=10000
set ECO_RF_TREES=200
set ECO_XGB_TREES=300
python run_demo.py
```

## Data Files

The `data/` directory includes sample datasets:

| File | Description | Rows |
|------|-------------|------|
| `sample_environment.csv` | Sample environmental features for manual entry | 10 |
| `environmental_monitoring_1200.csv` | Synthetic environmental monitoring data | 1,200 |
| `species_occurrences_1500.csv` | Synthetic GBIF-style species occurrences | 1,515 |
| `species_dependencies_1000.csv` | Synthetic species interaction network | 1,050 |
| `habitat_suitability_grid.csv` | 50x50 habitat suitability grid for spatial metrics | 2,500 cells |
| `maxent_training_data.csv` | Presence/background data for MaxEnt training | 1,200 |

## Project Structure

```
Eco_Tycoon/
├── app.py                      # Main Streamlit application
├── run_app.bat                 # Windows launcher
├── install_windows.bat         # Windows installation helper
├── run_demo.py                 # Command-line demo
├── requirements.txt            # Core dependencies
├── requirements-advanced.txt   # Advanced dependencies (MaxEnt, LSTM, GEE, GIS)
├── README.md
├── DATA_MODEL.md
├── DATA_SOURCES.md
├── .env.example
├── data/
│   ├── sample_environment.csv
│   ├── environmental_monitoring_1200.csv
│   ├── species_occurrences_1500.csv
│   ├── species_dependencies_1000.csv
│   ├── habitat_suitability_grid.csv
│   └── maxent_training_data.csv
├── sql/
│   └── schema.sql
└── eco_tycoon/
    ├── __init__.py
    ├── config.py               # Features, scenarios, model configuration
    ├── models.py               # RF/XGBoost training and synthetic data
    ├── analytics.py            # Site analysis and scenario comparison
    ├── data_sources.py         # GBIF client and file loading
    ├── preprocess.py           # Occurrence cleaning and spatial thinning
    ├── network.py              # Dependency graph and centrality
    ├── explainability.py       # Feature importance
    ├── advanced_models.py      # LSTM and MaxEnt
    ├── spatial.py              # Fragmentation metrics
    └── storage.py              # DuckDB schema (if implemented)
```

## Dashboard Tabs

1. **Project Analysis** - Manual feature entry, run biodiversity analysis
2. **Scenario Comparison** - Compare BAU, Mitigation, Conservation across time horizons
3. **Data & GBIF** - Upload files, fetch and clean GBIF occurrences
4. **Dependencies** - Upload species interaction CSV, analyze network centrality
5. **Spatial Metrics** - Upload habitat grid, calculate fragmentation/connectivity
6. **MaxEnt** - Train presence-only species distribution models
7. **LSTM Forecast** - Train temporal forecasting models
8. **About** - Methods documentation

## Model Configuration

Override default settings via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `ECO_DEMO_ROWS` | 50000 | Training dataset size |
| `ECO_RF_TREES` | 600 | Random Forest estimators |
| `ECO_RF_DEPTH` | 20 | Random Forest max depth |
| `ECO_XGB_TREES` | 800 | XGBoost estimators |
| `ECO_XGB_DEPTH` | 8 | XGBoost max depth |
| `ECO_XGB_LR` | 0.05 | XGBoost learning rate |

## Important Limitations

> **Eco Tycoon is intended as an AI-assisted predictive layer for early-stage environmental screening and planning. It is not a replacement for statutory Environmental Impact Assessment, field surveys, expert ecological judgement, legal review, or stakeholder consultation.**

The default large-model training set is **synthetic** and only for software validation. Real research runs should use validated data from:
- **GBIF**: Species occurrence observations
- **Google Dynamic World**: 10m land-cover information
- **Global Forest Watch**: Tree-cover and forest-change indicators
- **Google Earth Engine**: Scalable satellite/environmental feature extraction

## License

This project is provided for research and educational purposes.

## Citation

Rishabh Patil (2025). Eco Tycoon: AI-Based Predictive Biodiversity Impact Assessment Platform. Research Methodology and Academic Writing Lab.