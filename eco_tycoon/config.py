from __future__ import annotations
from dataclasses import dataclass
import os

FEATURES = [
    "ndvi",
    "forest_cover",
    "elevation",
    "slope",
    "rainfall",
    "temperature",
    "distance_road",
    "distance_water",
    "development_intensity",
    "climate_exposure",
    "species_vulnerability",
    "dependency_impact",
]

SCENARIOS = {
    "Business as Usual": {
        "ndvi_factor": 0.85,
        "forest_cover_factor": 0.80,
        "development_intensity_factor": 1.30,
        "climate_exposure_factor": 1.15,
    },
    "Mitigation": {
        "ndvi_factor": 0.95,
        "forest_cover_factor": 0.95,
        "development_intensity_factor": 1.05,
        "climate_exposure_factor": 1.05,
    },
    "Conservation": {
        "ndvi_factor": 1.05,
        "forest_cover_factor": 1.05,
        "development_intensity_factor": 0.80,
        "climate_exposure_factor": 0.90,
    },
}

@dataclass
class ModelConfig:
    demo_rows: int = int(os.getenv("ECO_DEMO_ROWS", "50000"))
    rf_estimators: int = int(os.getenv("ECO_RF_TREES", "600"))
    rf_depth: int = int(os.getenv("ECO_RF_DEPTH", "20"))
    xgb_estimators: int = int(os.getenv("ECO_XGB_TREES", "800"))
    xgb_depth: int = int(os.getenv("ECO_XGB_DEPTH", "8"))
    xgb_lr: float = float(os.getenv("ECO_XGB_LR", "0.05"))
    random_state: int = 42