from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

from .config import FEATURES, ModelConfig


@dataclass
class ModelBundle:
    habitat_model: RandomForestRegressor
    risk_model: XGBRegressor
    metrics: dict
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_habitat_train: pd.Series
    y_habitat_test: pd.Series
    y_risk_train: pd.Series
    y_risk_test: pd.Series


def build_large_demo_dataset(rows: int, seed: int = 42) -> pd.DataFrame:
    """Build a large synthetic dataset for demo/validation purposes."""
    rng = np.random.default_rng(seed)

    # Environmental features
    data = {
        "ndvi": rng.uniform(0.1, 0.9, rows),
        "forest_cover": rng.uniform(0, 100, rows),
        "elevation": rng.uniform(0, 4000, rows),
        "slope": rng.uniform(0, 60, rows),
        "rainfall": rng.uniform(200, 4000, rows),
        "temperature": rng.uniform(-5, 35, rows),
        "distance_road": rng.uniform(0, 50, rows),
        "distance_water": rng.uniform(0, 20, rows),
        "development_intensity": rng.uniform(0, 1, rows),
        "climate_exposure": rng.uniform(0, 1, rows),
        "species_vulnerability": rng.uniform(0, 1, rows),
        "dependency_impact": rng.uniform(0, 1, rows),
    }

    df = pd.DataFrame(data)

    # Habitat suitability: higher NDVI, forest cover, lower development, closer to water
    habitat = (
        0.35 * df["ndvi"]
        + 0.25 * (df["forest_cover"] / 100)
        + 0.10 * (1 - df["development_intensity"])
        + 0.08 * (1 - df["distance_water"] / 20).clip(0, 1)
        + 0.07 * (df["rainfall"] / 4000).clip(0, 1)
        + 0.05 * (1 - df["climate_exposure"])
        + 0.05 * (1 - df["distance_road"] / 50).clip(0, 1)
        + 0.05 * (1 - df["slope"] / 60).clip(0, 1)
        + rng.normal(0, 0.05, rows)
    )
    df["habitat_target"] = np.clip(habitat, 0, 1)

    # Biodiversity risk: inverse of habitat + vulnerability factors
    risk = (
        0.30 * (1 - df["habitat_target"])
        + 0.20 * df["development_intensity"]
        + 0.15 * df["climate_exposure"]
        + 0.15 * df["species_vulnerability"]
        + 0.10 * df["dependency_impact"]
        + 0.10 * (1 - df["forest_cover"] / 100)
        + rng.normal(0, 0.03, rows)
    )
    df["risk_target"] = np.clip(risk, 0, 1)

    return df


def train_large_models(data: pd.DataFrame, cfg: ModelConfig) -> ModelBundle:
    """Train Random Forest for habitat and XGBoost for risk."""
    X = data[FEATURES]
    y_habitat = data["habitat_target"]
    y_risk = data["risk_target"]

    X_train, X_test, y_habitat_train, y_habitat_test, y_risk_train, y_risk_test = train_test_split(
        X, y_habitat, y_risk, test_size=0.2, random_state=cfg.random_state
    )

    # Habitat model - Random Forest
    habitat_model = RandomForestRegressor(
        n_estimators=cfg.rf_estimators,
        max_depth=cfg.rf_depth,
        random_state=cfg.random_state,
        n_jobs=-1,
    )
    habitat_model.fit(X_train, y_habitat_train)
    habitat_pred = habitat_model.predict(X_test)
    habitat_mae = mean_absolute_error(y_habitat_test, habitat_pred)
    habitat_r2 = r2_score(y_habitat_test, habitat_pred)

    # Risk model - XGBoost
    risk_model = XGBRegressor(
        n_estimators=cfg.xgb_estimators,
        max_depth=cfg.xgb_depth,
        learning_rate=cfg.xgb_lr,
        random_state=cfg.random_state,
        n_jobs=-1,
        verbosity=0,
    )
    risk_model.fit(X_train, y_risk_train)
    risk_pred = risk_model.predict(X_test)
    risk_mae = mean_absolute_error(y_risk_test, risk_pred)
    risk_r2 = r2_score(y_risk_test, risk_pred)

    metrics = {
        "habitat_mae": float(habitat_mae),
        "habitat_r2": float(habitat_r2),
        "risk_mae": float(risk_mae),
        "risk_r2": float(risk_r2),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
    }

    return ModelBundle(
        habitat_model=habitat_model,
        risk_model=risk_model,
        metrics=metrics,
        X_train=X_train,
        X_test=X_test,
        y_habitat_train=y_habitat_train,
        y_habitat_test=y_habitat_test,
        y_risk_train=y_risk_train,
        y_risk_test=y_risk_test,
    )