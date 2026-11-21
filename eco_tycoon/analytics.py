from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from .config import FEATURES, SCENARIOS


@dataclass
class AnalysisResult:
    future_habitat: float
    habitat_loss_pct: float
    fragmentation_index: float
    final_risk_score: float
    risk_category: str
    recommendation: str
    habitat_uncertainty: float
    risk_uncertainty: float


def _apply_scenario(site: dict, scenario_name: str, years: int) -> dict:
    """Apply scenario modifiers to site features over time."""
    if scenario_name not in SCENARIOS:
        scenario_name = "Business as Usual"
    scenario = SCENARIOS[scenario_name]

    # Time scaling - effects compound over years
    time_factor = 1.0 + (years / 20.0) * 0.5  # 0-50% additional effect over 20 years

    future_site = site.copy()
    future_site["ndvi"] = np.clip(
        site["ndvi"] * scenario["ndvi_factor"] ** time_factor, 0, 1
    )
    future_site["forest_cover"] = np.clip(
        site["forest_cover"] * scenario["forest_cover_factor"] ** time_factor, 0, 100
    )
    future_site["development_intensity"] = np.clip(
        site["development_intensity"] * scenario["development_intensity_factor"] ** time_factor, 0, 1
    )
    future_site["climate_exposure"] = np.clip(
        site["climate_exposure"] * scenario["climate_exposure_factor"] ** time_factor, 0, 1
    )
    return future_site


def _calculate_fragmentation(site: dict) -> float:
    """Calculate fragmentation index from site features."""
    # Fragmentation driven by development, roads, low forest cover
    frag = (
        0.40 * site["development_intensity"]
        + 0.25 * (1 - np.clip(site["distance_road"] / 10, 0, 1))
        + 0.20 * (1 - site["forest_cover"] / 100)
        + 0.15 * (site["slope"] / 60)
    )
    return float(np.clip(frag, 0, 1))


def analyze_site(
    site: dict,
    scenario_name: str,
    years: int,
    habitat_model: RandomForestRegressor,
    risk_model: XGBRegressor,
) -> AnalysisResult:
    """Run the full biodiversity impact analysis pipeline."""
    # Baseline habitat prediction
    X_baseline = pd.DataFrame([site])[FEATURES]
    baseline_habitat = float(habitat_model.predict(X_baseline)[0])

    # Future site under scenario
    future_site = _apply_scenario(site, scenario_name, years)
    X_future = pd.DataFrame([future_site])[FEATURES]
    future_habitat = float(habitat_model.predict(X_future)[0])

    # Habitat loss
    habitat_loss_pct = 0.0
    if baseline_habitat > 0:
        habitat_loss_pct = max(0, (baseline_habitat - future_habitat) / baseline_habitat * 100)

    # Fragmentation
    fragmentation_index = _calculate_fragmentation(future_site)

    # Risk prediction
    risk_score = float(risk_model.predict(X_future)[0])
    risk_score = np.clip(risk_score, 0, 1)

    # Risk fusion with explicit components
    habitat_loss_component = habitat_loss_pct / 100
    vulnerability_component = future_site["species_vulnerability"]
    climate_component = future_site["climate_exposure"]
    dependency_component = future_site["dependency_impact"]

    final_risk = (
        0.30 * risk_score
        + 0.20 * habitat_loss_component
        + 0.15 * fragmentation_index
        + 0.15 * vulnerability_component
        + 0.10 * climate_component
        + 0.10 * dependency_component
    )
    final_risk = float(np.clip(final_risk, 0, 1))

    # Risk category
    if final_risk < 0.30:
        risk_category = "Low"
    elif final_risk < 0.55:
        risk_category = "Moderate"
    elif final_risk < 0.75:
        risk_category = "High"
    else:
        risk_category = "Critical"

    # Recommendations
    recommendations = {
        "Low": "Proceed with standard environmental monitoring. Maintain current conservation measures.",
        "Moderate": "Implement mitigation measures: habitat buffers, wildlife corridors, phased development. Monitor quarterly.",
        "High": "Require detailed EIA, redesign to avoid core habitat, mandatory offsets, continuous monitoring.",
        "Critical": "Recommend project rejection or major redesign. Critical habitat loss likely. Seek alternative sites.",
    }

    # Uncertainty indicators (model spread)
    habitat_uncertainty = float(
        np.std([tree.predict(X_future)[0] for tree in habitat_model.estimators_[:50]])
    ) if hasattr(habitat_model, "estimators_") else 0.05
    risk_uncertainty = 0.05  # XGBoost doesn't easily expose tree predictions

    return AnalysisResult(
        future_habitat=future_habitat,
        habitat_loss_pct=habitat_loss_pct,
        fragmentation_index=fragmentation_index,
        final_risk_score=final_risk,
        risk_category=risk_category,
        recommendation=recommendations[risk_category],
        habitat_uncertainty=habitat_uncertainty,
        risk_uncertainty=risk_uncertainty,
    )


def scenario_comparison(
    site: dict,
    habitat_model: RandomForestRegressor,
    risk_model: XGBRegressor,
) -> pd.DataFrame:
    """Compare all scenarios across time horizons."""
    results = []
    for scenario_name in SCENARIOS.keys():
        for years in [0, 5, 10, 20]:
            result = analyze_site(site, scenario_name, years, habitat_model, risk_model)
            results.append(
                {
                    "scenario": scenario_name,
                    "years": years,
                    "future_habitat": result.future_habitat,
                    "habitat_loss_pct": result.habitat_loss_pct,
                    "fragmentation_index": result.fragmentation_index,
                    "risk_score": result.final_risk_score,
                    "risk_category": result.risk_category,
                }
            )
    return pd.DataFrame(results)