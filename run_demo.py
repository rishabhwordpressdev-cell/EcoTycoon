#!/usr/bin/env python
"""Command-line demo for Eco Tycoon."""
from __future__ import annotations

from eco_tycoon.config import FEATURES, SCENARIOS, ModelConfig
from eco_tycoon.models import build_large_demo_dataset, train_large_models
from eco_tycoon.analytics import analyze_site, scenario_comparison
import os


def main():
    # Allow environment variable overrides
    demo_rows = int(os.getenv("ECO_DEMO_ROWS", "50000"))
    rf_trees = int(os.getenv("ECO_RF_TREES", "600"))
    xgb_trees = int(os.getenv("ECO_XGB_TREES", "800"))

    print(f"Eco Tycoon - Command Line Demo")
    print(f"Configuration: {demo_rows:,} rows, {rf_trees} RF trees, {xgb_trees} XGB estimators")
    print("-" * 60)

    cfg = ModelConfig(demo_rows=demo_rows, rf_estimators=rf_trees, xgb_estimators=xgb_trees)
    print("Building synthetic training dataset...")
    data = build_large_demo_dataset(demo_rows, cfg.random_state)

    print("Training models...")
    bundle = train_large_models(data, cfg)

    # Default site for demo
    site = {
        "ndvi": 0.62,
        "forest_cover": 68.0,
        "elevation": 420.0,
        "slope": 12.0,
        "rainfall": 1400.0,
        "temperature": 23.0,
        "distance_road": 3.0,
        "distance_water": 1.8,
        "development_intensity": 0.65,
        "climate_exposure": 0.40,
        "species_vulnerability": 0.55,
        "dependency_impact": 0.45,
    }

    scenario = "Mitigation"
    years = 10

    print(f"\nRunning analysis: {scenario} scenario, {years} years")
    result = analyze_site(site, scenario, years, bundle.habitat_model, bundle.risk_model)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Future habitat suitability:     {result.future_habitat:.3f}")
    print(f"Habitat loss:                   {result.habitat_loss_pct:.2f}%")
    print(f"Fragmentation index:            {result.fragmentation_index:.3f}")
    print(f"Final biodiversity risk score:  {result.final_risk_score:.3f}")
    print(f"Risk category:                  {result.risk_category}")
    print(f"\nRecommendation:")
    print(f"  {result.recommendation}")

    print("\n" + "-" * 60)
    print("MODEL VALIDATION METRICS")
    print("-" * 60)
    for key, value in bundle.metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    print("\n" + "-" * 60)
    print("SCENARIO COMPARISON (all scenarios, 0/5/10/20 years)")
    print("-" * 60)
    comp = scenario_comparison(site, bundle.habitat_model, bundle.risk_model)
    print(comp.to_string(index=False))


if __name__ == "__main__":
    main()