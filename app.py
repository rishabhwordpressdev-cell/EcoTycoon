from __future__ import annotations
import pandas as pd
import streamlit as st

from eco_tycoon.config import FEATURES, SCENARIOS, ModelConfig
from eco_tycoon.models import build_large_demo_dataset, train_large_models
from eco_tycoon.analytics import analyze_site, scenario_comparison
from eco_tycoon.data_sources import GBIFClient, GBIFQuery, load_tabular_file
from eco_tycoon.preprocess import clean_occurrences, spatial_thin_occurrences
from eco_tycoon.network import build_dependency_graph, dependency_scores
from eco_tycoon.explainability import feature_importance
from eco_tycoon.advanced_models import build_demo_time_series, train_lstm_regressor, predict_lstm, train_maxent, predict_maxent
from eco_tycoon.spatial import fragmentation_metrics

st.set_page_config(page_title="Eco Tycoon", page_icon="🌿", layout="wide")
st.title("🌿 Eco Tycoon")
st.caption("AI-Based Predictive Biodiversity Impact Assessment Platform")

@st.cache_resource(show_spinner=False)
def train_cached(rows:int, rf_trees:int, xgb_trees:int):
    cfg=ModelConfig(demo_rows=rows,rf_estimators=rf_trees,xgb_estimators=xgb_trees)
    data=build_large_demo_dataset(rows,cfg.random_state)
    return train_large_models(data,cfg)

def defaults():
    return {"ndvi":0.62,"forest_cover":68.0,"elevation":420.0,"slope":12.0,"rainfall":1400.0,
            "temperature":23.0,"distance_road":3.0,"distance_water":1.8,"development_intensity":0.65,
            "climate_exposure":0.40,"species_vulnerability":0.55,"dependency_impact":0.45}

with st.sidebar:
    st.header("Large Model Configuration")
    rows=st.select_slider("Training rows",[5000,10000,25000,50000],value=50000)
    rf_trees=st.select_slider("Random Forest trees",[150,300,600],value=600)
    xgb_trees=st.select_slider("XGBoost estimators",[200,400,800],value=800)
    st.info("Default research/demo mode: 50,000 synthetic rows + 600 RF trees + 800 XGBoost estimators. Replace synthetic data with validated research data before real environmental use.")

tabs=st.tabs(["Project Analysis","Scenario Comparison","Data & GBIF","Dependencies","Spatial Metrics","MaxEnt","LSTM Forecast","About"])

with tabs[0]:
    st.subheader("Project / Site Analytics")
    left,right=st.columns(2); d=defaults(); site={}
    with left:
        for label,key in [("NDVI (0–1)","ndvi"),("Forest cover (%)","forest_cover"),("Elevation (m)","elevation"),("Slope (degrees)","slope"),("Rainfall (mm)","rainfall"),("Temperature (°C)","temperature")]:
            site[key]=st.number_input(label,value=float(d[key]))
    with right:
        for label,key in [("Distance from road (km)","distance_road"),("Distance from water (km)","distance_water"),("Development intensity (0–1)","development_intensity"),("Climate exposure (0–1)","climate_exposure"),("Species vulnerability (0–1)","species_vulnerability"),("Dependency impact (0–1)","dependency_impact")]:
            kwargs={"value":float(d[key])}
            if "(0–1)" in label: kwargs.update(min_value=0.0,max_value=1.0)
            site[key]=st.number_input(label,**kwargs)
    scenario=st.selectbox("Development scenario",list(SCENARIOS),index=1)
    years=st.selectbox("Time horizon",[0,5,10,20],index=2)
    if st.button("🚀 Run Biodiversity Analysis",type="primary"):
        if not 0<=site["ndvi"]<=1 or not 0<=site["forest_cover"]<=100:
            st.error("Check NDVI (0–1) and forest cover (0–100)."); st.stop()
        with st.spinner("Training/loading large models..."):
            bundle=train_cached(rows,rf_trees,xgb_trees)
        result=analyze_site(site,scenario,years,bundle.habitat_model,bundle.risk_model)
        c=st.columns(4)
        c[0].metric("Future habitat",f"{result.future_habitat:.3f}")
        c[1].metric("Habitat loss",f"{result.habitat_loss_pct:.2f}%")
        c[2].metric("Fragmentation",f"{result.fragmentation_index:.3f}")
        c[3].metric("Biodiversity risk",f"{result.final_risk_score:.3f}")
        st.subheader(f"Risk category: {result.risk_category}")
        st.write(result.recommendation)
        st.write("### Model validation"); st.json(bundle.metrics)
        st.write("### Top risk-model features")
        fi=feature_importance(bundle.risk_model,FEATURES).head(10)
        st.bar_chart(fi.set_index("feature")["importance"])
        st.write("### Uncertainty indicators")
        st.json({"habitat_model_spread":result.habitat_uncertainty,
                  "risk_indicator":result.risk_uncertainty,
                  "note":"These are model-spread indicators, not calibrated confidence intervals."})

with tabs[1]:
    st.subheader("BAU vs Mitigation vs Conservation")
    st.write("Runs all three scenarios at 0, 5, 10 and 20 years using the same analytics pipeline.")
    if st.button("Compare scenarios"):
        b=train_cached(rows,rf_trees,xgb_trees)
        comp=scenario_comparison(defaults(),b.habitat_model,b.risk_model)
        st.dataframe(comp,use_container_width=True)
        st.write("Risk over time"); st.line_chart(comp.pivot(index="years",columns="scenario",values="risk_score"))
        st.write("Habitat loss over time"); st.line_chart(comp.pivot(index="years",columns="scenario",values="habitat_loss_pct"))

with tabs[2]:
    st.subheader("Data Ingestion")
    up=st.file_uploader("Upload environmental CSV/Parquet/JSON",type=["csv","parquet","json"])
    if up:
        df=load_tabular_file(up); st.success(f"Loaded {len(df):,} records."); st.dataframe(df.head(50),use_container_width=True)
    st.markdown("#### GBIF public occurrence search")
    species=st.text_input("Scientific name (optional)",placeholder="Panthera tigris")
    country=st.text_input("GBIF country code (optional)",placeholder="IN")
    limit=st.number_input("Maximum occurrence records",min_value=100,max_value=100000,value=5000,step=100)
    if st.button("Fetch GBIF records"):
        try:
            raw=GBIFClient().search(GBIFQuery(scientific_name=species or None,country=country or None,limit=int(limit)))
            cleaned=clean_occurrences(raw); thinned=spatial_thin_occurrences(cleaned)
            st.success(f"Fetched {len(raw):,}; cleaned {len(cleaned):,}; spatially thinned {len(thinned):,}.")
            st.dataframe(thinned.head(100),use_container_width=True)
            st.download_button("Download cleaned GBIF CSV",thinned.to_csv(index=False),"gbif_cleaned.csv","text/csv")
        except Exception as exc:
            st.error(f"GBIF request failed: {exc}")

with tabs[3]:
    st.subheader("Species Dependency Network")
    dep=st.file_uploader("Upload dependency edge CSV (source,target,weight)",type=["csv"],key="dep")
    if dep:
        try:
            edges=pd.read_csv(dep); graph=build_dependency_graph(edges); scores=dependency_scores(graph)
            st.metric("Species / nodes",graph.number_of_nodes()); st.metric("Interactions / edges",graph.number_of_edges())
            st.dataframe(scores.head(50),use_container_width=True)
        except Exception as exc: st.error(f"Dependency analysis failed: {exc}")

with tabs[4]:
    st.subheader("Optional LSTM Temporal Forecast")
    st.write("Use LSTM only when you have a sufficiently long environmental time series.")
    seq=build_demo_time_series()
    st.line_chart(seq.set_index("time_index")[["ndvi","climate_exposure","risk_target"]])
    if st.button("Train LSTM demo"):
        try:
            bundle=train_lstm_regressor(seq,["ndvi","climate_exposure","rainfall"],"risk_target",
                                        sequence_length=12,epochs=20,hidden_size=64,num_layers=2)
            pred=predict_lstm(bundle,seq)
            st.success(f"Forecast: {pred:.4f} | Training loss: {bundle['train_loss']:.6f}")
        except ImportError: st.error("Install requirements-advanced.txt to use PyTorch LSTM.")
        except Exception as exc: st.error(f"LSTM failed: {exc}")

with tabs[4]:
    st.subheader("GIS Fragmentation & Connectivity Metrics")
    st.write("Upload a 2-D CSV matrix of habitat-suitability values (0–1) to calculate patch count, largest-patch share, connectivity and fragmentation.")
    grid_file = st.file_uploader("Suitability grid CSV", type=["csv"], key="grid")
    if grid_file:
        try:
            grid = pd.read_csv(grid_file, header=None).to_numpy(dtype=float)
            threshold = st.slider("Habitat suitability threshold", 0.1, 0.9, 0.5, 0.05)
            metrics = fragmentation_metrics(grid, threshold=threshold)
            st.json(metrics)
        except Exception as exc:
            st.error(f"Spatial metric calculation failed: {exc}")

with tabs[5]:
    st.subheader("MaxEnt Species Distribution Model")
    st.write("MaxEnt is used for presence/background species-distribution modelling when reliable absence records are unavailable. This advanced option requires `elapid`.")
    mex = st.file_uploader("MaxEnt training CSV", type=["csv"], key="maxent")
    if mex:
        try:
            mdf = pd.read_csv(mex)
            required = FEATURES + ["presence"]
            missing = [c for c in required if c not in mdf.columns]
            if missing:
                st.error("Missing columns: " + ", ".join(missing))
            else:
                if st.button("Train MaxEnt"):
                    X = mdf[FEATURES].apply(pd.to_numeric, errors="coerce").fillna(mdf[FEATURES].median())
                    y = pd.to_numeric(mdf["presence"]).astype(int)
                    model = train_maxent(X, y, beta_multiplier=1.5)
                    pred = predict_maxent(model, X)
                    st.success(f"MaxEnt trained on {len(X):,} records.")
                    st.metric("Mean training prediction", f"{float(pred.mean()):.4f}")
                    fi = pd.DataFrame({"feature": FEATURES, "importance": getattr(model, "feature_importances_", [0]*len(FEATURES))})
                    st.dataframe(fi.sort_values("importance", ascending=False), use_container_width=True)
        except ImportError:
            st.error("Install requirements-advanced.txt to use MaxEnt.")
        except Exception as exc:
            st.error(f"MaxEnt training failed: {exc}")

with tabs[7]:
    st.subheader("Methods included")
    st.markdown("""
**Machine-learning**
- Random Forest Regressor → habitat suitability.
- XGBoost Regressor → biodiversity risk.
- MaxEnt → presence-only species distribution modelling through `elapid` (advanced option).
- LSTM → temporal forecasting through PyTorch (advanced option).

**GIS / ecological analytics**
- Habitat-loss percentage.
- Raster fragmentation and connectivity metrics.
- Species dependency-network analysis.
- BAU / Mitigation / Conservation scenario simulation.
- Risk fusion: habitat loss + fragmentation + species vulnerability + climate exposure + dependency impact.
- Feature importance and optional SHAP.
- Ensemble-spread uncertainty indicator.

**Large-data layer**
- DuckDB analytical schema.
- Parquet/Arrow interchange.
- Chunked CSV ingestion.
- Fact/dimension data model for project, species, scenario, environmental observations, occurrences and predictions.

The default large-model training set is synthetic and is only for software validation. Real research runs should use validated GBIF, Dynamic World, Global Forest Watch and Earth Engine-derived features.
""")
