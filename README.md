# CyberRiders — Data Storm v7.0 Final Round

## Setup

```bash
# 1. Install dependencies
pip install pandas numpy scipy scikit-learn xgboost lightgbm \
            requests streamlit plotly anthropic joblib

# 2. Place the 4 competition data files in data/raw/
mkdir -p data/raw
# Copy these 4 files into data/raw/:
#   transactions_history_final.csv
#   outlet_master.csv
#   distributor_seasonality_details.csv
#   holiday_list.csv

# 3. Set Anthropic API key (for XAI explanations)
export ANTHROPIC_API_KEY="your-key-here"

# 4. Run full pipeline
python run_pipeline.py

# 5. Launch web app
streamlit run 08_app.py
```

## Pipeline Steps

| File | Purpose |
|------|---------|
| `01_bronze_ingestion.py` | Raw CSV ingestion, schema checks, idempotency |
| `02_silver_cleaning.py` | Data cleaning, outlier handling, rejected records |
| `03_gold_poi_features.py` | Overpass API POI scraping + distance-decay features |
| `04_feature_engineering.py` | Transaction aggregation, censoring detection |
| `05_model_train_predict.py` | Tobit + XGBoost + LightGBM ensemble |
| `06_budget_optimization.py` | LKR 5M greedy marginal-ROI allocation |
| `07_xai_explanations.py` | Claude API outlet explanations |
| `08_app.py` | Streamlit web app |

## Outputs

- `outputs/cyberRiders_predictions.csv` — Outlet_ID + Maximum_Monthly_Liters
- `outputs/cyberRiders_budget_allocations.csv` — Outlet_ID + Trade_Spend_Allocation_LKR

## Key Design Decisions

**Censored Demand (Tobit):** Outlets with unusually low sales variance are flagged
as supply-constrained. Tobit regression explicitly models this left-censoring to
recover true latent demand.

**Distance Decay:** Gaussian decay `exp(-d²/2σ²)` with σ=300m gives POIs close to
the outlet exponentially more influence than distant ones.

**Budget Optimization:** Greedy marginal-ROI algorithm allocates in minimum-spend
increments, always choosing the outlet with highest marginal return at current
spend level. Exploits concavity of Cobb-Douglas response function.
