"""
STEP 7 — EXPLAINABLE AI (XAI): LLM-powered Outlet Intelligence Narratives
==========================================================================
Purpose : For each outlet, generate a plain-English business explanation of:
  • Why the model gave it that specific predicted potential score
  • Which factors drove the prediction up or down
  • How local spatial conditions influenced the result
  • What the budget allocation means in practical terms

Uses the Anthropic Claude API (claude-sonnet-4-20250514).

Run     : python 07_xai_explanations.py --outlet_id OUT_001
          python 07_xai_explanations.py --all   (slow — generates for every outlet)
"""

import os, json, logging, argparse, time
import pandas as pd
import numpy as np
import anthropic

logging.basicConfig(level=logging.INFO, format="%(asctime)s [XAI] %(message)s")
log = logging.getLogger(__name__)

GOLD_DIR    = "gold"
OUTPUTS_DIR = "outputs"
XAI_DIR     = "outputs/xai_explanations"
os.makedirs(XAI_DIR, exist_ok=True)

# ── Load Anthropic client ──────────────────────────────────────────────────────
client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))


def load_data():
    preds      = pd.read_csv(os.path.join(GOLD_DIR, "predictions_full.csv"))
    gold       = pd.read_csv(os.path.join(GOLD_DIR, "outlets_gold.csv"))
    importances = json.load(open("models/feature_importance.json"))

    # Merge western allocations if available
    alloc_path = os.path.join(GOLD_DIR, "western_allocations_full.csv")
    if os.path.exists(alloc_path):
        alloc = pd.read_csv(alloc_path)[["Outlet_ID","Trade_Spend_LKR","expected_volume_gain_L"]]
        preds = preds.merge(alloc, on="Outlet_ID", how="left")

    merged = preds.merge(gold, on="Outlet_ID", how="left", suffixes=("","_geo"))
    return merged, importances


def build_outlet_context(row: pd.Series, importances: dict) -> dict:
    """Assemble all signals for one outlet into a structured context dict."""

    # Top 5 model drivers for this outlet (feature × model weight)
    top_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:8]

    # Spatial signals
    spatial = {
        "total_poi_score_gaussian"    : float(row.get("total_gaussian_poi_score", 0)),
        "total_poi_count"             : int(row.get("total_poi_count", 0)),
        "bus_stops_nearby"            : int(row.get("poi_count_bus_stop", 0)),
        "schools_nearby"              : int(row.get("poi_count_school", 0)),
        "markets_nearby"              : int(row.get("poi_count_market", 0)),
        "competitor_count"            : int(row.get("competitor_count", 0)),
        "market_isolation_score"      : float(row.get("market_isolation_score", 0)),
        "competitor_density_score"    : float(row.get("competitor_density_score", 0)),
    }

    # Historical performance signals
    historical = {
        "avg_monthly_sales_L"         : float(row.get("hist_mean_monthly", 0)),
        "max_monthly_sales_L"         : float(row.get("hist_max_monthly", 0)),
        "sales_trend"                 : "upward" if row.get("sales_trend_slope",0) > 0 else "downward",
        "likely_supply_constrained"   : bool(row.get("censoring_indicator", 0)),
        "january_sales_ratio"         : float(row.get("jan_ratio", 1.0)),
        "distributor_jan_seasonality" : float(row.get("dist_jan_seasonality", 1.0)),
    }

    # Prediction
    prediction = {
        "predicted_max_monthly_liters": float(row.get("Maximum_Monthly_Liters", 0)),
        "incremental_vs_historical_L" : float(row.get("incremental_potential", 0)),
        "province"                    : str(row.get("Province","Unknown")),
        "distributor"                 : str(row.get("Distributor_ID","Unknown")),
        "outlet_type"                 : str(row.get("Outlet_Type", "Unknown")),
    }

    # Budget allocation (Western Province only)
    budget = {}
    if "Trade_Spend_LKR" in row and not pd.isna(row.get("Trade_Spend_LKR")):
        budget = {
            "allocated_lkr"        : float(row.get("Trade_Spend_LKR", 0)),
            "expected_gain_L"      : float(row.get("expected_volume_gain_L", 0)),
        }

    return {
        "outlet_id"  : str(row["Outlet_ID"]),
        "prediction" : prediction,
        "spatial"    : spatial,
        "historical" : historical,
        "top_model_drivers": [f[0] for f in top_features],
        "budget"     : budget,
    }


def generate_explanation(context: dict) -> str:
    """Call Claude API to generate a business-readable explanation."""

    cache_path = os.path.join(XAI_DIR, f"{context['outlet_id']}.json")
    if os.path.exists(cache_path):
        return json.load(open(cache_path))["explanation"]

    pred  = context["prediction"]
    hist  = context["historical"]
    spat  = context["spatial"]
    bdgt  = context.get("budget", {})

    system_prompt = """You are an expert trade marketing analyst for a leading Sri Lankan beverage company.
Your job is to explain AI model predictions to non-technical sales managers and business leaders.
Write in clear, direct business language — no mathematical jargon, no technical terms.
Be concise (150–200 words). Start with the most important insight."""

    user_prompt = f"""Explain the sales potential prediction for outlet {context['outlet_id']} 
in plain business language for a regional sales manager.

OUTLET DETAILS:
- Type: {pred['outlet_type']} in {pred['province']} Province
- Distributor: {pred['distributor']}

MODEL PREDICTION:
- Predicted Maximum Monthly Volume: {pred['predicted_max_monthly_liters']:.0f} litres
- Current Historical Average: {hist['avg_monthly_sales_L']:.0f} litres/month  
- Incremental Opportunity: {pred['incremental_vs_historical_L']:.0f} extra litres

KEY MODEL DRIVERS (in order of importance): {', '.join(context['top_model_drivers'][:5])}

LOCAL ENVIRONMENT:
- Nearby Points of Interest: {spat['total_poi_count']} (bus stops: {spat['bus_stops_nearby']}, markets: {spat['markets_nearby']}, schools: {spat['schools_nearby']})
- Nearby Competitors: {spat['competitor_count']}
- Market Isolation Score: {spat['market_isolation_score']:.2f} (higher = less competition)
- Sales Trend: {hist['sales_trend']}
- Supply-Constrained History: {'Yes — outlet may have been held back by stock/credit limits' if hist['likely_supply_constrained'] else 'No'}
- January Seasonality Uplift: {hist['january_sales_ratio']:.2f}x vs annual average

{f"BUDGET ALLOCATION: LKR {bdgt['allocated_lkr']:,.0f} allocated → expected to generate {bdgt['expected_gain_L']:.0f} additional litres" if bdgt else ""}

Write the explanation as 3 short paragraphs:
1. Why this outlet received this score
2. Key factors that pushed the prediction up or down  
3. What action the sales team should take"""

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=350,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}]
        )
        explanation = response.content[0].text

        # Cache result
        json.dump({"outlet_id": context["outlet_id"], "explanation": explanation,
                   "context": context},
                  open(cache_path, "w"), indent=2)
        return explanation

    except Exception as e:
        log.error(f"API error for {context['outlet_id']}: {e}")
        return fallback_explanation(context)


def fallback_explanation(context: dict) -> str:
    """Rule-based explanation when API is unavailable."""
    pred = context["prediction"]
    hist = context["historical"]
    spat = context["spatial"]

    uplift_pct = ((pred["predicted_max_monthly_liters"] / max(hist["avg_monthly_sales_L"], 1)) - 1) * 100

    reasons_up   = []
    reasons_down = []

    if spat["total_poi_count"] > 5:
        reasons_up.append(f"high footfall area ({spat['total_poi_count']} nearby POIs)")
    if spat["market_isolation_score"] > 0.5:
        reasons_up.append("low local competition")
    if hist["likely_supply_constrained"]:
        reasons_up.append("historical supply constraints masking true demand")
    if hist["january_sales_ratio"] > 1.1:
        reasons_up.append(f"January is a strong month ({hist['january_sales_ratio']:.1f}x uplift)")

    if spat["competitor_count"] > 6:
        reasons_down.append(f"high competition ({spat['competitor_count']} competitors nearby)")
    if hist["sales_trend"] == "downward":
        reasons_down.append("declining sales trend")

    explanation = (
        f"This {pred['outlet_type']} in {pred['province']} Province is predicted to achieve "
        f"{pred['predicted_max_monthly_liters']:.0f} litres/month — {uplift_pct:.0f}% above its "
        f"current {hist['avg_monthly_sales_L']:.0f} L average.\n\n"
    )
    if reasons_up:
        explanation += f"Positive factors: {', '.join(reasons_up)}. "
    if reasons_down:
        explanation += f"Limiting factors: {', '.join(reasons_down)}. "
    explanation += (
        f"\n\nAction: {"Prioritise stock replenishment and promotional support to unlock constrained demand." if hist['likely_supply_constrained'] else "Maintain standard service and monitor response to promotions."}"
    )
    return explanation


def get_outlet_explanation(outlet_id: str) -> dict:
    """Public function called by the web app for a single outlet."""
    df, importances = load_data()
    row = df[df["Outlet_ID"] == outlet_id]
    if len(row) == 0:
        return {"error": f"Outlet {outlet_id} not found"}

    context     = build_outlet_context(row.iloc[0], importances)
    explanation = generate_explanation(context)
    return {"outlet_id": outlet_id, "context": context, "explanation": explanation}


def run(outlet_id: str = None, run_all: bool = False):
    df, importances = load_data()

    if outlet_id:
        result = get_outlet_explanation(outlet_id)
        print(f"\n{'='*60}")
        print(f"Outlet: {outlet_id}")
        print(f"{'='*60}")
        print(result["explanation"])
        return result

    if run_all:
        log.info(f"Generating explanations for {len(df):,} outlets...")
        for i, (_, row) in enumerate(df.iterrows()):
            if i % 50 == 0:
                log.info(f"  {i}/{len(df)}...")
            ctx = build_outlet_context(row, importances)
            generate_explanation(ctx)
            time.sleep(0.5)   # rate limit
        log.info("All explanations generated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outlet_id", type=str, default=None)
    parser.add_argument("--all",       action="store_true")
    args = parser.parse_args()
    run(outlet_id=args.outlet_id, run_all=args.all)
