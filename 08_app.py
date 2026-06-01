"""
STEP 8 — OUTLET INTELLIGENCE WEB APP (Streamlit)
=================================================
Purpose : Interactive web app for business users to:
  • Browse all outlet predictions (filterable by province / distributor)
  • See a map of outlet locations coloured by potential
  • Drill into any outlet for its predicted score + AI explanation

Run     : streamlit run 08_app.py
Install : pip install streamlit plotly anthropic pandas numpy
"""

import os, json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Outlet Intelligence | CyberRiders",
    page_icon="🥤",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  [data-testid="stSidebar"] { background: #0d1b2a; color: white; }
  h1 { color: #00c8ff; }
  h2, h3 { color: #00c8ff; }
  .metric-card {
    background: #0d1b2a; border-radius: 12px; padding: 16px;
    border: 1px solid #00c8ff33; margin: 6px 0;
  }
  .stMetric label { color: #aaa !important; }
  .stMetric [data-testid="stMetricValue"] { color: #00c8ff !important; font-size: 28px !important; }
  .explanation-box {
    background: #0d1b2a; border-left: 4px solid #00c8ff;
    padding: 16px; border-radius: 8px; color: #e0e0e0;
    font-size: 15px; line-height: 1.7;
  }
</style>
""", unsafe_allow_html=True)


# ── Data loading (cached) ──────────────────────────────────────────────────────
@st.cache_data
def load_data():
    base = os.path.dirname(__file__) if "__file__" in dir() else "."
    gold_dir = os.path.join(base, "gold")
    out_dir  = os.path.join(base, "outputs")

    preds = pd.read_csv(os.path.join(gold_dir, "predictions_full.csv"))
    gold  = pd.read_csv(os.path.join(gold_dir, "outlets_gold.csv"))

    alloc_path = os.path.join(gold_dir, "western_allocations_full.csv")
    if os.path.exists(alloc_path):
<<<<<<< HEAD
        alloc = pd.read_csv(alloc_path)
        # Gracefully accommodate the evaluation-compliant column names
        if "Trade Spend Allocation (LKR)" in alloc.columns:
            alloc = alloc.rename(columns={"Trade Spend Allocation (LKR)": "Trade_Spend_LKR"})

        alloc = alloc[["Outlet_ID", "Trade_Spend_LKR", "expected_volume_gain_L", "roi_litres_per_1000_lkr"]]
        preds = preds.merge(alloc, on="Outlet_ID", how="left")

    df = preds.merge(
        gold[["Outlet_ID","Outlet_Type","Latitude","Longitude"]],
        on="Outlet_ID", how="left", suffixes=("","_geo")
    )

    # Ensure Outlet_Type exists
    if "Outlet_Type" not in df.columns and "Outlet_Type_geo" in df.columns:
        df["Outlet_Type"] = df["Outlet_Type_geo"]

    return df


@st.cache_data
def load_importance():
    path = "models/feature_importance.json"
    if os.path.exists(path):
        return json.load(open(path))
    return {}


def get_cached_explanation(outlet_id: str, context: dict) -> str:
    """Load from cache or generate via Claude API."""
    cache_path = os.path.join("outputs/xai_explanations", f"{outlet_id}.json")
    if os.path.exists(cache_path):
        return json.load(open(cache_path)).get("explanation","")

    # Generate live
    try:
        from xai_explanations_module import generate_explanation
        return generate_explanation(context)
    except:
        pass

    # Fallback
    return _rule_based_explanation(context)


def _rule_based_explanation(ctx: dict) -> str:
    p = ctx.get("prediction", {})
    h = ctx.get("historical", {})
    s = ctx.get("spatial", {})
    uplift = ((p.get("predicted_max_monthly_liters",0)/max(h.get("avg_monthly_sales_L",1),1))-1)*100

    lines = []
    lines.append(
        f"**Score Rationale:** This {p.get('outlet_type','outlet')} in {p.get('province','')} "
        f"is predicted to reach **{p.get('predicted_max_monthly_liters',0):.0f} L/month** — "
        f"representing a {uplift:.0f}% uplift over its current {h.get('avg_monthly_sales_L',0):.0f} L average."
    )
    if h.get("likely_supply_constrained"):
        lines.append(
            "⚠️ **Supply Constraint Detected:** Historical sales show unusually low variance, "
            "suggesting this outlet has been hitting a stock or credit ceiling. True demand is likely higher."
        )
    if s.get("total_poi_count",0) > 5:
        lines.append(f"📍 High-traffic location with {s['total_poi_count']} nearby points of interest driving footfall.")
    if s.get("market_isolation_score",0) > 0.5:
        lines.append("✅ Low competition in the catchment area — untapped market opportunity.")
    if s.get("competitor_count",0) > 6:
        lines.append(f"⚡ Competitive pressure from {s['competitor_count']} nearby outlets may limit growth.")
    lines.append(
        "**Recommended Action:** " + (
            "Prioritise cooler replenishment and credit limit review to unlock constrained demand."
            if h.get("likely_supply_constrained") else
            "Standard promotional support. Monitor sell-through rates monthly."
        )
    )
    return "\n\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR FILTERS
# ─────────────────────────────────────────────────────────────────────────────

def sidebar_filters(df: pd.DataFrame):
    st.sidebar.image("https://img.shields.io/badge/CyberRiders-DataStorm_v7-00c8ff?style=for-the-badge", width=250)
    st.sidebar.title("🔎 Filters")

    provinces = ["All"] + sorted(df["Province"].dropna().unique().tolist())
    prov = st.sidebar.selectbox("Province", provinces)

    dists = ["All"] + sorted(df["Distributor_ID"].dropna().unique().tolist())
    dist  = st.sidebar.selectbox("Distributor", dists)

    outlet_types = ["All"] + sorted(df["Outlet_Type"].dropna().unique().tolist()) if "Outlet_Type" in df.columns else ["All"]
    otype = st.sidebar.selectbox("Outlet Type", outlet_types)

    st.sidebar.markdown("---")
    min_pot = st.sidebar.number_input("Min Predicted Potential (L)", value=0, step=100)
    st.sidebar.markdown("---")

    # Apply filters
    fdf = df.copy()
    if prov != "All":
        fdf = fdf[fdf["Province"] == prov]
    if dist != "All":
        fdf = fdf[fdf["Distributor_ID"] == dist]
    if otype != "All" and "Outlet_Type" in fdf.columns:
        fdf = fdf[fdf["Outlet_Type"] == otype]
    fdf = fdf[fdf["Maximum_Monthly_Liters"] >= min_pot]

    return fdf, prov, dist


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: OVERVIEW DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────

def page_overview(df: pd.DataFrame, fdf: pd.DataFrame):
    st.title("🥤 Outlet Intelligence Platform")
    st.caption("CyberRiders | Data Storm v7.0 Final Round")

    # KPI row
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Outlets",          f"{len(fdf):,}")
    c2.metric("Avg Potential (L/mo)",   f"{fdf['Maximum_Monthly_Liters'].mean():,.0f}")
    c3.metric("Total Potential (L/mo)", f"{fdf['Maximum_Monthly_Liters'].sum():,.0f}")
    c4.metric("Supply-Constrained",     f"{fdf['censoring_indicator'].sum():,}")
    if "Trade_Spend_LKR" in fdf.columns:
        c5.metric("Budget Allocated (LKR)", f"{fdf['Trade_Spend_LKR'].sum():,.0f}")

    st.markdown("---")

    col1, col2 = st.columns([3, 2])

    # Map
    with col1:
        st.subheader("📍 Outlet Map — Potential Score")
        map_df = fdf.dropna(subset=["Latitude","Longitude"]).head(5000)
        if len(map_df) > 0:
            fig = px.scatter_mapbox(
                map_df,
                lat="Latitude", lon="Longitude",
                color="Maximum_Monthly_Liters",
                size="Maximum_Monthly_Liters",
                size_max=15,
                hover_name="Outlet_ID",
                hover_data={
                    "Province": True,
                    "Distributor_ID": True,
                    "Maximum_Monthly_Liters": ":.0f",
                    "hist_mean_monthly": ":.0f",
                },
                color_continuous_scale="plasma",
                mapbox_style="carto-darkmatter",
                zoom=7,
                center={"lat": 7.8731, "lon": 80.7718},
                height=450,
            )
            fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0},
                              coloraxis_colorbar_title="Predicted L/mo")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No lat/lon data available for map.")

    # Distributor bar
    with col2:
        st.subheader("📊 Potential by Distributor")
        dist_agg = fdf.groupby("Distributor_ID")["Maximum_Monthly_Liters"].sum().reset_index()
        fig2 = px.bar(
            dist_agg.sort_values("Maximum_Monthly_Liters", ascending=True),
            x="Maximum_Monthly_Liters", y="Distributor_ID",
            orientation="h", color="Maximum_Monthly_Liters",
            color_continuous_scale="plasma", height=450,
            labels={"Maximum_Monthly_Liters": "Predicted Litres/Mo"},
        )
        fig2.update_layout(showlegend=False, margin={"t":10},
                           paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font_color="white")
        st.plotly_chart(fig2, use_container_width=True)

    # Distribution histogram
    st.subheader("📈 Prediction Distribution")
    fig3 = px.histogram(
        fdf, x="Maximum_Monthly_Liters", nbins=80,
        color_discrete_sequence=["#00c8ff"],
        labels={"Maximum_Monthly_Liters": "Predicted Max Monthly Litres"},
        height=300,
    )
    fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                       font_color="white")
    st.plotly_chart(fig3, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: OUTLET TABLE
# ─────────────────────────────────────────────────────────────────────────────

def page_outlets(fdf: pd.DataFrame):
    st.title("📋 Outlet Predictions")

    display_cols = ["Outlet_ID", "Province", "Distributor_ID",
                    "Maximum_Monthly_Liters", "hist_mean_monthly",
                    "incremental_potential", "censoring_indicator"]
    if "Trade_Spend_LKR" in fdf.columns:
        display_cols.append("Trade_Spend_LKR")

    display_cols = [c for c in display_cols if c in fdf.columns]
    show = fdf[display_cols].copy()

    # Rename for display
    show = show.rename(columns={
        "Maximum_Monthly_Liters" : "Predicted Max (L)",
        "hist_mean_monthly"      : "Hist Avg (L)",
        "incremental_potential"  : "Upside (L)",
        "censoring_indicator"    : "Constrained?",
        "Trade_Spend_LKR"        : "Budget (LKR)",
    })
    show = show.sort_values("Predicted Max (L)", ascending=False)
    st.dataframe(show.reset_index(drop=True), height=500, use_container_width=True)
    st.caption(f"Showing {len(show):,} outlets")

    # Export button
    csv = show.to_csv(index=False)
    st.download_button("⬇️ Download CSV", csv, "outlet_predictions.csv", "text/csv")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: OUTLET DEEP DIVE + XAI
# ─────────────────────────────────────────────────────────────────────────────

def page_outlet_detail(df: pd.DataFrame, fdf: pd.DataFrame, importances: dict):
    st.title("🔬 Outlet Deep Dive")

    outlet_ids = sorted(fdf["Outlet_ID"].tolist())
    selected   = st.selectbox("Select Outlet", outlet_ids)

    if not selected:
        st.info("Select an outlet above.")
        return

    row = df[df["Outlet_ID"] == selected].iloc[0]

    # ── Metrics ──────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Predicted Potential",    f"{row.get('Maximum_Monthly_Liters',0):,.0f} L/mo")
    c2.metric("Historical Average",     f"{row.get('hist_mean_monthly',0):,.0f} L/mo")
    c3.metric("Incremental Upside",     f"{row.get('incremental_potential',0):,.0f} L/mo")
    c4.metric("Supply Constrained",     "Yes ⚠️" if row.get("censoring_indicator",0) else "No ✅")

    if "Trade_Spend_LKR" in row and not pd.isna(row.get("Trade_Spend_LKR")):
        st.metric("Budget Allocated", f"LKR {row['Trade_Spend_LKR']:,.0f}")

    st.markdown("---")

    col1, col2 = st.columns([2, 3])

    with col1:
        st.subheader("📌 Outlet Details")
        details = {
            "Province"       : row.get("Province",""),
            "Distributor"    : row.get("Distributor_ID",""),
            "Outlet Type"    : row.get("Outlet_Type",""),
            "Competitors"    : int(row.get("competitor_count",0)),
            "Total POIs"     : int(row.get("total_poi_count",0)),
            "Bus Stops"      : int(row.get("poi_count_bus_stop",0)),
            "Markets"        : int(row.get("poi_count_market",0)),
            "Schools"        : int(row.get("poi_count_school",0)),
            "Isolation Score": f"{row.get('market_isolation_score',0):.2f}",
            "Jan Uplift"     : f"{row.get('jan_ratio',1.0):.2f}×",
        }
        for k, v in details.items():
            st.markdown(f"**{k}:** {v}")

        # Mini bar chart: model components
        pred_vals = {
            "Tobit Model": row.get("pred_tobit",0),
            "XGBoost"    : row.get("pred_xgb",0),
            "LightGBM"   : row.get("pred_lgb",0),
            "Ensemble"   : row.get("Maximum_Monthly_Liters",0),
        }
        fig_comp = px.bar(
            x=list(pred_vals.keys()), y=list(pred_vals.values()),
            color=list(pred_vals.keys()),
            labels={"x":"Model","y":"Predicted L/mo"},
            title="Model Component Predictions",
            height=250,
            color_discrete_sequence=["#4a90d9","#00c8ff","#f5a623","#ff6b6b"],
        )
        fig_comp.update_layout(showlegend=False, paper_bgcolor="rgba(0,0,0,0)",
                                plot_bgcolor="rgba(0,0,0,0)", font_color="white")
        st.plotly_chart(fig_comp, use_container_width=True)

    with col2:
        st.subheader("🤖 AI-Generated Explanation")

        # Build context for XAI
        ctx = {
            "outlet_id": selected,
            "prediction": {
                "predicted_max_monthly_liters": float(row.get("Maximum_Monthly_Liters",0)),
                "incremental_vs_historical_L" : float(row.get("incremental_potential",0)),
                "province"   : str(row.get("Province","")),
                "distributor": str(row.get("Distributor_ID","")),
                "outlet_type": str(row.get("Outlet_Type","")),
            },
            "historical": {
                "avg_monthly_sales_L"        : float(row.get("hist_mean_monthly",0)),
                "max_monthly_sales_L"        : float(row.get("hist_max_monthly",0)),
                "sales_trend"                : "upward" if row.get("sales_trend_slope",0)>0 else "downward",
                "likely_supply_constrained"  : bool(row.get("censoring_indicator",0)),
                "january_sales_ratio"        : float(row.get("jan_ratio",1.0)),
                "distributor_jan_seasonality": float(row.get("dist_jan_seasonality",1.0)),
            },
            "spatial": {
                "total_poi_count"           : int(row.get("total_poi_count",0)),
                "bus_stops_nearby"          : int(row.get("poi_count_bus_stop",0)),
                "markets_nearby"            : int(row.get("poi_count_market",0)),
                "schools_nearby"            : int(row.get("poi_count_school",0)),
                "competitor_count"          : int(row.get("competitor_count",0)),
                "market_isolation_score"    : float(row.get("market_isolation_score",0)),
                "competitor_density_score"  : float(row.get("competitor_density_score",0)),
                "total_gaussian_poi_score"  : float(row.get("total_gaussian_poi_score",0)),
            },
            "top_model_drivers": list(importances.keys())[:5] if importances else [],
            "budget": {
                "allocated_lkr"   : float(row.get("Trade_Spend_LKR",0) or 0),
                "expected_gain_L" : float(row.get("expected_volume_gain_L",0) or 0),
            } if "Trade_Spend_LKR" in row and not pd.isna(row.get("Trade_Spend_LKR")) else {},
        }

        if st.button("🔄 Generate / Refresh Explanation"):

            with st.spinner("Querying Generative AI Model Engines..."):
                # Routes directly to your model instead of using rule-based strings
                explanation = get_cached_explanation(selected, ctx)
            st.session_state[f"expl_{selected}"] = explanation

        explanation = st.session_state.get(f"expl_{selected}", None)
        if explanation is None:
            explanation = get_cached_explanation(selected, ctx)

            with st.spinner("Generating AI explanation..."):
                explanation = _rule_based_explanation(ctx)
            st.session_state[f"expl_{selected}"] = explanation

        explanation = st.session_state.get(f"expl_{selected}", _rule_based_explanation(ctx))

        st.markdown(f'<div class="explanation-box">{explanation}</div>', unsafe_allow_html=True)

    # Feature importance bar
    if importances:
        st.markdown("---")
        st.subheader("📊 Global Feature Importance (Top 15)")
        top15 = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True)[:15])
        fig_imp = px.bar(
            x=list(top15.values()), y=list(top15.keys()),
            orientation="h", color=list(top15.values()),
            color_continuous_scale="plasma", height=400,
            labels={"x":"Importance","y":"Feature"},
        )
        fig_imp.update_layout(showlegend=False, yaxis=dict(autorange="reversed"),
                               paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               font_color="white")
        st.plotly_chart(fig_imp, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: BUDGET DASHBOARD (Western Province)
# ─────────────────────────────────────────────────────────────────────────────

def page_budget(df: pd.DataFrame):
    st.title("💰 Budget Allocation — Western Province (LKR 5M)")

    western = df[(df["Province"] == "Western") & df.get("Trade_Spend_LKR", pd.Series()).notna() if "Trade_Spend_LKR" in df.columns else df["Province"] == "Western"]

    if "Trade_Spend_LKR" not in df.columns:
        st.warning("Budget allocation file not found. Run 06_budget_optimization.py first.")
        return

    western = df[df["Province"] == "Western"].dropna(subset=["Trade_Spend_LKR"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Spent",        f"LKR {western['Trade_Spend_LKR'].sum():,.0f}")
    c2.metric("Outlets Supported",  f"{(western['Trade_Spend_LKR']>0).sum():,}")
    c3.metric("Expected Gain",      f"{western['expected_volume_gain_L'].sum():,.0f} L")
    c4.metric("Overall ROI",        f"{western['expected_volume_gain_L'].sum()/(western['Trade_Spend_LKR'].sum()/1000):.1f} L/1000 LKR")

    col1, col2 = st.columns(2)

    with col1:
        dist_agg = western.groupby("Distributor_ID").agg(
            spend=("Trade_Spend_LKR","sum"),
            gain=("expected_volume_gain_L","sum")
        ).reset_index()
        fig = px.pie(dist_agg, values="spend", names="Distributor_ID",
                     title="Budget Split by Distributor",
                     color_discrete_sequence=px.colors.sequential.Plasma_r, hole=0.4)
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="white")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = px.scatter(
            western[western["Trade_Spend_LKR"]>0],
            x="Trade_Spend_LKR", y="expected_volume_gain_L",
            color="Distributor_ID", size="Maximum_Monthly_Liters",
            hover_name="Outlet_ID",
            title="Spend vs Expected Volume Gain",
            labels={"Trade_Spend_LKR":"Budget (LKR)","expected_volume_gain_L":"Volume Gain (L)"},
            height=400,
        )
        fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font_color="white")
        st.plotly_chart(fig2, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    df          = load_data()
    importances = load_importance()

    fdf, prov, dist = sidebar_filters(df)

    # Navigation
    st.sidebar.markdown("---")
    pages = ["📊 Overview", "📋 All Outlets", "🔬 Outlet Deep Dive", "💰 Budget Dashboard"]
    page  = st.sidebar.radio("Navigate", pages)

    if page == "📊 Overview":
        page_overview(df, fdf)
    elif page == "📋 All Outlets":
        page_outlets(fdf)
    elif page == "🔬 Outlet Deep Dive":
        page_outlet_detail(df, fdf, importances)
    elif page == "💰 Budget Dashboard":
        page_budget(df)


if __name__ == "__main__":
    main()
