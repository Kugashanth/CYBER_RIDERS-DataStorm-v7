"""
STEP 4 — FEATURE ENGINEERING: Transaction Aggregation & Historical Features
============================================================================
Purpose : Turn raw transaction rows into outlet-level ML features.
          Handle left-censored demand (outlets constrained by supply/credit).

Key idea: Historical sales = what they DID sell, not what they COULD sell.
          We engineer features that capture this "ceiling effect".

Run     : python 04_feature_engineering.py
"""

import os, logging
import pandas as pd
import numpy as np
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s [FEATURES] %(message)s")
log = logging.getLogger(__name__)

SILVER_DIR = "silver"
GOLD_DIR   = "gold"
os.makedirs(GOLD_DIR, exist_ok=True)


def build_transaction_features(tx: pd.DataFrame, holidays: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate transaction history into outlet-level features.
    """
    log.info("Building transaction features...")

    # ── Holiday flag ──────────────────────────────────────────────────────────
    holiday_dates = set(pd.to_datetime(holidays["Date"]).dt.date)
    tx["is_holiday"] = pd.to_datetime(tx["Date"]).dt.date.isin(holiday_dates).astype(int)

    # ── Monthly aggregation ───────────────────────────────────────────────────
    tx["YearMonth"] = pd.to_datetime(tx["Date"]).dt.to_period("M")
    monthly = (tx.groupby(["Outlet_ID", "YearMonth"])["Quantity_Liters"]
                 .sum()
                 .reset_index()
                 .rename(columns={"Quantity_Liters": "Monthly_Liters"}))

    # ── Outlet-level historical features ──────────────────────────────────────
    grp = monthly.groupby("Outlet_ID")["Monthly_Liters"]

    features = pd.DataFrame({
        "Outlet_ID"             : grp.mean().index,
        "hist_mean_monthly"     : grp.mean().values,
        "hist_median_monthly"   : grp.median().values,
        "hist_max_monthly"      : grp.max().values,
        "hist_std_monthly"      : grp.std().fillna(0).values,
        "hist_cv"               : (grp.std() / grp.mean().replace(0, np.nan)).fillna(0).values,
        "hist_p90_monthly"      : grp.quantile(0.90).values,
        "hist_p75_monthly"      : grp.quantile(0.75).values,
        "hist_months_active"    : grp.count().values,
    })

    # ── Censoring detection features ──────────────────────────────────────────
    # If an outlet's std is very low relative to its mean, it may be supply-constrained
    # (hitting a ceiling consistently = artificial cap)
    features["censoring_indicator"] = (
        (features["hist_cv"] < 0.15) &          # very low variance
        (features["hist_months_active"] >= 6)   # enough history to judge
    ).astype(int)

    # ── Trend: slope of monthly sales over time ────────────────────────────────
    def compute_trend(outlet_id):
        sub = monthly[monthly["Outlet_ID"] == outlet_id].sort_values("YearMonth")
        if len(sub) < 3:
            return 0.0
        x = np.arange(len(sub))
        y = sub["Monthly_Liters"].values
        slope, *_ = np.polyfit(x, y, 1)
        return slope

    # (Vectorised version using groupby apply for speed)
    def trend_func(grp_data):
        if len(grp_data) < 3:
            return 0.0
        x = np.arange(len(grp_data))
        y = grp_data["Monthly_Liters"].values
        return float(np.polyfit(x, y, 1)[0])

    trends = (monthly.sort_values("YearMonth")
                     .groupby("Outlet_ID")
                     .apply(trend_func)
                     .reset_index()
                     .rename(columns={0: "sales_trend_slope"}))
    features = features.merge(trends, on="Outlet_ID", how="left")
    features["sales_trend_slope"] = features["sales_trend_slope"].fillna(0)

    # ── Holiday uplift ─────────────────────────────────────────────────────────
    holiday_sales = (tx[tx["is_holiday"] == 1]
                       .groupby("Outlet_ID")["Quantity_Liters"]
                       .mean()
                       .rename("avg_holiday_sales"))
    normal_sales  = (tx[tx["is_holiday"] == 0]
                       .groupby("Outlet_ID")["Quantity_Liters"]
                       .mean()
                       .rename("avg_normal_sales"))
    uplift = pd.concat([holiday_sales, normal_sales], axis=1).reset_index()
    uplift["holiday_uplift_ratio"] = (
        uplift["avg_holiday_sales"] / uplift["avg_normal_sales"].replace(0, np.nan)
    ).fillna(1.0)
    features = features.merge(uplift[["Outlet_ID","holiday_uplift_ratio"]],
                               on="Outlet_ID", how="left")
    features["holiday_uplift_ratio"] = features["holiday_uplift_ratio"].fillna(1.0)

    # ── January seasonality ───────────────────────────────────────────────────
    jan_sales = (tx[tx["Month"] == 1]
                   .groupby("Outlet_ID")["Quantity_Liters"]
                   .mean()
                   .rename("avg_jan_sales"))
    features = features.merge(jan_sales.reset_index(), on="Outlet_ID", how="left")
    features["avg_jan_sales"] = features["avg_jan_sales"].fillna(
        features["hist_mean_monthly"]
    )

    # ── Jan ratio (how much better Jan is vs average) ─────────────────────────
    features["jan_ratio"] = (
        features["avg_jan_sales"] / features["hist_mean_monthly"].replace(0, np.nan)
    ).fillna(1.0).clip(0.5, 3.0)

    log.info(f"Transaction features built for {len(features):,} outlets")
    return features


def merge_seasonality(features: pd.DataFrame,
                       outlets: pd.DataFrame,
                       seasonality: pd.DataFrame) -> pd.DataFrame:
    """Attach the distributor-level January seasonality index."""
    log.info("Merging seasonality...")
    jan_season = seasonality[seasonality["Month"] == 1][
        ["Distributor_ID", "Seasonality_Index"]
    ].rename(columns={"Seasonality_Index": "dist_jan_seasonality"})

    df = features.merge(outlets[["Outlet_ID","Outlet_Type","Province","Distributor_ID"]],
                        on="Outlet_ID", how="left")
    df = df.merge(jan_season, on="Distributor_ID", how="left")
    df["dist_jan_seasonality"] = df["dist_jan_seasonality"].fillna(1.0)
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode Outlet_Type and Province — keep raw columns too."""
    log.info("Encoding categoricals...")
    df["Province_raw"] = df["Province"].copy()
    df["Outlet_Type_raw"] = df["Outlet_Type"].copy() if "Outlet_Type" in df.columns else "Unknown"
    df = pd.get_dummies(df, columns=["Outlet_Type", "Province"], drop_first=False)
    df = df.rename(columns={"Province_raw": "Province", "Outlet_Type_raw": "Outlet_Type"})
    return df


def run():
    tx        = pd.read_csv(os.path.join(SILVER_DIR, "transactions.csv"),
                            parse_dates=["Date"])
    tx["Month"] = tx["Date"].dt.month
    outlets   = pd.read_csv(os.path.join(SILVER_DIR, "outlets.csv"))
    seasonal  = pd.read_csv(os.path.join(SILVER_DIR, "seasonality.csv"))
    holidays  = pd.read_csv(os.path.join(SILVER_DIR, "holidays.csv"))
    gold_out  = pd.read_csv(os.path.join(GOLD_DIR,   "outlets_gold.csv"))

    # Build transaction features
    tx_feats = build_transaction_features(tx, holidays)

    # Merge seasonality and distributor info
    tx_feats = merge_seasonality(tx_feats, outlets, seasonal)

    # Merge spatial (gold) features
    spatial_cols = [c for c in gold_out.columns
                    if c.startswith(("poi_","total_","competitor","market_","spatial"))]
    merged = tx_feats.merge(
        gold_out[["Outlet_ID"] + spatial_cols],
        on="Outlet_ID", how="left"
    )

    # Encode categoricals
    merged = encode_categoricals(merged)

    # Fill remaining NAs
    num_cols = merged.select_dtypes(include=[np.number]).columns
    merged[num_cols] = merged[num_cols].fillna(0)

    out_path = os.path.join(GOLD_DIR, "ml_features.csv")
    merged.to_csv(out_path, index=False)
    log.info(f"ML features saved: {len(merged):,} outlets × {len(merged.columns)} features → {out_path}")
    return merged


if __name__ == "__main__":
    run()
