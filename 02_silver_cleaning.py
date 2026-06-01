"""
STEP 2 — SILVER LAYER: Data Cleaning & Quality Checks
======================================================
Purpose : Clean dirty data, fix anomalies, standardize formats.
          Route bad rows to Rejected Records Store with reasons.
Run     : python 02_silver_cleaning.py
"""

import os, logging
import pandas as pd
import numpy as np
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s [SILVER] %(message)s")
log = logging.getLogger(__name__)

BRONZE_DIR  = "bronze"
SILVER_DIR  = "silver"
REJECTED_DIR = "silver/rejected"
os.makedirs(SILVER_DIR, exist_ok=True)
os.makedirs(REJECTED_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# REUSABLE QUALITY CHECK FUNCTIONS
# Each returns (clean_df, rejected_df) — parameterized and reusable
# ─────────────────────────────────────────────────────────────────────────────

def check_nulls(df: pd.DataFrame, required_cols: list, reason: str = "null_required_field") -> tuple:
    """Route rows missing any required column to rejected."""
    mask = df[required_cols].notna().all(axis=1)
    rejected = df[~mask].copy()
    rejected["rejection_reason"] = reason
    return df[mask].copy(), rejected


def check_non_negative(df: pd.DataFrame, cols: list) -> tuple:
    """Route rows where numeric columns are negative."""
    mask = (df[cols] >= 0).all(axis=1)
    rejected = df[~mask].copy()
    rejected["rejection_reason"] = f"negative_value_in: {cols}"
    return df[mask].copy(), rejected


def check_date_range(df: pd.DataFrame, date_col: str, min_date: str, max_date: str) -> tuple:
    """Route rows with dates outside expected range."""
    dates = pd.to_datetime(df[date_col], errors="coerce")
    mask  = dates.between(min_date, max_date)
    rejected = df[~mask].copy()
    rejected["rejection_reason"] = f"date_out_of_range: {min_date} to {max_date}"
    return df[mask].copy(), rejected


def check_lat_lon(df: pd.DataFrame, lat_col="Latitude", lon_col="Longitude") -> tuple:
    """Validate Sri Lanka bounding box: lat 5.9–9.9, lon 79.6–81.9"""
    mask = (
        df[lat_col].between(5.9, 9.9) &
        df[lon_col].between(79.6, 81.9)
    )
    rejected = df[~mask].copy()
    rejected["rejection_reason"] = "coordinates_outside_sri_lanka"
    return df[mask].copy(), rejected


def remove_duplicates(df: pd.DataFrame, subset: list) -> pd.DataFrame:
    """Keep first occurrence, drop full duplicates."""
    before = len(df)
    df = df.drop_duplicates(subset=subset, keep="first")
    log.info(f"  Removed {before - len(df)} duplicate rows on {subset}")
    return df


def cap_outliers_iqr(df: pd.DataFrame, col: str, factor: float = 3.0) -> pd.DataFrame:
    """Cap extreme outliers using IQR method (not remove — soft cap)."""
    Q1  = df[col].quantile(0.25)
    Q3  = df[col].quantile(0.75)
    IQR = Q3 - Q1
    upper = Q3 + factor * IQR
    lower = max(0, Q1 - factor * IQR)
    capped = df[col].clip(lower=lower, upper=upper)
    n_capped = (df[col] != capped).sum()
    df = df.copy()
    df[col] = capped
    if n_capped > 0:
        log.info(f"  IQR cap on '{col}': {n_capped} rows adjusted")
    return df


def save_rejected(df: pd.DataFrame, name: str, stage: str = "silver"):
    if len(df) == 0:
        return
    df["_rejection_ts"] = datetime.utcnow().isoformat()
    path = os.path.join(REJECTED_DIR, f"{name}_rejected.csv")
    # Append if file exists (idempotent accumulation)
    if os.path.exists(path):
        existing = pd.read_csv(path)
        df = pd.concat([existing, df], ignore_index=True)
    df.to_csv(path, index=False)
    log.warning(f"  → {len(df)} rows in rejected store: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# INDIVIDUAL TABLE CLEANERS
# ─────────────────────────────────────────────────────────────────────────────

def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    log.info("Cleaning transactions...")
    rejected_all = []

    # 1. Required fields
    df, rej = check_nulls(df, ["Outlet_ID", "Date", "Quantity_Liters", "Distributor_ID"])
    rejected_all.append(rej)

    # 2. Parse date
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df, rej = check_nulls(df, ["Date"], "invalid_date_format")
    rejected_all.append(rej)

    # 3. Date range: 2020–2025 (historical window)
    df, rej = check_date_range(df, "Date", "2020-01-01", "2025-12-31")
    rejected_all.append(rej)

    # 4. Non-negative quantity
    df, rej = check_non_negative(df, ["Quantity_Liters"])
    rejected_all.append(rej)

    # 5. Soft-cap extreme outliers
    df = cap_outliers_iqr(df, "Quantity_Liters", factor=3.0)

    # 6. Remove duplicates
    df = remove_duplicates(df, ["Outlet_ID", "Date", "Distributor_ID"])

    # 7. Feature: month, year, day_of_week
    df["Year"]        = df["Date"].dt.year
    df["Month"]       = df["Date"].dt.month
    df["Day_of_Week"] = df["Date"].dt.dayofweek

    save_rejected(pd.concat(rejected_all, ignore_index=True), "transactions")
    log.info(f"  Clean transactions: {len(df):,} rows")
    return df


def clean_outlets(df: pd.DataFrame) -> pd.DataFrame:
    log.info("Cleaning outlet master...")
    rejected_all = []

    # 1. Required fields
    df, rej = check_nulls(df, ["Outlet_ID", "Province", "Distributor_ID", "Latitude", "Longitude"])
    rejected_all.append(rej)

    # 2. Standardize Province names
    province_map = {
        "western": "Western", "central": "Central",
        "north western": "North-Western", "north-western": "North-Western",
        "southern": "Southern"
    }
    df["Province"] = df["Province"].str.strip().str.lower().map(
        lambda x: province_map.get(x, x.title() if isinstance(x, str) else x)
    )

    # 3. Valid provinces only
    valid_provinces = {"Western", "Central", "North-Western", "Southern"}
    mask = df["Province"].isin(valid_provinces)
    rej  = df[~mask].copy(); rej["rejection_reason"] = "invalid_province"
    rejected_all.append(rej)
    df   = df[mask].copy()

    # 4. Valid distributor IDs
    valid_dists = {
        "DIST_W_01","DIST_W_02","DIST_W_03",
        "DIST_C_01","DIST_C_02","DIST_C_03",
        "DIST_NW_01","DIST_NW_02",
        "DIST_S_01","DIST_S_02"
    }
    mask = df["Distributor_ID"].isin(valid_dists)
    rej  = df[~mask].copy(); rej["rejection_reason"] = "invalid_distributor_id"
    rejected_all.append(rej)
    df   = df[mask].copy()

    # 5. Coordinates within Sri Lanka
    df["Latitude"]  = pd.to_numeric(df["Latitude"],  errors="coerce")
    df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")
    df, rej = check_lat_lon(df)
    rejected_all.append(rej)

    # 6. Deduplicate
    df = remove_duplicates(df, ["Outlet_ID"])

    save_rejected(pd.concat(rejected_all, ignore_index=True), "outlets")
    log.info(f"  Clean outlets: {len(df):,} rows")
    return df


def clean_seasonality(df: pd.DataFrame) -> pd.DataFrame:
    log.info("Cleaning seasonality...")
    df, rej = check_nulls(df, ["Distributor_ID", "Month", "Seasonality_Index"])
    save_rejected(rej, "seasonality")

    df["Month"] = pd.to_numeric(df["Month"], errors="coerce").astype("Int64")
    mask = df["Month"].between(1, 12)
    save_rejected(df[~mask].assign(rejection_reason="invalid_month"), "seasonality")
    df = df[mask].copy()

    df["Seasonality_Index"] = pd.to_numeric(df["Seasonality_Index"], errors="coerce")
    df = df.dropna(subset=["Seasonality_Index"])
    log.info(f"  Clean seasonality: {len(df):,} rows")
    return df


def clean_holidays(df: pd.DataFrame) -> pd.DataFrame:
    log.info("Cleaning holidays...")
    df, rej = check_nulls(df, ["Date", "Holiday_Name"])
    save_rejected(rej, "holidays")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    log.info(f"  Clean holidays: {len(df):,} rows")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def run():
    # Load from Bronze
    def load(fname):
        path = os.path.join(BRONZE_DIR, fname)
        return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()

    tx       = clean_transactions(load("transactions_history_final.csv"))
    outlets  = clean_outlets(load("outlet_master.csv"))
    seasonal = clean_seasonality(load("distributor_seasonality_details.csv"))
    holidays = clean_holidays(load("holiday_list.csv"))

    # Save to Silver
    tx.to_csv(      os.path.join(SILVER_DIR, "transactions.csv"),  index=False)
    outlets.to_csv( os.path.join(SILVER_DIR, "outlets.csv"),       index=False)
    seasonal.to_csv(os.path.join(SILVER_DIR, "seasonality.csv"),   index=False)
    holidays.to_csv(os.path.join(SILVER_DIR, "holidays.csv"),      index=False)

    log.info("Silver layer complete.")
    return {"transactions": tx, "outlets": outlets, "seasonality": seasonal, "holidays": holidays}


if __name__ == "__main__":
    run()
