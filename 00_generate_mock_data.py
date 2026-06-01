"""
STEP 0 — MOCK DATA GENERATOR
=============================
Generates realistic raw CSV files that mimic the actual competition data.
Run this FIRST if you don't have the real data files.

Creates:
  data/raw/transactions_history_final.csv
  data/raw/outlet_master.csv
  data/raw/distributor_seasonality_details.csv
  data/raw/holiday_list.csv

Run: python 00_generate_mock_data.py
"""

import os
import pandas as pd
import numpy as np
from datetime import date, timedelta

np.random.seed(42)
os.makedirs("data/raw", exist_ok=True)

# ── CONFIG ─────────────────────────────────────────────────────────────────────
N_OUTLETS     = 500          # use 500 for speed; real data has 20,000
N_DISTRIBUTORS = 10
START_DATE    = date(2022, 1, 1)
END_DATE      = date(2025, 12, 31)

DISTRIBUTORS = {
    "DIST_W_01":  "Western",
    "DIST_W_02":  "Western",
    "DIST_W_03":  "Western",
    "DIST_C_01":  "Central",
    "DIST_C_02":  "Central",
    "DIST_C_03":  "Central",
    "DIST_NW_01": "North-Western",
    "DIST_NW_02": "North-Western",
    "DIST_S_01":  "Southern",
    "DIST_S_02":  "Southern",
}

# Province bounding boxes [lat_min, lat_max, lon_min, lon_max]
PROVINCE_BBOX = {
    "Western"       : [6.80, 7.10, 79.85, 80.10],
    "Central"       : [6.90, 7.35, 80.55, 80.90],
    "North-Western" : [7.60, 8.00, 79.90, 80.30],
    "Southern"      : [5.90, 6.40, 80.20, 81.00],
}

OUTLET_TYPES = ["Grocery", "Kade", "Eatery", "Pharmacy", "Supermarket", "Convenience"]

print("Generating outlet_master.csv ...")

outlets = []
dist_list = list(DISTRIBUTORS.keys())

for i in range(N_OUTLETS):
    dist_id  = dist_list[i % len(dist_list)]
    province = DISTRIBUTORS[dist_id]
    bbox     = PROVINCE_BBOX[province]

    lat = np.random.uniform(bbox[0], bbox[1])
    lon = np.random.uniform(bbox[2], bbox[3])

    # Inject ~5% dirty rows (missing coords, wrong province spelling)
    if np.random.rand() < 0.03:
        lat = np.nan
    if np.random.rand() < 0.02:
        province = province.lower()   # dirty: wrong case

    outlets.append({
        "Outlet_ID"     : f"OUT_{i+1:05d}",
        "Outlet_Type"   : np.random.choice(OUTLET_TYPES, p=[0.30,0.30,0.15,0.08,0.10,0.07]),
        "Province"      : province,
        "Distributor_ID": dist_id,
        "Latitude"      : round(lat, 6) if not (isinstance(lat, float) and np.isnan(lat)) else np.nan,
        "Longitude"     : round(lon, 6),
        "Cooler_Capacity_L": np.random.choice([0, 50, 100, 150, 200], p=[0.20,0.25,0.30,0.15,0.10]),
    })

outlets_df = pd.DataFrame(outlets)
outlets_df.to_csv("data/raw/outlet_master.csv", index=False)
print(f"  → {len(outlets_df)} outlets saved")


print("Generating transactions_history_final.csv ...")

# Per-outlet baseline: some outlets are supply-constrained (will hit ceiling)
outlet_baseline = {}
for _, row in outlets_df.iterrows():
    oid = row["Outlet_ID"]
    # Western province gets higher baseline
<<<<<<< HEAD
    prov_mult = 1.5 if str(row["Province"]).strip().lower() == "western" else 1.0
=======
    prov_mult = 1.5 if row["Province"] == "Western" else 1.0
>>>>>>> a93e7b3865e12cdd31fc18e65587875f20aedf50
    # Supermarkets sell more
    type_mult = {"Supermarket":2.0,"Grocery":1.2,"Convenience":1.1,"Eatery":0.9,"Kade":0.8,"Pharmacy":0.6}.get(row["Outlet_Type"],1.0)
    base_daily = np.random.lognormal(mean=2.5, sigma=0.6) * prov_mult * type_mult
    # 20% of outlets are supply-constrained — they have an artificial ceiling
    is_constrained = np.random.rand() < 0.20
    ceiling = base_daily * np.random.uniform(1.1, 1.4) if is_constrained else base_daily * 10
    outlet_baseline[oid] = {"base": base_daily, "ceiling": ceiling, "constrained": is_constrained}

transactions = []
date_range = pd.date_range(START_DATE, END_DATE, freq="D")

# Sample 1 transaction per week per outlet (not every day — realistic)
for _, out_row in outlets_df.iterrows():
    oid  = out_row["Outlet_ID"]
    dist = out_row["Distributor_ID"]
    info = outlet_baseline[oid]

    for tx_date in date_range:
        # ~3 transactions per week on average
        if np.random.rand() > 0.43:
            continue

        month = tx_date.month
        # Seasonal multiplier: Jan/Dec/April(New Year) are high
        seasonal = {1:1.3, 2:0.9, 3:1.0, 4:1.4, 5:0.95, 6:0.9,
                    7:0.85, 8:0.9, 9:1.0, 10:1.1, 11:1.2, 12:1.3}.get(month, 1.0)

        qty = info["base"] * seasonal * np.random.lognormal(0, 0.3)
        qty = min(qty, info["ceiling"])   # apply ceiling (censoring)
        qty = max(qty, 0)

        # Inject ~3% anomalies (zero sales, spikes, negative)
        if np.random.rand() < 0.015:
            qty = 0.0
        elif np.random.rand() < 0.01:
            qty = qty * np.random.uniform(5, 15)   # spike anomaly

        transactions.append({
            "Outlet_ID"      : oid,
            "Distributor_ID" : dist,
            "Date"           : tx_date.strftime("%Y-%m-%d"),
            "Quantity_Liters": round(qty, 2),
        })

tx_df = pd.DataFrame(transactions)
tx_df.to_csv("data/raw/transactions_history_final.csv", index=False)
print(f"  → {len(tx_df):,} transactions saved")


print("Generating distributor_seasonality_details.csv ...")

seasonality_rows = []
for dist_id in DISTRIBUTORS:
    for month in range(1, 13):
        base_idx = {1:1.25, 2:0.88, 3:0.95, 4:1.35, 5:0.92, 6:0.88,
                    7:0.82, 8:0.88, 9:0.97, 10:1.08, 11:1.18, 12:1.28}[month]
        # Add distributor-specific variation
        idx = base_idx * np.random.uniform(0.92, 1.08)
        seasonality_rows.append({
            "Distributor_ID"   : dist_id,
            "Month"            : month,
            "Seasonality_Index": round(idx, 4),
        })

season_df = pd.DataFrame(seasonality_rows)
season_df.to_csv("data/raw/distributor_seasonality_details.csv", index=False)
print(f"  → {len(season_df)} rows saved")


print("Generating holiday_list.csv ...")

holidays = [
    # 2022
    ("2022-01-14","Thai Pongal"),("2022-02-04","Independence Day"),
    ("2022-04-13","Sinhala & Tamil New Year Eve"),("2022-04-14","Sinhala & Tamil New Year"),
    ("2022-05-01","May Day"),("2022-05-15","Vesak Full Moon"),
    ("2022-12-25","Christmas Day"),
    # 2023
    ("2023-01-14","Thai Pongal"),("2023-02-04","Independence Day"),
    ("2023-04-13","Sinhala & Tamil New Year Eve"),("2023-04-14","Sinhala & Tamil New Year"),
    ("2023-05-01","May Day"),("2023-06-03","Poson Full Moon"),
    ("2023-12-25","Christmas Day"),
    # 2024
    ("2024-01-14","Thai Pongal"),("2024-02-04","Independence Day"),
    ("2024-04-13","Sinhala & Tamil New Year Eve"),("2024-04-14","Sinhala & Tamil New Year"),
    ("2024-05-01","May Day"),("2024-05-23","Vesak Full Moon"),
    ("2024-12-25","Christmas Day"),
    # 2025
    ("2025-01-14","Thai Pongal"),("2025-02-04","Independence Day"),
    ("2025-04-13","Sinhala & Tamil New Year Eve"),("2025-04-14","Sinhala & Tamil New Year"),
    ("2025-05-01","May Day"),("2025-05-12","Vesak Full Moon"),
    ("2025-12-25","Christmas Day"),
]

hol_df = pd.DataFrame(holidays, columns=["Date","Holiday_Name"])
hol_df.to_csv("data/raw/holiday_list.csv", index=False)
print(f"  → {len(hol_df)} holidays saved")

print("\n✅ All mock raw data files created in data/raw/")
print("   Next step: python run_pipeline.py")
