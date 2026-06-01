"""
STEP 1 — BRONZE LAYER: Raw Data Ingestion
==========================================
Purpose : Load all raw CSVs exactly as-is into the Bronze layer.
          No transformations. Just schema validation + rejected records routing.
Run     : python 01_bronze_ingestion.py
"""

import os, hashlib, json, logging
from datetime import datetime
import pandas as pd

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [BRONZE] %(message)s")
log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
RAW_DIR     = "data/raw"          # put your CSVs here
BRONZE_DIR  = "bronze"
REJECTED_DIR = "bronze/rejected"
os.makedirs(BRONZE_DIR, exist_ok=True)
os.makedirs(REJECTED_DIR, exist_ok=True)

# ── Expected schemas (column presence check) ───────────────────────────────────
SCHEMAS = {
    "transactions_history_final.csv": [
        "Outlet_ID", "Date", "Quantity_Liters", "Distributor_ID"
    ],
    "outlet_master.csv": [
        "Outlet_ID", "Outlet_Type", "Province", "Distributor_ID",
        "Latitude", "Longitude"
    ],
    "distributor_seasonality_details.csv": [
        "Distributor_ID", "Month", "Seasonality_Index"
    ],
    "holiday_list.csv": [
        "Date", "Holiday_Name"
    ],
}


def compute_checksum(path: str) -> str:
    """MD5 checksum for idempotency — skip re-ingestion if unchanged."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def load_checksum_registry() -> dict:
    path = os.path.join(BRONZE_DIR, "_checksums.json")
    if os.path.exists(path):
        return json.load(open(path))
    return {}


def save_checksum_registry(registry: dict):
    path = os.path.join(BRONZE_DIR, "_checksums.json")
    json.dump(registry, open(path, "w"), indent=2)


def ingest_file(filename: str, registry: dict) -> pd.DataFrame | None:
    src = os.path.join(RAW_DIR, filename)
    dst = os.path.join(BRONZE_DIR, filename)
    rej = os.path.join(REJECTED_DIR, filename.replace(".csv", "_rejected.csv"))

    if not os.path.exists(src):
        log.warning(f"File not found: {src} — skipping")
        return None

    # ── Idempotency check ──────────────────────────────────────────────────────
    checksum = compute_checksum(src)
    if registry.get(filename) == checksum and os.path.exists(dst):
        log.info(f"[SKIP] {filename} — unchanged (checksum match)")
        return pd.read_csv(dst)

    log.info(f"[INGEST] {filename}")
    df = pd.read_csv(src, low_memory=False)

    # ── Schema validation ──────────────────────────────────────────────────────
    expected_cols = SCHEMAS.get(filename, [])
    missing_cols  = [c for c in expected_cols if c not in df.columns]

    if missing_cols:
        log.error(f"Schema mismatch in {filename}: missing {missing_cols}")
        df.to_csv(rej, index=False)
        log.info(f"  → Entire file routed to rejected store: {rej}")
        return None

    # ── Route rows with ALL required fields null to rejected ──────────────────
    mask_valid   = df[expected_cols].notna().any(axis=1)
    df_valid     = df[mask_valid].copy()
    df_rejected  = df[~mask_valid].copy()

    if len(df_rejected) > 0:
        df_rejected["rejection_reason"] = "All key fields null"
        df_rejected["ingestion_ts"]     = datetime.utcnow().isoformat()
        df_rejected.to_csv(rej, index=False)
        log.warning(f"  → {len(df_rejected)} rows rejected → {rej}")

    # ── Add metadata columns ───────────────────────────────────────────────────
    df_valid["_ingestion_ts"]   = datetime.utcnow().isoformat()
    df_valid["_source_file"]    = filename
    df_valid["_checksum"]       = checksum

    df_valid.to_csv(dst, index=False)
    registry[filename] = checksum
    log.info(f"  → {len(df_valid)} valid rows saved to {dst}")
    return df_valid


def run():
    registry = load_checksum_registry()
    results  = {}

    for fname in SCHEMAS.keys():
        results[fname] = ingest_file(fname, registry)

    save_checksum_registry(registry)
    log.info("Bronze ingestion complete.")
    return results


if __name__ == "__main__":
    run()
