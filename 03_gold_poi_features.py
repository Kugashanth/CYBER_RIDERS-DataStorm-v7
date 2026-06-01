"""
STEP 3 — GOLD LAYER: POI Scraping + Spatial Feature Engineering
================================================================
Purpose : Fetch Points of Interest (POI) via Overpass API (OpenStreetMap),
          then engineer distance-decay features and competitor density scores.

Key concepts:
  - Gaussian decay  : weight = exp(-d² / (2σ²))   σ = bandwidth in meters
  - Gravity model   : weight = 1 / d²
  - Exponential     : weight = exp(-λd)

Run     : python 03_gold_poi_features.py
"""

import os, time, logging, json
import pandas as pd
import numpy as np
import requests
from math import radians, cos, sin, asin, sqrt

logging.basicConfig(level=logging.INFO, format="%(asctime)s [GOLD] %(message)s")
log = logging.getLogger(__name__)

SILVER_DIR = "silver"
GOLD_DIR   = "gold"
CACHE_DIR  = "gold/poi_cache"
os.makedirs(GOLD_DIR,   exist_ok=True)
os.makedirs(CACHE_DIR,  exist_ok=True)

# ── Tunable parameters ─────────────────────────────────────────────────────────
SEARCH_RADIUS_M   = 1000      # metres around each outlet
GAUSSIAN_SIGMA_M  = 300       # Gaussian bandwidth (metres)
DECAY_LAMBDA      = 0.003     # Exponential decay λ (per metre)
COMPETITOR_RADIUS = 500       # metres for competitor count
OVERPASS_URL      = "https://overpass-api.de/api/interpreter"
REQUEST_DELAY_S   = 1.5       # be polite to the free API

# ── POI categories and their demand weights ───────────────────────────────────
POI_TAGS = {
    "bus_stop"       : {"highway": "bus_stop"},
    "school"         : {"amenity": "school"},
    "hospital"       : {"amenity": "hospital"},
    "market"         : {"amenity": "marketplace"},
    "supermarket"    : {"shop": "supermarket"},
    "fuel_station"   : {"amenity": "fuel"},
    "restaurant"     : {"amenity": "restaurant"},
    "bank"           : {"amenity": "bank"},
    "place_of_worship": {"amenity": "place_of_worship"},
    "sports_centre"  : {"leisure": "sports_centre"},
}

# Demand multipliers — how much each POI type drives footfall
POI_WEIGHTS = {
    "bus_stop": 2.0, "market": 2.5, "school": 1.5, "hospital": 1.8,
    "supermarket": 1.2, "fuel_station": 1.3, "restaurant": 1.0,
    "bank": 1.4, "place_of_worship": 1.1, "sports_centre": 0.9,
}

# ─────────────────────────────────────────────────────────────────────────────
# UTILITY: Haversine distance (metres)
# ─────────────────────────────────────────────────────────────────────────────

def haversine_m(lat1, lon1, lat2, lon2) -> float:
    """Distance in metres between two lat/lon points."""
    R = 6_371_000
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    return 2 * R * asin(sqrt(a))


# ─────────────────────────────────────────────────────────────────────────────
# DECAY FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def gaussian_decay(d_m: float, sigma: float = GAUSSIAN_SIGMA_M) -> float:
    """Influence reduces smoothly — strongest near centre, near-zero beyond σ."""
    return float(np.exp(-(d_m**2) / (2 * sigma**2)))


def exponential_decay(d_m: float, lam: float = DECAY_LAMBDA) -> float:
    """Constant % reduction per metre."""
    return float(np.exp(-lam * d_m))


def gravity_model(d_m: float, min_d: float = 10.0) -> float:
    """Influence ∝ 1/d² — classic gravity model."""
    return 1.0 / max(d_m, min_d)**2


# ─────────────────────────────────────────────────────────────────────────────
# OVERPASS API QUERY
# ─────────────────────────────────────────────────────────────────────────────

def build_overpass_query(lat: float, lon: float, radius_m: int, tag_dict: dict) -> str:
    """Build Overpass QL query for one tag around a point."""
    key, val = next(iter(tag_dict.items()))
    if val == "*":
        tag_filter = f'["{key}"]'
    else:
        tag_filter = f'["{key}"="{val}"]'
    return f"""
    [out:json][timeout:25];
    (
      node{tag_filter}(around:{radius_m},{lat},{lon});
      way{tag_filter}(around:{radius_m},{lat},{lon});
    );
    out center;
    """


def fetch_pois(lat: float, lon: float, poi_type: str, tag: dict, radius: int) -> list:
    """Fetch POIs from Overpass API with disk cache."""
    cache_key  = f"{poi_type}_{lat:.4f}_{lon:.4f}_{radius}.json"
    cache_path = os.path.join(CACHE_DIR, cache_key)

    if os.path.exists(cache_path):
        return json.load(open(cache_path))

    query = build_overpass_query(lat, lon, radius, tag)
    try:
        resp = requests.post(OVERPASS_URL, data={"data": query}, timeout=30)
        resp.raise_for_status()
        elements = resp.json().get("elements", [])
        # Normalise: extract lat/lon from nodes and way centres
        pois = []
        for el in elements:
            if el["type"] == "node":
                pois.append({"lat": el["lat"], "lon": el["lon"]})
            elif "center" in el:
                pois.append({"lat": el["center"]["lat"], "lon": el["center"]["lon"]})
        json.dump(pois, open(cache_path, "w"))
        time.sleep(REQUEST_DELAY_S)
        return pois
    except Exception as e:
        log.warning(f"Overpass error for {poi_type} @ ({lat:.4f},{lon:.4f}): {e}")
        return []


def fetch_competitors(lat: float, lon: float, radius: int) -> list:
    """Fetch other retail/kade outlets as competitors."""
    competitor_tags = [
        {"shop": "convenience"}, {"shop": "grocery"}, {"amenity": "fast_food"},
        {"shop": "general"}, {"shop": "kiosk"},
    ]
    cache_key  = f"competitors_{lat:.4f}_{lon:.4f}_{radius}.json"
    cache_path = os.path.join(CACHE_DIR, cache_key)
    if os.path.exists(cache_path):
        return json.load(open(cache_path))

    all_comps = []
    for tag in competitor_tags:
        key, val = next(iter(tag.items()))
        query = build_overpass_query(lat, lon, radius, tag)
        try:
            resp = requests.post(OVERPASS_URL, data={"data": query}, timeout=30)
            resp.raise_for_status()
            for el in resp.json().get("elements", []):
                if el["type"] == "node":
                    all_comps.append({"lat": el["lat"], "lon": el["lon"]})
                elif "center" in el:
                    all_comps.append({"lat": el["center"]["lat"], "lon": el["center"]["lon"]})
            time.sleep(REQUEST_DELAY_S)
        except:
            pass

    json.dump(all_comps, open(cache_path, "w"))
    return all_comps


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING PER OUTLET
# ─────────────────────────────────────────────────────────────────────────────

def compute_spatial_features(row: pd.Series) -> dict:
    """For one outlet, compute all spatial decay features."""
    lat, lon = row["Latitude"], row["Longitude"]
    features = {}

    # ── POI decay scores ──────────────────────────────────────────────────────
    total_gaussian_score = 0.0
    total_gravity_score  = 0.0
    total_exp_score      = 0.0
    poi_counts           = {}

    for poi_type, tag in POI_TAGS.items():
        pois = fetch_pois(lat, lon, poi_type, tag, SEARCH_RADIUS_M)
        w    = POI_WEIGHTS.get(poi_type, 1.0)

        gauss_sum = 0.0
        grav_sum  = 0.0
        exp_sum   = 0.0

        for poi in pois:
            d = haversine_m(lat, lon, poi["lat"], poi["lon"])
            if d < 1:
                d = 1.0   # avoid division by zero
            gauss_sum += w * gaussian_decay(d)
            grav_sum  += w * gravity_model(d)
            exp_sum   += w * exponential_decay(d)

        features[f"poi_gaussian_{poi_type}"] = gauss_sum
        features[f"poi_gravity_{poi_type}"]  = grav_sum
        features[f"poi_exp_{poi_type}"]      = exp_sum
        features[f"poi_count_{poi_type}"]    = len(pois)

        total_gaussian_score += gauss_sum
        total_gravity_score  += grav_sum
        total_exp_score      += exp_sum
        poi_counts[poi_type]  = len(pois)

    features["total_gaussian_poi_score"] = total_gaussian_score
    features["total_gravity_poi_score"]  = total_gravity_score
    features["total_exp_poi_score"]      = total_exp_score
    features["total_poi_count"]          = sum(poi_counts.values())

    # ── Competitor density ────────────────────────────────────────────────────
    competitors = fetch_competitors(lat, lon, COMPETITOR_RADIUS)
    comp_decay  = 0.0
    for comp in competitors:
        d = haversine_m(lat, lon, comp["lat"], comp["lon"])
        if d > 5:   # exclude self
            comp_decay += gaussian_decay(d, sigma=200)

    features["competitor_count"]         = len(competitors)
    features["competitor_density_score"] = comp_decay
    # Isolation score: high = less competition (opportunity)
    features["market_isolation_score"]   = 1.0 / (1.0 + comp_decay)

    return features


# ─────────────────────────────────────────────────────────────────────────────
# MOCK DATA GENERATOR (for demo if Overpass is unavailable)
# ─────────────────────────────────────────────────────────────────────────────

def generate_mock_spatial_features(outlets_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate realistic-looking spatial features using seeded random numbers
    correlated with lat/lon. Use this if Overpass API is unavailable.
    """
    log.info("Generating MOCK spatial features (Overpass API not called)")
    np.random.seed(42)
    n = len(outlets_df)

    # Simulate urban vs rural based on Western Province having more activity
    is_western = (outlets_df["Province"] == "Western").astype(float).values

    features = pd.DataFrame({
        "Outlet_ID": outlets_df["Outlet_ID"].values,

        # POI decay scores — Western outlets get higher baseline
        "total_gaussian_poi_score": np.random.gamma(2, 2, n) * (1 + is_western),
        "total_gravity_poi_score" : np.random.gamma(1.5, 0.001, n) * (1 + is_western),
        "total_exp_poi_score"     : np.random.gamma(2, 1.5, n) * (1 + is_western),
        "total_poi_count"         : np.random.poisson(8, n) + (is_western * 5).astype(int),

        # Individual POI counts
        "poi_count_bus_stop"      : np.random.poisson(2, n),
        "poi_count_school"        : np.random.poisson(1, n),
        "poi_count_market"        : np.random.poisson(1, n),
        "poi_count_hospital"      : np.random.poisson(0.3, n),
        "poi_count_restaurant"    : np.random.poisson(3, n),
        "poi_count_supermarket"   : np.random.poisson(0.5, n),

        # Competitor features
        "competitor_count"        : np.random.poisson(4, n) + (is_western * 3).astype(int),
        "competitor_density_score": np.random.exponential(0.5, n) * (1 + 0.5 * is_western),
        "market_isolation_score"  : np.random.beta(2, 5, n),
    })

    return features


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def run(use_mock: bool = True):
    outlets = pd.read_csv(os.path.join(SILVER_DIR, "outlets.csv"))
    log.info(f"Computing spatial features for {len(outlets):,} outlets")

    if use_mock:
        # ── MOCK mode: fast, no API calls ─────────────────────────────────────
        spatial_df = generate_mock_spatial_features(outlets)
    else:
        # ── LIVE mode: real Overpass API calls ────────────────────────────────
        # WARNING: 20,000 outlets × ~10 queries each = slow. Use batching/caching.
        all_features = []
        for i, row in outlets.iterrows():
            if i % 100 == 0:
                log.info(f"  Processing outlet {i}/{len(outlets)}...")
            feats = compute_spatial_features(row)
            feats["Outlet_ID"] = row["Outlet_ID"]
            all_features.append(feats)
        spatial_df = pd.DataFrame(all_features)

    # Merge with outlets
    gold = outlets.merge(spatial_df, on="Outlet_ID", how="left")

    gold.to_csv(os.path.join(GOLD_DIR, "outlets_gold.csv"), index=False)
    log.info(f"Gold outlets saved: {len(gold):,} rows, {len(gold.columns)} columns")
    return gold


if __name__ == "__main__":
    # Set use_mock=False to call real Overpass API
    run(use_mock=True)
