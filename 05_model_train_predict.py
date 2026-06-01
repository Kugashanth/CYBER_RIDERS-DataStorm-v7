"""
STEP 5 — MODELING: Tobit Regression + Ensemble for Latent Potential Prediction
===============================================================================
Purpose : Predict Maximum_Monthly_Liters (latent/uncapped demand) for Jan 2026.

Why Tobit?
  Historical sales are LEFT-CENSORED — an outlet constrained by supply/credit
  reports lower sales than its true demand. Standard OLS would underestimate.
  Tobit regression explicitly models this censoring mechanism.

We also train:
  - XGBoost (captures non-linear spatial interactions)
  - LightGBM (fast, handles categorical features)
  - Ensemble: weighted average of all three

Run     : python 05_model_train_predict.py
"""

import os, logging, json, warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [MODEL] %(message)s")
log = logging.getLogger(__name__)

GOLD_DIR    = "gold"
MODELS_DIR  = "models"
OUTPUTS_DIR = "outputs"
os.makedirs(MODELS_DIR,  exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# TOBIT REGRESSION (Custom implementation using scipy MLE)
# ─────────────────────────────────────────────────────────────────────────────

from scipy.optimize import minimize
from scipy.stats import norm

class TobitRegressor:
    """
    Type I Tobit model for left-censored data.

    Model: y* = Xβ + ε,  ε ~ N(0, σ²)
           y  = y*  if y* > L  (observed, above lower limit)
           y  = L   if y* ≤ L  (censored, at lower limit)

    Log-likelihood:
      For uncensored obs: log φ((y - Xβ)/σ) - log σ
      For censored obs  : log Φ((L - Xβ)/σ)
    """

    def __init__(self, lower_limit: float = 0.0):
        self.lower_limit = lower_limit
        self.coef_       = None
        self.sigma_      = None
        self.scaler_     = StandardScaler()

    def _neg_log_likelihood(self, params, X, y):
        beta  = params[:-1]
        sigma = max(params[-1], 1e-6)
        L     = self.lower_limit

        xb       = X @ beta
        censored = (y <= L)

        ll = 0.0
        # Uncensored observations
        if censored.sum() < len(y):
            y_unc  = y[~censored]
            xb_unc = xb[~censored]
            ll += np.sum(norm.logpdf(y_unc, xb_unc, sigma))

        # Censored observations
        if censored.sum() > 0:
            xb_cen = xb[censored]
            ll += np.sum(norm.logcdf(L, xb_cen, sigma))

        return -ll

    def fit(self, X: np.ndarray, y: np.ndarray):
        X_s = self.scaler_.fit_transform(X)
        n_features = X_s.shape[1]

        # Initial params: OLS coefficients + std(y)
        beta0  = np.linalg.lstsq(X_s, y, rcond=None)[0]
        sigma0 = max(np.std(y), 1.0)
        params0 = np.append(beta0, sigma0)

        result = minimize(
            self._neg_log_likelihood,
            params0,
            args=(X_s, y),
            method="L-BFGS-B",
            bounds=[(None, None)] * n_features + [(0.01, None)],
            options={"maxiter": 500, "ftol": 1e-8}
        )

        self.coef_  = result.x[:-1]
        self.sigma_ = result.x[-1]
        log.info(f"  Tobit fitted | σ={self.sigma_:.3f} | success={result.success}")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """E[y | X] — expected value accounting for censoring."""
        X_s  = self.scaler_.transform(X)
        xb   = X_s @ self.coef_
        L    = self.lower_limit
        σ    = self.sigma_

        # E[y* | y* > L] × P(y* > L) + L × P(y* ≤ L)
        # Simplified: unconditional E[y*] = xb (latent mean)
        # For prediction of UNCAPPED potential we return xb (+ positive correction)
        alpha = (L - xb) / σ
        # Mills ratio correction for censored portion
        lambda_mills = norm.pdf(alpha) / (1 - norm.cdf(alpha) + 1e-10)
        y_hat = xb + σ * lambda_mills
        return np.maximum(y_hat, 0)

    def predict_latent(self, X: np.ndarray) -> np.ndarray:
        """Raw latent E[y*] = Xβ — the true uncapped potential."""
        X_s = self.scaler_.transform(X)
        return np.maximum(X_s @ self.coef_, 0)

    def save(self, path: str):
        joblib.dump({"coef": self.coef_, "sigma": self.sigma_,
                     "scaler": self.scaler_, "lower_limit": self.lower_limit}, path)

    @classmethod
    def load(cls, path: str):
        d   = joblib.load(path)
        obj = cls(lower_limit=d["lower_limit"])
        obj.coef_  = d["coef"]
        obj.sigma_ = d["sigma"]
        obj.scaler_= d["scaler"]
        return obj


# ─────────────────────────────────────────────────────────────────────────────
# XGBOOST + LIGHTGBM WRAPPERS
# ─────────────────────────────────────────────────────────────────────────────

def train_xgboost(X_train, y_train, X_val, y_val):
    try:
        import xgboost as xgb
        model = xgb.XGBRegressor(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            eval_metric="mae", early_stopping_rounds=30, verbosity=0
        )
        model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)], verbose=False)
        return model
    except ImportError:
        log.warning("xgboost not installed — skipping XGB")
        return None


def train_lightgbm(X_train, y_train, X_val, y_val):
    try:
        import lightgbm as lgb
        model = lgb.LGBMRegressor(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            verbose=-1
        )
        model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(30, verbose=False)])
        return model
    except ImportError:
        log.warning("lightgbm not installed — skipping LGB")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE COLUMNS
# ─────────────────────────────────────────────────────────────────────────────

BASE_FEATURE_COLS = [
    "hist_mean_monthly", "hist_median_monthly", "hist_max_monthly",
    "hist_std_monthly", "hist_cv", "hist_p90_monthly", "hist_p75_monthly",
    "hist_months_active", "sales_trend_slope", "holiday_uplift_ratio",
    "avg_jan_sales", "jan_ratio", "dist_jan_seasonality",
    "total_gaussian_poi_score", "total_gravity_poi_score", "total_exp_poi_score",
    "total_poi_count", "poi_count_bus_stop", "poi_count_school",
    "poi_count_market", "poi_count_hospital", "poi_count_restaurant",
    "competitor_count", "competitor_density_score", "market_isolation_score",
    "censoring_indicator",
]


def get_feature_cols(df: pd.DataFrame) -> list:
    """Return feature cols present in df (handles missing cols gracefully)."""
    available = [c for c in BASE_FEATURE_COLS if c in df.columns]
    # Also include one-hot encoded cols
    ohe_cols  = [c for c in df.columns if c.startswith(("Outlet_Type_","Province_"))]
    return available + ohe_cols


# ─────────────────────────────────────────────────────────────────────────────
# TARGET: UNCAPPED POTENTIAL
# ─────────────────────────────────────────────────────────────────────────────

def compute_target(df: pd.DataFrame) -> pd.Series:
    """
    Proxy for latent potential:
      - For censored outlets (low CV): use 90th percentile × Jan ratio × seasonality
      - For normal outlets : use max_monthly × Jan ratio × seasonality
    """
    base = np.where(
        df["censoring_indicator"] == 1,
        df["hist_p90_monthly"] * 1.35,   # uplift censored outlets by 35%
        df["hist_max_monthly"]
    )
    target = (base
              * df["jan_ratio"].clip(0.5, 3.0)
              * df["dist_jan_seasonality"].clip(0.5, 2.0))
    return pd.Series(np.maximum(target, df["hist_mean_monthly"]), name="target")


# ─────────────────────────────────────────────────────────────────────────────
# CROSS-VALIDATION TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def train_and_predict(df: pd.DataFrame) -> pd.DataFrame:
    feat_cols = get_feature_cols(df)
    y         = compute_target(df).values
    X         = df[feat_cols].fillna(0).values

    log.info(f"Training on {len(X):,} outlets × {len(feat_cols)} features")

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_tobit = np.zeros(len(X))
    oof_xgb   = np.zeros(len(X))
    oof_lgb   = np.zeros(len(X))

    tobit_models, xgb_models, lgb_models = [], [], []

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X), 1):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        log.info(f"  Fold {fold}/5 ...")

        # Tobit
        tobit = TobitRegressor(lower_limit=0.0)
        tobit.fit(X_tr, y_tr)
        oof_tobit[val_idx] = tobit.predict_latent(X_val)
        tobit_models.append(tobit)

        # XGBoost
        xgb_m = train_xgboost(X_tr, y_tr, X_val, y_val)
        if xgb_m:
            oof_xgb[val_idx] = xgb_m.predict(X_val)
            xgb_models.append(xgb_m)

        # LightGBM
        lgb_m = train_lightgbm(X_tr, y_tr, X_val, y_val)
        if lgb_m:
            oof_lgb[val_idx] = lgb_m.predict(X_val)
            lgb_models.append(lgb_m)

    # ── OOF metrics ───────────────────────────────────────────────────────────
    log.info(f"OOF Tobit  MAE={mean_absolute_error(y, oof_tobit):.2f}  R²={r2_score(y, oof_tobit):.3f}")
    if xgb_models:
        log.info(f"OOF XGB    MAE={mean_absolute_error(y, oof_xgb):.2f}  R²={r2_score(y, oof_xgb):.3f}")
    if lgb_models:
        log.info(f"OOF LGB    MAE={mean_absolute_error(y, oof_lgb):.2f}  R²={r2_score(y, oof_lgb):.3f}")

    # ── Retrain on full data, predict final ───────────────────────────────────
    tobit_final = TobitRegressor(lower_limit=0.0).fit(X, y)
    pred_tobit  = tobit_final.predict_latent(X)

    if xgb_models:
        xgb_final = train_xgboost(X, y, X, y)
        pred_xgb  = xgb_final.predict(X) if xgb_final else pred_tobit
    else:
        pred_xgb = pred_tobit
        xgb_final = None

    if lgb_models:
        lgb_final = train_lightgbm(X, y, X, y)
        pred_lgb  = lgb_final.predict(X) if lgb_final else pred_tobit
    else:
        pred_lgb = pred_tobit
        lgb_final = None

    # Ensemble: 40% Tobit + 30% XGB + 30% LGB
    w_tobit = 0.4
    w_xgb   = 0.3 if xgb_final else 0.0
    w_lgb   = 0.3 if lgb_final else 0.0
    total_w = w_tobit + w_xgb + w_lgb

    pred_ensemble = (w_tobit * pred_tobit + w_xgb * pred_xgb + w_lgb * pred_lgb) / total_w
    pred_ensemble = np.maximum(pred_ensemble, df["hist_mean_monthly"].values * 0.5)

    # Save models
    tobit_final.save(os.path.join(MODELS_DIR, "tobit_final.pkl"))
    if xgb_final:
        joblib.dump(xgb_final, os.path.join(MODELS_DIR, "xgb_final.pkl"))
    if lgb_final:
        joblib.dump(lgb_final, os.path.join(MODELS_DIR, "lgb_final.pkl"))

    # ── Feature importance (from XGB or LGB) ──────────────────────────────────
    if xgb_final:
        importances = dict(zip(feat_cols, xgb_final.feature_importances_))
    elif lgb_final:
        importances = dict(zip(feat_cols, lgb_final.feature_importances_))
    else:
        # Tobit coefficient magnitudes as proxy
        tobit_coefs = np.abs(tobit_final.coef_)
        importances = dict(zip(feat_cols, tobit_coefs / tobit_coefs.sum()))

    json.dump({k: float(v) for k, v in importances.items()},
              open(os.path.join(MODELS_DIR, "feature_importance.json"), "w"), indent=2)

    # ── Output dataframe ──────────────────────────────────────────────────────
    result = df[["Outlet_ID"]].copy()
    result["Maximum_Monthly_Liters"]   = np.round(pred_ensemble, 2)
    result["pred_tobit"]               = np.round(pred_tobit, 2)
    result["pred_xgb"]                 = np.round(pred_xgb, 2)
    result["pred_lgb"]                 = np.round(pred_lgb, 2)
    result["hist_mean_monthly"]        = df["hist_mean_monthly"].values
    result["censoring_indicator"]      = df["censoring_indicator"].values
    result["Province"]                 = df.get("Province", pd.Series(["Unknown"]*len(df))).values
    result["Distributor_ID"]           = df.get("Distributor_ID", pd.Series(["Unknown"]*len(df))).values
    result["incremental_potential"]    = np.maximum(
        result["Maximum_Monthly_Liters"] - result["hist_mean_monthly"], 0
    )

    log.info(f"Predictions complete. Avg latent potential: {pred_ensemble.mean():.1f} L/month")
    return result, feat_cols, importances


def run():
    df = pd.read_csv(os.path.join(GOLD_DIR, "ml_features.csv"))

    result, feat_cols, importances = train_and_predict(df)

    # ── cyberRiders_predictions.csv ────────────────────────────────────────────
    submission = result[["Outlet_ID", "Maximum_Monthly_Liters"]]
    submission.to_csv(os.path.join(OUTPUTS_DIR, "cyberRiders_predictions.csv"), index=False)
    log.info(f"Submission saved → outputs/cyberRiders_predictions.csv")

    # ── Full result with all details ──────────────────────────────────────────
    result.to_csv(os.path.join(GOLD_DIR, "predictions_full.csv"), index=False)

    top5 = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:5]
    log.info(f"Top 5 features: {top5}")
    return result


if __name__ == "__main__":
    run()
