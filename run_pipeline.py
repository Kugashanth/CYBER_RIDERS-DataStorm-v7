"""
MASTER PIPELINE: Run all steps in order
========================================
Run: python run_pipeline.py
"""

import logging, time
logging.basicConfig(level=logging.INFO, format="%(asctime)s [PIPELINE] %(message)s")
log = logging.getLogger(__name__)

def run():
    steps = [
        ("Step 1: Bronze Ingestion",      "01_bronze_ingestion",      "run"),
        ("Step 2: Silver Cleaning",        "02_silver_cleaning",       "run"),
        ("Step 3: Gold POI Features",      "03_gold_poi_features",     "run"),
        ("Step 4: Feature Engineering",    "04_feature_engineering",   "run"),
        ("Step 5: Model Train & Predict",  "05_model_train_predict",   "run"),
        ("Step 6: Budget Optimization",    "06_budget_optimization",   "run"),
    ]

    for name, module_name, func in steps:
        log.info(f"{'='*60}")
        log.info(f"  {name}")
        log.info(f"{'='*60}")
        start = time.time()
        try:
            mod = __import__(module_name)
            getattr(mod, func)()
            log.info(f"  ✅ Done in {time.time()-start:.1f}s")
        except Exception as e:
            log.error(f"  ❌ FAILED: {e}")
            raise

    log.info("\n✅ Pipeline complete!")
    log.info("  → outputs/cyberRiders_predictions.csv")
    log.info("  → outputs/cyberRiders_budget_allocations.csv")
    log.info("  → Run web app: streamlit run 08_app.py")

if __name__ == "__main__":
    run()
