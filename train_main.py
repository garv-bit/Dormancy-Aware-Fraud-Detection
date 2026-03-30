"""
Train_main.py — Model Training Orchestrator
Dormancy-Aware Fraud Detection | COMP 385 Capstone

Loads feature-engineered parquet files, trains LR / RF / XGBoost,
evaluates all three, saves the best model for the Flask backend.
"""

import sys
import logging
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from imblearn.over_sampling import SMOTE

sys.path.insert(0, str(Path(__file__).parent))
from models.base_model import BaseModel
from models.logistic_model import LogisticModel
from models.random_forest import RandomForestModel
from models.xgboost_model import XGBoostModel
from models.evaluator import Evaluator

# ── LOGGING ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("model_training.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# ── CONFIG ────────────────────────────────────────────────────────────────────
FEATURES_DIR  = Path("features")
MODELS_DIR    = Path("models")
RANDOM_STATE  = 42
SMOTE_RATIO   = 0.2        # oversample minority to 20% of majority
SMOTE_SAMPLE  = 800_000    # subsample train for SMOTE (memory safety)


def load_data():
    logger.info("Loading feature-engineered parquet files...")
    X_train = pd.read_parquet(FEATURES_DIR / "X_train.parquet")
    X_test  = pd.read_parquet(FEATURES_DIR / "X_test.parquet")
    y_train = pd.read_parquet(FEATURES_DIR / "y_train.parquet").squeeze()
    y_test  = pd.read_parquet(FEATURES_DIR / "y_test.parquet").squeeze()

    logger.info(f"X_train: {X_train.shape} | fraud: {y_train.sum():,} ({y_train.mean()*100:.2f}%)")
    logger.info(f"X_test:  {X_test.shape}  | fraud: {y_test.sum():,} ({y_test.mean()*100:.2f}%)")
    return X_train, X_test, y_train, y_test


def apply_smote(X_train, y_train):
    """
    Apply SMOTE on a subsample to keep memory manageable on 4M rows.
    We subsample, oversample, then return the SMOTE'd set.
    """
    logger.info(f"Applying SMOTE (subsample={SMOTE_SAMPLE:,}, ratio={SMOTE_RATIO})...")

    if len(X_train) > SMOTE_SAMPLE:
        rng = np.random.default_rng(RANDOM_STATE)
        idx = rng.choice(len(X_train), size=SMOTE_SAMPLE, replace=False)
        X_sub = X_train.iloc[idx]
        y_sub = y_train.iloc[idx]
    else:
        X_sub, y_sub = X_train, y_train

    smote = SMOTE(sampling_strategy=SMOTE_RATIO, random_state=RANDOM_STATE)
    X_res, y_res = smote.fit_resample(X_sub, y_sub)

    logger.info(f"After SMOTE: {X_res.shape[0]:,} rows | "
                f"fraud: {y_res.sum():,} ({y_res.mean()*100:.2f}%)")
    return X_res, y_res


def print_banner():
    print("=" * 72)
    print("         DORMANCY-AWARE FRAUD DETECTION — MODEL TRAINING")
    print("         Phase 3A: LR Baseline → Random Forest → XGBoost")
    print("=" * 72)


def main():
    print_banner()
    MODELS_DIR.mkdir(exist_ok=True)

    X_train, X_test, y_train, y_test = load_data()
    feature_names = list(X_train.columns)

    X_smote, y_smote = apply_smote(X_train, y_train)

    models = [
        LogisticModel(output_dir=str(MODELS_DIR)),
        RandomForestModel(output_dir=str(MODELS_DIR)),
        XGBoostModel(output_dir=str(MODELS_DIR)),
    ]

    evaluator = Evaluator()
    all_results = []

    for model in models:
        print(f"\n{'─'*72}")
        print(f"  Training: {model.name.upper()}")
        print(f"{'─'*72}")

        # LR benefits most from SMOTE; RF and XGBoost handle imbalance natively
        # but we pass SMOTE data to all for a fair comparison
        use_X = X_smote if model.name == "logistic_regression" else X_train
        use_y = y_smote if model.name == "logistic_regression" else y_train

        model.train(use_X, use_y)
        results = evaluator.evaluate(model, X_test, y_test, feature_names)
        all_results.append(results)
        model.save()

    print(f"\n{'='*72}")
    print("  EVALUATION SUMMARY")
    print(f"{'='*72}")

    for r in all_results:
        print(f"\n  {r['model'].upper()}")
        print(f"    PR-AUC   : {r['pr_auc']}   (target >0.85)")
        print(f"    Recall   : {r['recall']}   (target >0.90)")
        print(f"    Precision: {r['precision']}")
        print(f"    FPR      : {r['false_positive_rate']}")
        print(f"    Threshold: {r['optimal_threshold']}")

    best = evaluator.compare(all_results)

    # Save the best model separately as best_model.joblib for Flask to load
    best_model = next(m for m in models if m.name == best["model"])
    best_path = MODELS_DIR / "best_model.joblib"
    joblib.dump(best_model, best_path)
    logger.info(f"Best model saved to {best_path}")

    # Save all evaluation results
    evaluator.save_results(all_results)

    # Save metadata Flask needs at inference time
    meta = {
        "best_model_name": best["model"],
        "threshold":       best["optimal_threshold"],
        "feature_names":   feature_names,
        "pr_auc":          best["pr_auc"],
        "recall":          best["recall"],
        "all_results":     all_results,
    }
    joblib.dump(meta, MODELS_DIR / "training_meta.joblib")
    logger.info("Training metadata saved to models/training_meta.joblib")

    print(f"\n{'='*72}")
    print(f"  DONE. Best model: {best['model'].upper()}")
    print(f"  PR-AUC: {best['pr_auc']} | Recall: {best['recall']} | Threshold: {best['optimal_threshold']}")
    print(f"  Saved to: {best_path}")
    print(f"{'='*72}\n")


if __name__ == "__main__":
    main()