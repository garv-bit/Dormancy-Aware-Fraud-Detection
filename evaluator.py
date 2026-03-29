import logging
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Unified evaluation for all fraud detection models.
    Computes PR-AUC, ROC-AUC, F1, confusion matrix, and optimal threshold.
    """

    def evaluate(self, model, X_test, y_test, feature_names: list = None) -> dict:
        logger.info(f"[Evaluator] Evaluating {model.name}...")
        y_proba = model.predict_proba(X_test)

        pr_auc  = average_precision_score(y_test, y_proba)
        roc_auc = roc_auc_score(y_test, y_proba)
        opt_thresh, opt_f1 = self._optimal_threshold(y_test, y_proba)

        model.set_threshold(opt_thresh)
        y_pred = model.predict(X_test)

        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        fpr       = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        results = {
            "model":             model.name,
            "pr_auc":            round(pr_auc, 4),
            "roc_auc":           round(roc_auc, 4),
            "optimal_threshold": round(opt_thresh, 3),
            "f1":                round(opt_f1, 4),
            "recall":            round(recall, 4),
            "precision":         round(precision, 4),
            "false_positive_rate": round(fpr, 4),
            "confusion_matrix":  {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        }

        if feature_names and hasattr(model, "get_feature_importance"):
            results["feature_importance"] = model.get_feature_importance(feature_names)

        self._log_results(results)
        return results

    def compare(self, results_list: list) -> dict:
        best = max(results_list, key=lambda r: r["pr_auc"])
        logger.info(f"\n{'='*60}")
        logger.info(f"BEST MODEL: {best['model'].upper()}")
        logger.info(f"  PR-AUC  : {best['pr_auc']}")
        logger.info(f"  Recall  : {best['recall']} (target: >0.90)")
        logger.info(f"  FPR     : {best['false_positive_rate']} (target: reduce by >=15%)")
        logger.info(f"{'='*60}")
        return best

    def save_results(self, all_results: list, path: str = "models/evaluation_results.joblib"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(all_results, path)
        logger.info(f"[Evaluator] Results saved to {path}")

    def _optimal_threshold(self, y_true, y_proba):
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
        f1_scores = np.where(
            (precisions + recalls) > 0,
            2 * precisions * recalls / (precisions + recalls),
            0.0,
        )
        best_idx = np.argmax(f1_scores[:-1])
        return float(thresholds[best_idx]), float(f1_scores[best_idx])

    def _log_results(self, r: dict):
        cm = r["confusion_matrix"]
        logger.info(f"\n  -- {r['model'].upper()} --")
        logger.info(f"  PR-AUC   : {r['pr_auc']}   (target: >0.85)")
        logger.info(f"  ROC-AUC  : {r['roc_auc']}")
        logger.info(f"  Threshold: {r['optimal_threshold']}")
        logger.info(f"  F1       : {r['f1']}")
        logger.info(f"  Recall   : {r['recall']}   (target: >0.90)")
        logger.info(f"  Precision: {r['precision']}")
        logger.info(f"  FPR      : {r['false_positive_rate']}")
        logger.info(f"  Confusion matrix:")
        logger.info(f"    TN={cm['tn']:>8,}  FP={cm['fp']:>7,}")
        logger.info(f"    FN={cm['fn']:>8,}  TP={cm['tp']:>7,}")