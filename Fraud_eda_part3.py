"""
Fraud EDA Part 3 — Statistical Tests, Model Readiness, Recommendations,
                    run_full_analysis, save_results

All methods are attached to FraudEDA in Fraud_eda.py.
Leakage guard: config.LEAKY_COLUMNS enforced throughout.
Checkpointing: each section saves/loads from eda_checkpoints/<section>.joblib.
"""

import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from scipy.stats import chi2_contingency

from Utils import (
    safe_divide,
    get_numeric_columns,
    get_categorical_columns,
    print_section_header,
    format_bytes,
)


# ==============================================================================
# 11. STATISTICAL HYPOTHESIS TESTS
# ==============================================================================

def analyze_statistical_tests(self) -> Dict[str, Any]:
    """
    Chi-square independence tests between low-cardinality categorical columns.
    Skips leaky columns.
    """
    print_section_header("10. STATISTICAL HYPOTHESIS TESTS", logger=self.logger)

    cached = self._ckpt_load("statistical_tests")
    if cached is not None:
        self.results['statistical_tests'] = cached
        return cached

    leaky = self._leaky()
    statistical_tests = {'chi_square_tests': [], 't_tests': []}

    cat_cols_test = [
        c for c in get_categorical_columns(
            self.df, exclude=self.config.SKIP_CATEGORICAL_ANALYSIS
        )
        if c not in leaky
        and 1 < self.df[c].nunique() < self.config.MAX_CATEGORICAL_UNIQUE
    ]

    if len(cat_cols_test) >= 2:
        print("\n📊 Chi-Square Independence Tests:\n")
        test_count = 0

        for i, col1 in enumerate(cat_cols_test):
            if test_count >= self.config.MAX_CHI_SQUARE_TESTS:
                break
            for col2 in cat_cols_test[i + 1:]:
                if test_count >= self.config.MAX_CHI_SQUARE_TESTS:
                    break
                try:
                    contingency = pd.crosstab(self.df[col1], self.df[col2])
                    if contingency.size == 0:
                        continue
                    chi2, p_val, dof, _ = chi2_contingency(contingency)
                    sig = p_val < self.config.ALPHA
                    statistical_tests['chi_square_tests'].append({
                        'variable1':       col1,
                        'variable2':       col2,
                        'chi2_statistic':  float(chi2),
                        'p_value':         float(p_val),
                        'dof':             int(dof),
                        'significant':     bool(sig),
                    })
                    result_str = "Significant" if sig else "Not significant"
                    print(f"{col1} vs {col2}:")
                    print(f"   χ² = {chi2:.4f}, p = {p_val:.4f}, {result_str}\n")
                    test_count += 1
                except Exception as e:
                    self.logger.warning(f"Chi-square failed {col1} vs {col2}: {e}")

    self.results['statistical_tests'] = statistical_tests
    self._ckpt_save("statistical_tests", statistical_tests)
    self.logger.info(f"Completed {len(statistical_tests['chi_square_tests'])} chi-square tests")
    return statistical_tests


# ==============================================================================
# 12. MODEL READINESS ASSESSMENT
# ==============================================================================

def assess_model_readiness(self) -> Dict[str, Any]:
    """
    Evaluate whether the dataset is ready for model training.

    Fixes vs previous version
    --------------------------
    - Feature count uses only leak-free columns (excludes LEAKY_COLUMNS).
    - Leakage warning only flags columns that are actually still in the
      dataframe AND are in LEAKY_COLUMNS but are NOT 'is_fraud' (the target).
      This prevents the false CRITICAL warning that fired on ['is_fraud'].
    """
    print_section_header("11. MODEL READINESS ASSESSMENT", logger=self.logger)

    cached = self._ckpt_load("model_readiness")
    if cached is not None:
        self.results['model_readiness'] = cached
        return cached

    leaky        = self._leaky()
    quality_data = self.results.get('data_quality', {})
    completeness = quality_data.get('completeness_pct', 0)

    numerical_cols = [
        c for c in get_numeric_columns(self.df, exclude=['is_fraud'])
        if c not in leaky
    ]
    categorical_cols = [
        c for c in get_categorical_columns(
            self.df, exclude=self.config.SKIP_CATEGORICAL_ANALYSIS
        )
        if c not in leaky
    ]
    total_features = len(numerical_cols) + len(categorical_cols)

    # Only flag columns that are actually leaked:
    #   - present in the dataframe
    #   - in LEAKY_COLUMNS
    #   - NOT 'is_fraud' (which is the target, not a leak)
    actual_leaky_in_df = [
        c for c in leaky
        if c in self.df.columns and c != 'is_fraud'
    ]

    model_readiness = {
        'completeness_score':    float(completeness),
        'completeness_pass':     bool(completeness >= self.config.COMPLETENESS_THRESHOLD),
        'sample_size':           len(self.df),
        'sample_size_pass':      bool(len(self.df) >= self.config.MIN_SAMPLE_SIZE),
        'feature_count':         total_features,
        'feature_pass':          bool(total_features >= self.config.MIN_FEATURES),
        'class_balance_score':   0.0,
        'class_balance_pass':    False,
        'overall_readiness':     'NOT READY',
        'leaky_columns_excluded': actual_leaky_in_df,
    }

    if 'is_fraud' in self.df.columns:
        fraud_rate = safe_divide(self.df['is_fraud'].sum(), len(self.df), 0)
        model_readiness['class_balance_score'] = float(fraud_rate * 100)
        model_readiness['class_balance_pass']  = bool(
            self.config.MIN_FRAUD_RATE <= fraud_rate <= self.config.MAX_FRAUD_RATE
        )

        checks = [
            model_readiness['completeness_pass'],
            model_readiness['sample_size_pass'],
            model_readiness['class_balance_pass'],
            model_readiness['feature_pass'],
        ]

        print("\n🎯 Model Readiness Checklist:")
        print(f"   1. Data Completeness : {completeness:.1f}%"
              f"  — {'✅ PASS' if checks[0] else '❌ FAIL'}")
        print(f"   2. Sample Size        : {len(self.df):,}"
              f"  — {'✅ PASS' if checks[1] else '❌ FAIL'}")
        print(f"   3. Class Balance      : {fraud_rate*100:.2f}% fraud"
              f"  — {'✅ PASS' if checks[2] else '❌ FAIL'}")
        print(f"   4. Feature Count      : {total_features} (leak-free)"
              f"  — {'✅ PASS' if checks[3] else '❌ FAIL'}")

        if actual_leaky_in_df:
            print(f"\n   ⚠️  Leaky columns still in dataframe: {actual_leaky_in_df}")
        else:
            print(f"\n   ✅ No leaky columns found in dataframe.")

        model_readiness['overall_readiness'] = (
            'READY'      if sum(checks) >= 3
            else 'NEEDS WORK' if sum(checks) >= 2
            else 'NOT READY'
        )
        print(f"\n   ⚖️  Overall Assessment: {model_readiness['overall_readiness']}")

    self.results['model_readiness'] = model_readiness
    self._ckpt_save("model_readiness", model_readiness)
    self.logger.info(f"Model readiness: {model_readiness['overall_readiness']}")
    return model_readiness


# ==============================================================================
# 13. RECOMMENDATIONS
# ==============================================================================

def generate_recommendations(self) -> List[Dict[str, str]]:
    """
    Generate prioritised, actionable recommendations from EDA results.

    Fixes vs previous version
    --------------------------
    - CRITICAL leakage warning only fires when actual leaked columns are
      still present (not on 'is_fraud' which is the target).
    - Soft leak moved to HIGH priority (18.6pp difference is a real risk).
    - Structural month↔quarter correlation filtered before flagging multicollinearity.
    """
    print_section_header("12. RECOMMENDATIONS & ACTION ITEMS", logger=self.logger)

    cached = self._ckpt_load("recommendations")
    if cached is not None:
        self.results['recommendations'] = cached
        return cached

    recommendations = []
    quality_data = self.results.get('data_quality', {})
    model_data   = self.results.get('model_readiness', {})

    completeness   = quality_data.get('completeness_pct', 100)
    duplicates     = quality_data.get('duplicate_rows', 0)
    fraud_rate     = model_data.get('class_balance_score', 0) / 100
    null_by_fraud  = quality_data.get('time_since_null_by_fraud', {})
    actual_leaky   = model_data.get('leaky_columns_excluded', [])

    # ---- CRITICAL: only fire if genuinely leaked columns remain -----------
    if actual_leaky:
        recommendations.append({
            'priority': 'CRITICAL',
            'category': 'Data Leakage',
            'issue':    (
                f"Leaky columns still present in dataframe: {actual_leaky}. "
                "These are post-event labels that must not enter the feature matrix."
            ),
            'action':   (
                "Add them to config.DROP_ON_LOAD so they are removed on CSV read. "
                "Ensure config.LEAKY_COLUMNS excludes them from X in model training."
            ),
        })

    # ---- HIGH: soft leak (18.6pp difference is a real signal leak) -------
    if null_by_fraud:
        vals = list(null_by_fraud.values())
        if len(vals) == 2 and abs(vals[0] - vals[1]) > 5:
            diff = abs(vals[0] - vals[1])
            recommendations.append({
                'priority': 'HIGH',
                'category': 'Soft Leak Risk',
                'issue':    (
                    f"'time_since_last_transaction' null rate differs by {diff:.1f}pp "
                    "between fraud (0.0%) and non-fraud (18.6%) rows. "
                    "The missingness pattern itself leaks signal."
                ),
                'action':   (
                    "Add 'is_first_transaction' = time_since_last_transaction.isnull()"
                    ".astype(int) BEFORE imputing. "
                    "Then impute nulls with the column median."
                ),
            })

    # ---- Data quality -----------------------------------------------------
    if completeness < self.config.COMPLETENESS_THRESHOLD:
        recommendations.append({
            'priority': 'HIGH',
            'category': 'Data Quality',
            'issue':    f"Data is only {completeness:.1f}% complete",
            'action':   "Implement imputation (mean/median/mode) or collect more data.",
        })

    if duplicates > len(self.df) * 0.01:
        recommendations.append({
            'priority': 'MEDIUM',
            'category': 'Data Quality',
            'issue':    f"{duplicates} duplicate rows found",
            'action':   "Remove duplicates or verify they are valid repeated transactions.",
        })

    # ---- Target / class balance -------------------------------------------
    if 'is_fraud' in self.df.columns:
        if self.df['is_fraud'].sum() == 0:
            recommendations.append({
                'priority': 'CRITICAL',
                'category': 'Target Variable',
                'issue':    "No fraud cases in dataset",
                'action':   "Collect fraud examples or apply SMOTE.",
            })
        elif fraud_rate < self.config.MIN_FRAUD_RATE:
            recommendations.append({
                'priority': 'HIGH',
                'category': 'Class Imbalance',
                'issue':    f"Severe imbalance ({fraud_rate*100:.2f}% fraud)",
                'action':   (
                    "Use SMOTE, class_weight='balanced', or "
                    "XGBoost scale_pos_weight to handle imbalance."
                ),
            })

    # ---- Multicollinearity (filter structural correlations) ---------------
    corr_data   = self.results.get('correlation_analysis', {})
    strong_corr = corr_data.get('strong_correlations', [])
    real_strong = [
        c for c in strong_corr
        if set([c['feature1'], c['feature2']]) != {'month', 'quarter'}
    ]
    if len(real_strong) > 5:
        recommendations.append({
            'priority': 'MEDIUM',
            'category': 'Multicollinearity',
            'issue':    (
                f"{len(real_strong)} strongly correlated feature pairs "
                "(structural month↔quarter excluded)."
            ),
            'action':   (
                "Consider dropping 'quarter' (redundant with 'month') "
                "and applying PCA or Lasso regularisation."
            ),
        })

    # ---- Feature engineering ----------------------------------------------
    recommendations.append({
        'priority': 'MEDIUM',
        'category': 'Feature Engineering',
        'issue':    "Additional features could improve model performance",
        'action':   (
            "Add 'is_first_transaction' flag, amount z-score per account, "
            "hour-of-day risk buckets, and rolling account velocity features."
        ),
    })

    # ---- Model selection --------------------------------------------------
    if 'timestamp' in self.df.columns:
        recommendations.append({
            'priority': 'MEDIUM',
            'category': 'Model Selection',
            'issue':    "Dataset has temporal ordering — random CV will leak future data",
            'action':   (
                "Use sklearn.model_selection.TimeSeriesSplit for cross-validation "
                "and consider sequential models."
            ),
        })

    # Sort by priority
    priority_order = {'CRITICAL': 0, 'HIGH': 1, 'MEDIUM': 2, 'LOW': 3}
    recommendations.sort(key=lambda x: priority_order.get(x['priority'], 4))

    icons = {'CRITICAL': '🔴', 'HIGH': '🟠', 'MEDIUM': '🟡', 'LOW': '🟢'}
    print("\n🎯 Actionable Recommendations:\n")
    for i, rec in enumerate(recommendations, 1):
        icon = icons.get(rec['priority'], '⚪')
        print(f"{i}. {icon} [{rec['priority']}] {rec['category']}")
        print(f"   Issue  : {rec['issue']}")
        print(f"   Action : {rec['action']}\n")

    self.results['recommendations'] = recommendations
    self._ckpt_save("recommendations", recommendations)
    self.logger.info(f"Generated {len(recommendations)} recommendations")
    return recommendations


# ==============================================================================
# 14. FULL PIPELINE ORCHESTRATOR
# ==============================================================================

def run_full_analysis(self, filepath: Optional[str] = None) -> Dict[str, Any]:
    """
    Run all EDA sections in sequence.
    Completed sections are loaded from checkpoints and skipped.
    """
    self.logger.info("=" * 100)
    self.logger.info("STARTING COMPREHENSIVE EDA PIPELINE")
    self.logger.info("=" * 100)

    print(f"\n📁 Checkpoint dir: {self.config.CHECKPOINT_DIR}/")
    print("   Completed sections load from disk automatically.")
    print("   Delete the folder to force a full re-run.\n")

    self.load_data(filepath)
    self.analyze_metadata()
    self.analyze_data_quality()
    self.analyze_numerical_features()
    self.analyze_categorical_features()
    self.analyze_temporal_features()
    self.analyze_correlation()
    self.analyze_feature_importance()
    self.analyze_pca()
    self.analyze_clustering()
    self.analyze_statistical_tests()
    self.assess_model_readiness()
    self.generate_recommendations()

    self.results['timestamp']    = datetime.now().isoformat()
    self.results['dataset_name'] = filepath or self.config.INPUT_CSV

    if self.config.STORE_RAW_DATA:
        self.results['data_sample'] = (
            self.df.head(self.config.SAMPLE_SIZE_FOR_STORAGE).to_dict()
        )

    self.logger.info("=" * 100)
    self.logger.info("EDA PIPELINE COMPLETED SUCCESSFULLY")
    self.logger.info("=" * 100)
    return self.results


# ==============================================================================
# 15. SAVE RESULTS
# ==============================================================================

def save_results(self, output_path: Optional[str] = None):
    """Persist the full results dict to a joblib file."""
    output_path = output_path or self.config.OUTPUT_PICKLE
    joblib.dump(self.results, output_path)
    size = Path(output_path).stat().st_size
    self.logger.info(f"Results saved to: {output_path} ({format_bytes(size)})")
    print(f"\n💾 Results saved to: {output_path}")