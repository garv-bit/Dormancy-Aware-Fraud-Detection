"""
output.py — Load and display saved EDA results from eda_results.joblib

Usage:
    python src/output.py
    python src/output.py path/to/custom_results.joblib
"""

import sys
import io
import joblib
from pathlib import Path
from typing import Any, Dict

# Force UTF-8 on stdout so emoji don't crash on Windows cp1252 terminals
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True
    )


# ==============================================================================
# FORMATTING HELPERS
# ==============================================================================

def _divider(char: str = "=", width: int = 100):
    print(char * width)


def _section(title: str, width: int = 100):
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


def _sub(title: str):
    print(f"\n  -- {title}")


# ==============================================================================
# SECTION PRINTERS
# ==============================================================================

def print_metadata(meta: Dict[str, Any]):
    _section("DATASET METADATA")
    print(f"  Rows        : {meta.get('total_rows', 'N/A'):,}")
    print(f"  Columns     : {meta.get('total_columns', 'N/A')}")
    print(f"  Memory      : {meta.get('memory_mb', 0):.2f} MB")
    print(f"  Bytes / Row : {meta.get('bytes_per_row', 0):.0f}")

    col_info = meta.get('column_info', {})
    if col_info:
        _sub("Column Overview")
        header = f"  {'Column':<35} {'Dtype':<12} {'Nulls %':>8} {'Unique':>10}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for col, info in col_info.items():
            print(
                f"  {col:<35} {info.get('dtype', ''):<12} "
                f"{info.get('null_pct', 0):>7.2f}% "
                f"{info.get('unique', 0):>10,}"
            )


def print_data_quality(dq: Dict[str, Any]):
    _section("DATA QUALITY")
    qm = dq.get('quality_metrics', {})
    for key in ['completeness', 'uniqueness', 'consistency', 'validity', 'overall']:
        print(f"  {key.capitalize():<16}: {qm.get(key, 0):.2f}%")

    print(f"\n  Duplicate rows : {dq.get('duplicate_rows', 0):,} "
          f"({dq.get('duplicate_pct', 0):.2f}%)")
    print(f"  Missing cells  : {dq.get('total_missing', 0):,}")

    null_by_fraud = dq.get('time_since_null_by_fraud', {})
    if null_by_fraud:
        _sub("time_since_last_transaction null rate by label")
        for label, pct in null_by_fraud.items():
            tag = "fraud" if label else "non-fraud"
            print(f"   {tag}: {pct:.2f}% missing")


def print_numerical_analysis(na: Dict[str, Any]):
    _section(f"NUMERICAL FEATURES  ({na.get('feature_count', 0)} columns)")
    for col, stats in na.get('features', {}).items():
        print(f"\n  {col}")
        print(f"    Mean     : {stats.get('mean', 0):.4f}   "
              f"Median : {stats.get('median', 0):.4f}")
        print(f"    Std      : {stats.get('std', 0):.4f}   "
              f"CV     : {stats.get('cv', 0):.2f}%")
        print(f"    Skewness : {stats.get('skewness', 0):.4f}   "
              f"Kurt   : {stats.get('kurtosis', 0):.4f}")
        iqr = stats.get('outliers_iqr', {})
        print(f"    Outliers (IQR)  : {iqr.get('count', 0):,} ({iqr.get('percentage', 0):.2f}%)")
        print(f"    Normal dist     : {stats.get('is_normal', 'N/A')}")


def print_categorical_analysis(ca: Dict[str, Any]):
    _section(f"CATEGORICAL FEATURES  ({ca.get('feature_count', 0)} columns)")
    for col, stats in ca.get('features', {}).items():
        print(f"\n  {col}  ({stats.get('unique_count', 0)} unique — "
              f"{stats.get('cardinality', '?')} cardinality)")
        for val, cnt in stats.get('value_counts', {}).items():
            pct = cnt / max(1, stats.get('top_count', 1)) * stats.get('top_pct', 0)
            print(f"    • {str(val):<25} {cnt:>10,}")


def print_temporal_analysis(ta: Dict[str, Any]):
    _section("TEMPORAL ANALYSIS")
    print(f"  Date Range  : {ta.get('min_date', 'N/A')}  to  {ta.get('max_date', 'N/A')}")
    print(f"  Time Span   : {ta.get('time_span_days', 'N/A')} days")
    print(f"  Peak Hour   : {ta.get('peak_hour', 'N/A')}:00")
    print(f"  Peak Day    : {ta.get('peak_day', 'N/A')}")
    print(f"  Peak Month  : {ta.get('peak_month', 'N/A')}")
    print(f"  Weekend %   : {ta.get('weekend_pct', 0):.1f}%")


def print_correlation(corr: Dict[str, Any]):
    strong = corr.get('strong_correlations', [])
    _section(f"CORRELATION ANALYSIS  ({len(strong)} strong pairs)")
    if strong:
        print(f"\n  {'Feature 1':<35} {'Feature 2':<35} {'Pearson':>8} {'Spearman':>9}")
        print("  " + "-" * 90)
        for c in strong:
            print(f"  {c['feature1']:<35} {c['feature2']:<35} "
                  f"{c['pearson']:>8.3f} {c['spearman']:>9.3f}")
    else:
        print("  No strong correlations found.")


def print_feature_importance(fi: Dict[str, Any]):
    _section("FEATURE IMPORTANCE")
    print(f"  Mode        : {'Supervised' if fi.get('has_target') else 'Unsupervised'}")
    print(f"  Sample rows : {fi.get('sample_size_used', 'N/A'):,}")
    dropped = fi.get('leaky_columns_removed', [])
    if dropped:
        print(f"  Leaky cols removed from X: {dropped}")

    rf = fi.get('rf_importance', {})
    if rf:
        _sub("Random Forest Importance")
        for feat, imp in sorted(rf.items(), key=lambda x: x[1], reverse=True)[:15]:
            bar = "#" * int(imp * 50)
            print(f"  {feat:<35} {imp:.4f}  {bar}")

    gb = fi.get('gb_importance', {})
    if gb:
        _sub("Gradient Boosting Importance")
        for feat, imp in sorted(gb.items(), key=lambda x: x[1], reverse=True)[:15]:
            bar = "#" * int(imp * 50)
            print(f"  {feat:<35} {imp:.4f}  {bar}")

    mi = fi.get('mutual_info', {})
    if mi:
        _sub("Mutual Information")
        for feat, score in sorted(mi.items(), key=lambda x: x[1], reverse=True)[:15]:
            print(f"  {feat:<35} {score:.4f}")


def print_pca(pca: Dict[str, Any]):
    _section("PCA - DIMENSIONALITY REDUCTION")
    print(f"  Components (95% var) : {pca.get('n_components_95', 'N/A')} / "
          f"{pca.get('total_components', 'N/A')}")
    print(f"  Reduction potential  : {pca.get('reduction_potential_pct', 0):.1f}%")
    evr = pca.get('explained_variance_ratio', [])
    if evr:
        _sub("Explained Variance per Component")
        cumulative = 0.0
        for i, v in enumerate(evr[:10], 1):
            cumulative += v
            bar = "#" * int(v * 100)
            print(f"  PC{i:<3}  {v:.4f}  (cumulative {cumulative:.4f})  {bar}")


def print_clustering(cl: Dict[str, Any]):
    _section("CLUSTERING ANALYSIS")
    print(f"  Optimal k    : {cl.get('optimal_k', 'N/A')}")
    print(f"  Sample rows  : {cl.get('sample_size_used', 'N/A'):,}")
    sil = cl.get('silhouette_scores', [])
    if sil:
        best = max(sil)
        print(f"  Best silhouette score : {best:.4f}")
    sizes = cl.get('cluster_sizes', {})
    if sizes:
        _sub("Cluster Sizes")
        total = sum(sizes.values())
        for k, s in sorted(sizes.items()):
            pct = s / max(1, total) * 100
            print(f"   Cluster {k}: {s:,} ({pct:.1f}%)")


def print_statistical_tests(st: Dict[str, Any]):
    chi = st.get('chi_square_tests', [])
    _section(f"STATISTICAL TESTS  ({len(chi)} chi-square tests)")
    if chi:
        sig   = [t for t in chi if t.get('significant')]
        print(f"  Significant pairs: {len(sig)} / {len(chi)}")
        _sub("Results")
        for t in chi:
            flag = "[SIG]" if t.get('significant') else "     "
            print(f"  {flag} {t['variable1']:<25} vs {t['variable2']:<25} "
                  f"chi2={t['chi2_statistic']:.3f}  p={t['p_value']:.4f}")


def print_model_readiness(mr: Dict[str, Any]):
    _section("MODEL READINESS")
    checks = {
        'Completeness':  (mr.get('completeness_pass'), f"{mr.get('completeness_score', 0):.1f}%"),
        'Sample Size':   (mr.get('sample_size_pass'),  f"{mr.get('sample_size', 0):,}"),
        'Class Balance': (mr.get('class_balance_pass'), f"{mr.get('class_balance_score', 0):.2f}% fraud"),
        'Feature Count': (mr.get('feature_pass'),      f"{mr.get('feature_count', 0)} (leak-free)"),
    }
    for label, (passed, value) in checks.items():
        mark = 'PASS' if passed else 'FAIL'
        print(f"  {label:<18}: {value:<25} {mark}")

    leaky = mr.get('leaky_columns_excluded', [])
    if leaky:
        print(f"\n  [WARNING] Leaky columns still in dataframe: {leaky}")
    else:
        print("\n  [OK] No leaky columns found in dataframe.")

    print(f"\n  Overall : {mr.get('overall_readiness', 'UNKNOWN')}")


def print_recommendations(recs: list):
    _section(f"RECOMMENDATIONS  ({len(recs)} total)")
    icons = {'CRITICAL': '[!!!]', 'HIGH': '[!!]', 'MEDIUM': '[!]', 'LOW': '[i]'}
    for i, rec in enumerate(recs, 1):
        icon = icons.get(rec.get('priority', ''), '[ ]')
        print(f"\n  {i}. {icon} [{rec['priority']}] {rec['category']}")
        print(f"     Issue  : {rec['issue']}")
        print(f"     Action : {rec['action']}")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    results_path = sys.argv[1] if len(sys.argv) > 1 else "eda_results.joblib"

    if not Path(results_path).exists():
        print(f"\nERROR: Results file not found: {results_path}")
        print("   Run Main.py first to generate eda_results.joblib")
        sys.exit(1)

    print(f"\nLoading results from: {results_path}")
    results = joblib.load(results_path)

    _divider()
    print(" " * 30 + "EDA RESULTS REPORT")
    _divider()
    if 'timestamp' in results:
        print(f"  Generated : {results['timestamp']}")
    if 'dataset_name' in results:
        print(f"  Dataset   : {results['dataset_name']}")

    print_metadata(          results.get('metadata', {}))
    print_data_quality(      results.get('data_quality', {}))
    print_numerical_analysis(results.get('numerical_analysis', {}))
    print_categorical_analysis(results.get('categorical_analysis', {}))
    print_temporal_analysis( results.get('temporal_analysis', {}))
    print_correlation(       results.get('correlation_analysis', {}))
    print_feature_importance(results.get('feature_importance', {}))
    print_pca(               results.get('pca_analysis', {}))
    print_clustering(        results.get('clustering_analysis', {}))
    print_statistical_tests( results.get('statistical_tests', {}))
    print_model_readiness(   results.get('model_readiness', {}))
    print_recommendations(   results.get('recommendations', []))

    _divider()
    print("  END OF REPORT")
    _divider()


if __name__ == "__main__":
    main()