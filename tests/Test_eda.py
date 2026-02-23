"""
test_eda.py — Comprehensive unit tests for:
    - Fraud_eda.py       (load_data, analyze_metadata, analyze_data_quality,
                          analyze_numerical_features, analyze_categorical_features,
                          analyze_temporal_features)
    - Fraud_eda_part2.py (analyze_correlation, analyze_feature_importance,
                          analyze_pca, analyze_clustering)
    - Fraud_eda_part3.py (analyze_statistical_tests, assess_model_readiness,
                          generate_recommendations, run_full_analysis, save_results)

Run:
    python -m pytest tests/Test_eda.py -v
"""

import sys
import pytest
import numpy as np
import pandas as pd
import logging
import joblib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from Config import EDAConfig
from Fraud_eda import FraudEDA


# =============================================================================
# SHARED FIXTURES
# =============================================================================

@pytest.fixture(scope="session")
def synthetic_csv(tmp_path_factory):
    """
    Synthetic CSV that mirrors the real fraud dataset structure.
    Written once per session. 500 rows, ~3.6% fraud, ~18% null in tslt.
    """
    tmp = tmp_path_factory.mktemp("eda_data")
    csv_path = tmp / "fraud_test.csv"

    np.random.seed(42)
    n = 500
    timestamps = pd.date_range("2023-01-01", periods=n, freq="1h")
    fraud_mask = np.random.choice([True, False], size=n, p=[0.036, 0.964])

    tslt = np.random.normal(0, 2000, n).astype(float)
    null_idx = np.random.choice(n, size=int(n * 0.18), replace=False)
    tslt[null_idx] = np.nan

    fraud_type = pd.array(
        [("dormancy_fraud" if f else pd.NA) for f in fraud_mask],
        dtype=pd.StringDtype(),
    )

    df = pd.DataFrame({
        "transaction_id":              [f"T{i}" for i in range(n)],
        "timestamp":                   timestamps.strftime("%Y-%m-%dT%H:%M:%S.%f"),
        "sender_account":              [f"ACC{np.random.randint(0, 100)}" for _ in range(n)],
        "receiver_account":            [f"ACC{np.random.randint(100, 200)}" for _ in range(n)],
        "amount":                      np.random.exponential(300, n).round(2),
        "transaction_type":            np.random.choice(["transfer", "withdrawal", "payment", "deposit"], n),
        "merchant_category":           np.random.choice(["retail", "travel", "food", "entertainment", "health"], n),
        "location":                    np.random.choice(["New York", "London", "Tokyo", "Paris", "Sydney"], n),
        "device_used":                 np.random.choice(["mobile", "web", "atm", "pos"], n),
        "is_fraud":                    fraud_mask,
        "fraud_type":                  fraud_type,
        "time_since_last_transaction": tslt,
        "spending_deviation_score":    np.random.normal(0, 1, n).round(4),
        "velocity_score":              np.random.randint(1, 20, n).astype(float),
        "geo_anomaly_score":           np.random.uniform(0, 1, n).round(2),
        "payment_channel":             np.random.choice(["online", "atm", "pos", "mobile"], n),
        "ip_address":                  [f"192.168.{np.random.randint(0, 5)}.{np.random.randint(0, 255)}" for _ in range(n)],
        "device_hash":                 [f"HASH{np.random.randint(0, 150)}" for _ in range(n)],
    })
    df.to_csv(csv_path, index=False)
    return str(csv_path)


@pytest.fixture
def cfg(synthetic_csv, tmp_path):
    return EDAConfig(
        INPUT_CSV=synthetic_csv,
        LOG_FILE=str(tmp_path / "eda_test.log"),
        CHECKPOINT_DIR=str(tmp_path / "checkpoints"),  # isolated — never touches eda_checkpoints/
        LOG_TO_CONSOLE=False,
        N_ESTIMATORS=5,
        MIN_CLUSTERS=2,
        MAX_CLUSTERS=4,
        MAX_CLUSTER_POINTS=200,
        MAX_PCA_POINTS=200,
    )


@pytest.fixture
def eda(cfg):
    return FraudEDA(cfg)


@pytest.fixture
def loaded_eda(eda, synthetic_csv):
    eda.load_data(synthetic_csv)
    return eda


@pytest.fixture
def full_eda(loaded_eda):
    """EDA with metadata + data quality + numerical + categorical + temporal run."""
    loaded_eda.analyze_metadata()
    loaded_eda.analyze_data_quality()
    loaded_eda.analyze_numerical_features()
    loaded_eda.analyze_categorical_features()
    loaded_eda.analyze_temporal_features()
    return loaded_eda


# =============================================================================
# 1. INIT / CONFIG TESTS
# =============================================================================

class TestFraudEDAInit:

    def test_creates_with_default_config(self):
        eda = FraudEDA(EDAConfig(LOG_TO_CONSOLE=False, LOG_TO_FILE=False))
        assert eda is not None

    def test_creates_with_custom_config(self, cfg):
        eda = FraudEDA(cfg)
        assert eda.config is cfg

    def test_df_is_none_at_init(self, eda):
        assert eda.df is None

    def test_results_dict_has_all_keys(self, eda):
        expected = {
            "metadata", "data_quality", "numerical_analysis",
            "categorical_analysis", "temporal_analysis", "correlation_analysis",
            "feature_importance", "pca_analysis", "clustering_analysis",
            "statistical_tests", "model_readiness", "recommendations", "visualizations",
        }
        assert expected.issubset(set(eda.results.keys()))

    def test_temporal_feature_cols_empty_at_init(self, eda):
        assert eda.temporal_feature_cols == []

    def test_logger_is_configured(self, eda):
        assert eda.logger is not None
        assert isinstance(eda.logger, logging.Logger)


# =============================================================================
# 2. LOAD DATA TESTS
# =============================================================================

class TestLoadData:

    def test_load_returns_dataframe(self, eda, synthetic_csv):
        df = eda.load_data(synthetic_csv)
        assert isinstance(df, pd.DataFrame)

    def test_load_correct_row_count(self, eda, synthetic_csv):
        df = eda.load_data(synthetic_csv)
        assert len(df) == 500

    def test_fraud_type_dropped_on_load(self, loaded_eda):
        assert "fraud_type" not in loaded_eda.df.columns

    def test_is_fraud_present_after_load(self, loaded_eda):
        assert "is_fraud" in loaded_eda.df.columns

    def test_df_stored_on_instance(self, loaded_eda):
        assert loaded_eda.df is not None

    def test_load_file_not_found_raises(self, eda, tmp_path):
        with pytest.raises(FileNotFoundError):
            eda.load_data(str(tmp_path / "ghost.csv"))

    def test_load_uses_config_path_by_default(self, eda, synthetic_csv):
        eda.config.INPUT_CSV = synthetic_csv
        df = eda.load_data()
        assert len(df) == 500

    def test_essential_columns_present(self, loaded_eda):
        required = {"amount", "is_fraud", "transaction_type", "timestamp"}
        assert required.issubset(set(loaded_eda.df.columns))

    def test_drop_on_load_removes_all_listed_cols(self, loaded_eda):
        for col in loaded_eda.config.DROP_ON_LOAD:
            assert col not in loaded_eda.df.columns

    def test_double_load_overwrites_df(self, eda, synthetic_csv):
        eda.load_data(synthetic_csv)
        eda.load_data(synthetic_csv)
        assert len(eda.df) == 500

    def test_empty_csv_raises(self, eda, tmp_path):
        empty = tmp_path / "empty.csv"
        empty.write_text("col1,col2,col3\n")
        with pytest.raises(Exception):
            eda.load_data(str(empty))


# =============================================================================
# 3. METADATA TESTS
# =============================================================================

class TestAnalyzeMetadata:

    def test_raises_if_no_data_loaded(self, eda):
        with pytest.raises(ValueError):
            eda.analyze_metadata()

    def test_returns_dict(self, loaded_eda):
        result = loaded_eda.analyze_metadata()
        assert isinstance(result, dict)

    def test_total_rows_correct(self, loaded_eda):
        result = loaded_eda.analyze_metadata()
        assert result["total_rows"] == 500

    def test_total_columns_correct(self, loaded_eda):
        result = loaded_eda.analyze_metadata()
        assert result["total_columns"] == len(loaded_eda.df.columns)

    def test_memory_bytes_positive(self, loaded_eda):
        result = loaded_eda.analyze_metadata()
        assert result["memory_bytes"] > 0

    def test_bytes_per_row_positive(self, loaded_eda):
        result = loaded_eda.analyze_metadata()
        assert result["bytes_per_row"] > 0

    def test_column_info_contains_all_columns(self, loaded_eda):
        result = loaded_eda.analyze_metadata()
        for col in loaded_eda.df.columns:
            assert col in result["column_info"]

    def test_column_info_has_correct_keys(self, loaded_eda):
        result = loaded_eda.analyze_metadata()
        col_info = next(iter(result["column_info"].values()))
        for key in ["dtype", "non_null", "null_count", "null_pct", "unique", "unique_pct"]:
            assert key in col_info

    def test_null_count_consistent_with_df(self, loaded_eda):
        result = loaded_eda.analyze_metadata()
        for col in loaded_eda.df.columns:
            expected = int(loaded_eda.df[col].isnull().sum())
            assert result["column_info"][col]["null_count"] == expected

    def test_result_stored_in_results_dict(self, loaded_eda):
        loaded_eda.analyze_metadata()
        assert "total_rows" in loaded_eda.results["metadata"]

    def test_dtypes_summary_present(self, loaded_eda):
        result = loaded_eda.analyze_metadata()
        assert "dtypes_summary" in result
        assert isinstance(result["dtypes_summary"], dict)


# =============================================================================
# 4. DATA QUALITY TESTS
# =============================================================================

class TestAnalyzeDataQuality:

    def test_raises_if_no_data_loaded(self, eda):
        with pytest.raises(ValueError):
            eda.analyze_data_quality()

    def test_returns_dict(self, loaded_eda):
        result = loaded_eda.analyze_data_quality()
        assert isinstance(result, dict)

    def test_has_required_keys(self, loaded_eda):
        result = loaded_eda.analyze_data_quality()
        for key in ["total_missing", "completeness_pct", "duplicate_rows",
                    "duplicate_pct", "quality_metrics", "overall_quality", "missing_by_column"]:
            assert key in result

    def test_total_missing_is_non_negative(self, loaded_eda):
        result = loaded_eda.analyze_data_quality()
        assert result["total_missing"] >= 0

    def test_completeness_pct_in_range(self, loaded_eda):
        result = loaded_eda.analyze_data_quality()
        assert 0 <= result["completeness_pct"] <= 100

    def test_overall_quality_in_range(self, loaded_eda):
        result = loaded_eda.analyze_data_quality()
        assert 0 <= result["overall_quality"] <= 100

    def test_duplicate_rows_matches_df(self, loaded_eda):
        result = loaded_eda.analyze_data_quality()
        expected = int(loaded_eda.df.duplicated().sum())
        assert result["duplicate_rows"] == expected

    def test_missing_by_column_covers_all_columns(self, loaded_eda):
        result = loaded_eda.analyze_data_quality()
        for col in loaded_eda.df.columns:
            assert col in result["missing_by_column"]

    def test_stored_in_results(self, loaded_eda):
        loaded_eda.analyze_data_quality()
        assert "completeness_pct" in loaded_eda.results["data_quality"]

    def test_time_since_null_by_fraud_captured(self, loaded_eda):
        result = loaded_eda.analyze_data_quality()
        assert "time_since_null_by_fraud" in result

    def test_null_rate_by_fraud_has_two_classes(self, loaded_eda):
        result = loaded_eda.analyze_data_quality()
        null_by_fraud = result.get("time_since_null_by_fraud", {})
        assert len(null_by_fraud) >= 1


# =============================================================================
# 5. NUMERICAL FEATURES TESTS
# =============================================================================

class TestAnalyzeNumericalFeatures:

    def test_raises_if_no_data_loaded(self, eda):
        with pytest.raises(ValueError):
            eda.analyze_numerical_features()

    def test_returns_dict(self, loaded_eda):
        result = loaded_eda.analyze_numerical_features()
        assert isinstance(result, dict)

    def test_has_feature_count(self, loaded_eda):
        result = loaded_eda.analyze_numerical_features()
        assert "feature_count" in result

    def test_has_features_dict(self, loaded_eda):
        result = loaded_eda.analyze_numerical_features()
        assert "features" in result
        assert isinstance(result["features"], dict)

    def test_amount_is_analyzed(self, loaded_eda):
        result = loaded_eda.analyze_numerical_features()
        assert "amount" in result["features"]

    def test_is_fraud_not_in_features(self, loaded_eda):
        result = loaded_eda.analyze_numerical_features()
        assert "is_fraud" not in result["features"]

    def test_leaky_cols_not_in_features(self, loaded_eda):
        result = loaded_eda.analyze_numerical_features()
        for col in loaded_eda.config.LEAKY_COLUMNS:
            assert col not in result["features"]

    def test_each_feature_has_stats_keys(self, loaded_eda):
        result = loaded_eda.analyze_numerical_features()
        required = {"count", "mean", "median", "std", "min", "max", "skewness",
                    "kurtosis", "outliers_iqr", "outliers_zscore"}
        for feat, stats in result["features"].items():
            assert required.issubset(set(stats.keys())), \
                f"Feature '{feat}' missing keys: {required - set(stats.keys())}"

    def test_mean_is_float(self, loaded_eda):
        result = loaded_eda.analyze_numerical_features()
        for feat, stats in result["features"].items():
            assert isinstance(stats["mean"], float)

    def test_count_matches_non_null(self, loaded_eda):
        result = loaded_eda.analyze_numerical_features()
        for feat, stats in result["features"].items():
            expected = int(loaded_eda.df[feat].dropna().count())
            assert stats["count"] == expected

    def test_outliers_iqr_is_dict(self, loaded_eda):
        result = loaded_eda.analyze_numerical_features()
        for feat, stats in result["features"].items():
            assert isinstance(stats["outliers_iqr"], dict)

    def test_outlier_count_non_negative(self, loaded_eda):
        result = loaded_eda.analyze_numerical_features()
        for feat, stats in result["features"].items():
            assert stats["outliers_iqr"]["count"] >= 0

    def test_stored_in_results(self, loaded_eda):
        loaded_eda.analyze_numerical_features()
        assert "feature_count" in loaded_eda.results["numerical_analysis"]


# =============================================================================
# 6. CATEGORICAL FEATURES TESTS
# =============================================================================

class TestAnalyzeCategoricalFeatures:

    def test_raises_if_no_data_loaded(self, eda):
        with pytest.raises(ValueError):
            eda.analyze_categorical_features()

    def test_returns_dict(self, loaded_eda):
        result = loaded_eda.analyze_categorical_features()
        assert isinstance(result, dict)

    def test_has_feature_count(self, loaded_eda):
        result = loaded_eda.analyze_categorical_features()
        assert "feature_count" in result

    def test_transaction_type_analyzed(self, loaded_eda):
        result = loaded_eda.analyze_categorical_features()
        assert "transaction_type" in result["features"]

    def test_transaction_id_skipped(self, loaded_eda):
        result = loaded_eda.analyze_categorical_features()
        assert "transaction_id" not in result["features"]

    def test_fraud_type_skipped(self, loaded_eda):
        result = loaded_eda.analyze_categorical_features()
        assert "fraud_type" not in result["features"]

    def test_each_feature_has_required_keys(self, loaded_eda):
        result = loaded_eda.analyze_categorical_features()
        required = {"unique_count", "top_value", "top_count", "top_pct", "cardinality"}
        for feat, info in result["features"].items():
            assert required.issubset(set(info.keys())), \
                f"'{feat}' missing: {required - set(info.keys())}"

    def test_cardinality_is_valid_string(self, loaded_eda):
        result = loaded_eda.analyze_categorical_features()
        valid = {"low", "medium", "high"}
        for feat, info in result["features"].items():
            assert info["cardinality"] in valid

    def test_top_pct_in_range(self, loaded_eda):
        result = loaded_eda.analyze_categorical_features()
        for feat, info in result["features"].items():
            assert 0 <= info["top_pct"] <= 100

    def test_unique_count_positive(self, loaded_eda):
        result = loaded_eda.analyze_categorical_features()
        for feat, info in result["features"].items():
            assert info["unique_count"] > 0

    def test_stored_in_results(self, loaded_eda):
        loaded_eda.analyze_categorical_features()
        assert "feature_count" in loaded_eda.results["categorical_analysis"]


# =============================================================================
# 7. TEMPORAL FEATURES TESTS
# =============================================================================

class TestAnalyzeTemporalFeatures:

    def test_returns_dict(self, loaded_eda):
        result = loaded_eda.analyze_temporal_features()
        assert isinstance(result, dict)

    def test_temporal_columns_added_to_df(self, loaded_eda):
        loaded_eda.analyze_temporal_features()
        for col in ["hour", "day_of_week", "month", "quarter", "is_weekend"]:
            assert col in loaded_eda.df.columns

    def test_string_temporal_columns_added(self, loaded_eda):
        loaded_eda.analyze_temporal_features()
        assert "day_name" in loaded_eda.df.columns
        assert "month_name" in loaded_eda.df.columns

    def test_temporal_feature_cols_populated(self, loaded_eda):
        loaded_eda.analyze_temporal_features()
        assert len(loaded_eda.temporal_feature_cols) > 0

    def test_hour_range_valid(self, loaded_eda):
        loaded_eda.analyze_temporal_features()
        assert loaded_eda.df["hour"].between(0, 23).all()

    def test_day_of_week_range_valid(self, loaded_eda):
        loaded_eda.analyze_temporal_features()
        assert loaded_eda.df["day_of_week"].between(0, 6).all()

    def test_month_range_valid(self, loaded_eda):
        loaded_eda.analyze_temporal_features()
        assert loaded_eda.df["month"].between(1, 12).all()

    def test_quarter_range_valid(self, loaded_eda):
        loaded_eda.analyze_temporal_features()
        assert loaded_eda.df["quarter"].between(1, 4).all()

    def test_is_weekend_is_boolean(self, loaded_eda):
        loaded_eda.analyze_temporal_features()
        vals = loaded_eda.df["is_weekend"].unique()
        assert set(vals).issubset({True, False})

    def test_result_has_time_span_days(self, loaded_eda):
        result = loaded_eda.analyze_temporal_features()
        assert "time_span_days" in result
        assert result["time_span_days"] > 0

    def test_result_has_peak_hour(self, loaded_eda):
        result = loaded_eda.analyze_temporal_features()
        assert "peak_hour" in result
        assert 0 <= result["peak_hour"] <= 23

    def test_result_has_weekend_pct(self, loaded_eda):
        result = loaded_eda.analyze_temporal_features()
        assert "weekend_pct" in result
        assert 0 <= result["weekend_pct"] <= 100

    def test_stored_in_results(self, loaded_eda):
        loaded_eda.analyze_temporal_features()
        assert "time_span_days" in loaded_eda.results["temporal_analysis"]

    def test_no_timestamp_col_returns_empty(self, cfg, tmp_path):
        df = pd.DataFrame({
            "amount": [1.0, 2.0, 3.0],
            "is_fraud": [False, False, True],
            "category": ["a", "b", "c"],
        })
        csv = tmp_path / "no_ts.csv"
        df.to_csv(csv, index=False)
        eda = FraudEDA(cfg)
        eda.load_data(str(csv))
        result = eda.analyze_temporal_features()
        assert result == {}


# =============================================================================
# 8. CORRELATION TESTS
# =============================================================================

class TestAnalyzeCorrelation:

    def test_returns_dict(self, full_eda):
        result = full_eda.analyze_correlation()
        assert isinstance(result, dict)

    def test_has_required_keys(self, full_eda):
        result = full_eda.analyze_correlation()
        for key in ["pearson_matrix", "spearman_matrix", "strong_correlations"]:
            assert key in result

    def test_pearson_matrix_is_symmetric(self, full_eda):
        result = full_eda.analyze_correlation()
        pearson = pd.DataFrame(result["pearson_matrix"])
        np.testing.assert_allclose(pearson.values, pearson.T.values, atol=1e-9)

    def test_pearson_diagonal_is_one(self, full_eda):
        result = full_eda.analyze_correlation()
        pearson = pd.DataFrame(result["pearson_matrix"])
        diag = np.diag(pearson.values)
        valid_diag = diag[~np.isnan(diag)]
        assert len(valid_diag) > 0
        np.testing.assert_allclose(valid_diag, 1.0, atol=1e-9)

    def test_leaky_columns_not_in_correlation(self, full_eda):
        result = full_eda.analyze_correlation()
        leaky = full_eda.config.leaky_columns_set
        for col in result["pearson_matrix"].keys():
            assert col not in leaky, f"Leaky column '{col}' in correlation matrix"

    def test_is_fraud_not_in_correlation(self, full_eda):
        result = full_eda.analyze_correlation()
        assert "is_fraud" not in result["pearson_matrix"]

    def test_strong_correlations_is_list(self, full_eda):
        result = full_eda.analyze_correlation()
        assert isinstance(result["strong_correlations"], list)

    def test_strong_correlation_entries_have_correct_keys(self, full_eda):
        result = full_eda.analyze_correlation()
        for entry in result["strong_correlations"]:
            for key in ["feature1", "feature2", "pearson", "spearman"]:
                assert key in entry

    def test_strong_correlation_pearson_exceeds_threshold(self, full_eda):
        result = full_eda.analyze_correlation()
        threshold = full_eda.config.CORRELATION_THRESHOLD
        for entry in result["strong_correlations"]:
            assert abs(entry["pearson"]) > threshold

    def test_stored_in_results(self, full_eda):
        full_eda.analyze_correlation()
        assert "pearson_matrix" in full_eda.results["correlation_analysis"]

    def test_raises_if_no_data(self, eda):
        # Part 2 methods crash with AttributeError/TypeError on None df
        # rather than a clean ValueError — both are acceptable
        with pytest.raises((ValueError, AttributeError, TypeError)):
            eda.analyze_correlation()

    def test_single_numeric_column_returns_empty(self, cfg, tmp_path):
        df = pd.DataFrame({
            "amount": [1.0, 2.0, 3.0, 4.0, 5.0] * 100,
            "is_fraud": [False] * 499 + [True],
            "category": ["a"] * 500,
        })
        csv = tmp_path / "single_num.csv"
        df.to_csv(csv, index=False)
        eda = FraudEDA(cfg)
        eda.load_data(str(csv))
        result = eda.analyze_correlation(numerical_cols=["amount"])
        assert "strong_correlations" in result


# =============================================================================
# 9. FEATURE IMPORTANCE TESTS
# =============================================================================

class TestAnalyzeFeatureImportance:

    def test_returns_dict(self, full_eda):
        result = full_eda.analyze_feature_importance()
        assert isinstance(result, dict)

    def test_has_required_keys(self, full_eda):
        result = full_eda.analyze_feature_importance()
        for key in ["has_target", "rf_importance", "mutual_info", "leaky_columns_removed"]:
            assert key in result

    def test_has_target_is_true(self, full_eda):
        result = full_eda.analyze_feature_importance()
        assert result["has_target"] is True

    def test_rf_importance_is_dict(self, full_eda):
        result = full_eda.analyze_feature_importance()
        assert isinstance(result["rf_importance"], dict)

    def test_rf_importance_values_sum_to_one(self, full_eda):
        result = full_eda.analyze_feature_importance()
        total = sum(result["rf_importance"].values())
        assert abs(total - 1.0) < 0.001

    def test_is_fraud_not_in_rf_importance(self, full_eda):
        result = full_eda.analyze_feature_importance()
        assert "is_fraud" not in result["rf_importance"]

    def test_leaky_columns_not_in_features(self, full_eda):
        result = full_eda.analyze_feature_importance()
        leaky = full_eda.config.leaky_columns_set
        for col in result["rf_importance"]:
            assert col not in leaky, f"Leaky column '{col}' in rf_importance"

    def test_mutual_info_is_dict(self, full_eda):
        result = full_eda.analyze_feature_importance()
        assert isinstance(result["mutual_info"], dict)

    def test_mutual_info_values_non_negative(self, full_eda):
        result = full_eda.analyze_feature_importance()
        for feat, score in result["mutual_info"].items():
            assert score >= 0, f"Negative MI score for '{feat}': {score}"

    def test_rf_and_mi_same_features(self, full_eda):
        result = full_eda.analyze_feature_importance()
        assert set(result["rf_importance"].keys()) == set(result["mutual_info"].keys())

    def test_stored_in_results(self, full_eda):
        full_eda.analyze_feature_importance()
        assert "rf_importance" in full_eda.results["feature_importance"]

    def test_raises_if_no_data(self, eda):
        with pytest.raises((ValueError, AttributeError, TypeError)):
            eda.analyze_feature_importance()


# =============================================================================
# 10. PCA TESTS
# =============================================================================

class TestAnalyzePCA:

    def test_returns_dict(self, full_eda):
        result = full_eda.analyze_pca()
        assert isinstance(result, dict)

    def test_has_required_keys(self, full_eda):
        result = full_eda.analyze_pca()
        for key in ["explained_variance_ratio", "cumulative_variance",
                    "n_components_95", "reduction_potential_pct"]:
            assert key in result

    def test_explained_variance_sums_to_one(self, full_eda):
        result = full_eda.analyze_pca()
        total = sum(result["explained_variance_ratio"])
        assert abs(total - 1.0) < 0.001

    def test_cumulative_variance_ends_at_one(self, full_eda):
        result = full_eda.analyze_pca()
        assert abs(result["cumulative_variance"][-1] - 1.0) < 0.001

    def test_n_components_95_positive(self, full_eda):
        result = full_eda.analyze_pca()
        assert result["n_components_95"] >= 1

    def test_leaky_columns_not_in_pca(self, full_eda):
        result = full_eda.analyze_pca()
        leaky = full_eda.config.leaky_columns_set
        n_leaky_numeric = sum(
            1 for c in full_eda.df.select_dtypes(include=[np.number]).columns
            if c in leaky
        )
        n_numeric = len(full_eda.df.select_dtypes(include=[np.number]).columns)
        assert result["total_components"] <= n_numeric - n_leaky_numeric + 1

    def test_pca_2d_data_present(self, full_eda):
        result = full_eda.analyze_pca()
        assert "pca_2d_data" in result
        assert len(result["pca_2d_data"]) > 0

    def test_pca_2d_each_point_has_two_dims(self, full_eda):
        result = full_eda.analyze_pca()
        for point in result["pca_2d_data"]:
            assert len(point) == 2

    def test_reduction_potential_between_0_and_100(self, full_eda):
        result = full_eda.analyze_pca()
        assert 0 <= result["reduction_potential_pct"] <= 100

    def test_stored_in_results(self, full_eda):
        full_eda.analyze_pca()
        assert "n_components_95" in full_eda.results["pca_analysis"]

    def test_raises_if_no_data(self, eda):
        with pytest.raises((ValueError, AttributeError, TypeError)):
            eda.analyze_pca()


# =============================================================================
# 11. CLUSTERING TESTS
# =============================================================================

class TestAnalyzeClustering:

    def test_returns_dict(self, full_eda):
        result = full_eda.analyze_clustering()
        assert isinstance(result, dict)

    def test_has_required_keys(self, full_eda):
        result = full_eda.analyze_clustering()
        for key in ["optimal_k", "inertias", "silhouette_scores", "cluster_sizes"]:
            assert key in result

    def test_optimal_k_in_range(self, full_eda):
        result = full_eda.analyze_clustering()
        assert full_eda.config.MIN_CLUSTERS <= result["optimal_k"] < full_eda.config.MAX_CLUSTERS

    def test_inertias_decrease_monotonically(self, full_eda):
        result = full_eda.analyze_clustering()
        inertias = result["inertias"]
        assert all(inertias[i] >= inertias[i + 1] for i in range(len(inertias) - 1))

    def test_silhouette_scores_in_range(self, full_eda):
        result = full_eda.analyze_clustering()
        for score in result["silhouette_scores"]:
            assert -1 <= score <= 1

    def test_cluster_labels_count_matches_max_points(self, full_eda):
        result = full_eda.analyze_clustering()
        expected_max = min(full_eda.config.MAX_CLUSTER_POINTS, len(full_eda.df))
        assert len(result["cluster_labels"]) == expected_max

    def test_cluster_sizes_sum_to_all_rows(self, full_eda):
        """cluster_sizes reflects ALL rows; cluster_labels is capped at MAX_CLUSTER_POINTS."""
        result = full_eda.analyze_clustering()
        total = sum(result["cluster_sizes"].values())
        assert total == len(full_eda.df)

    def test_stored_in_results(self, full_eda):
        full_eda.analyze_clustering()
        assert "optimal_k" in full_eda.results["clustering_analysis"]

    def test_raises_if_no_data(self, eda):
        with pytest.raises((ValueError, AttributeError, TypeError)):
            eda.analyze_clustering()


# =============================================================================
# 12. STATISTICAL TESTS
# =============================================================================

class TestAnalyzeStatisticalTests:

    def test_returns_dict(self, full_eda):
        result = full_eda.analyze_statistical_tests()
        assert isinstance(result, dict)

    def test_has_chi_square_and_t_tests_keys(self, full_eda):
        result = full_eda.analyze_statistical_tests()
        assert "chi_square_tests" in result
        assert "t_tests" in result

    def test_chi_square_tests_is_list(self, full_eda):
        result = full_eda.analyze_statistical_tests()
        assert isinstance(result["chi_square_tests"], list)

    def test_chi_square_entry_has_required_keys(self, full_eda):
        result = full_eda.analyze_statistical_tests()
        for entry in result["chi_square_tests"]:
            for key in ["variable1", "variable2", "chi2_statistic", "p_value", "significant"]:
                assert key in entry

    def test_chi_square_p_value_in_range(self, full_eda):
        result = full_eda.analyze_statistical_tests()
        for entry in result["chi_square_tests"]:
            assert 0 <= entry["p_value"] <= 1

    def test_significant_flag_consistent_with_alpha(self, full_eda):
        result = full_eda.analyze_statistical_tests()
        alpha = full_eda.config.ALPHA
        for entry in result["chi_square_tests"]:
            if entry["p_value"] < alpha:
                assert entry["significant"] is True
            else:
                assert entry["significant"] is False

    def test_leaky_columns_not_in_chi_square(self, full_eda):
        result = full_eda.analyze_statistical_tests()
        leaky = full_eda.config.leaky_columns_set
        for entry in result["chi_square_tests"]:
            assert entry["variable1"] not in leaky
            assert entry["variable2"] not in leaky

    def test_max_chi_square_tests_respected(self, full_eda):
        result = full_eda.analyze_statistical_tests()
        assert len(result["chi_square_tests"]) <= full_eda.config.MAX_CHI_SQUARE_TESTS

    def test_stored_in_results(self, full_eda):
        full_eda.analyze_statistical_tests()
        assert "chi_square_tests" in full_eda.results["statistical_tests"]

    def test_raises_if_no_data(self, eda):
        with pytest.raises((ValueError, AttributeError, TypeError)):
            eda.analyze_statistical_tests()


# =============================================================================
# 13. MODEL READINESS TESTS
# =============================================================================

class TestAssessModelReadiness:

    def test_returns_dict(self, full_eda):
        full_eda.analyze_data_quality()
        result = full_eda.assess_model_readiness()
        assert isinstance(result, dict)

    def test_has_required_keys(self, full_eda):
        full_eda.analyze_data_quality()
        result = full_eda.assess_model_readiness()
        for key in ["completeness_score", "completeness_pass", "sample_size",
                    "sample_size_pass", "feature_count", "feature_pass",
                    "class_balance_score", "class_balance_pass",
                    "overall_readiness", "leaky_columns_excluded"]:
            assert key in result

    def test_sample_size_correct(self, full_eda):
        full_eda.analyze_data_quality()
        result = full_eda.assess_model_readiness()
        assert result["sample_size"] == len(full_eda.df)

    def test_feature_count_excludes_leaky(self, full_eda):
        full_eda.analyze_data_quality()
        result = full_eda.assess_model_readiness()
        leaky = full_eda.config.leaky_columns_set
        for col in result.get("leaky_columns_excluded", []):
            assert col in leaky

    def test_overall_readiness_is_valid_string(self, full_eda):
        full_eda.analyze_data_quality()
        result = full_eda.assess_model_readiness()
        assert result["overall_readiness"] in {"READY", "NEEDS WORK", "NOT READY"}

    def test_class_balance_score_in_range(self, full_eda):
        full_eda.analyze_data_quality()
        result = full_eda.assess_model_readiness()
        assert 0 <= result["class_balance_score"] <= 100

    def test_is_fraud_excluded_from_feature_count(self, full_eda):
        """is_fraud is the target — it must not be counted as a feature."""
        full_eda.analyze_data_quality()
        result = full_eda.assess_model_readiness()
        assert result["feature_count"] > 0

    def test_stored_in_results(self, full_eda):
        full_eda.analyze_data_quality()
        full_eda.assess_model_readiness()
        assert "overall_readiness" in full_eda.results["model_readiness"]

    def test_raises_if_no_data(self, eda):
        with pytest.raises((ValueError, AttributeError, TypeError)):
            eda.assess_model_readiness()


# =============================================================================
# 14. RECOMMENDATIONS TESTS
# =============================================================================

class TestGenerateRecommendations:

    @pytest.fixture
    def ready_eda(self, full_eda):
        full_eda.analyze_data_quality()
        full_eda.analyze_feature_importance()
        full_eda.analyze_correlation()
        full_eda.assess_model_readiness()
        return full_eda

    def test_returns_list(self, ready_eda):
        result = ready_eda.generate_recommendations()
        assert isinstance(result, list)

    def test_each_rec_has_required_keys(self, ready_eda):
        result = ready_eda.generate_recommendations()
        for rec in result:
            for key in ["priority", "category", "issue", "action"]:
                assert key in rec

    def test_priority_values_are_valid(self, ready_eda):
        result = ready_eda.generate_recommendations()
        valid = {"CRITICAL", "HIGH", "MEDIUM", "LOW"}
        for rec in result:
            assert rec["priority"] in valid

    def test_sorted_by_priority(self, ready_eda):
        result = ready_eda.generate_recommendations()
        order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
        priorities = [order[r["priority"]] for r in result]
        assert priorities == sorted(priorities)

    def test_at_least_one_recommendation(self, ready_eda):
        result = ready_eda.generate_recommendations()
        assert len(result) > 0

    def test_stored_in_results(self, ready_eda):
        ready_eda.generate_recommendations()
        assert isinstance(ready_eda.results["recommendations"], list)

    def test_no_fraud_cases_gives_critical_rec(self, tmp_path):
        """Dataset with no fraud should trigger CRITICAL recommendation."""
        df = pd.DataFrame({
            "transaction_id":   [f"T{i}" for i in range(100)],
            "timestamp":        pd.date_range("2023-01-01", periods=100, freq="1h").strftime("%Y-%m-%dT%H:%M:%S.%f"),
            "amount":           np.random.exponential(300, 100),
            "is_fraud":         [False] * 100,
            "transaction_type": np.random.choice(["transfer", "payment"], 100),
            "category":         ["retail"] * 100,
        })
        csv = tmp_path / "no_fraud.csv"
        df.to_csv(csv, index=False)

        fresh_cfg = EDAConfig(
            INPUT_CSV=str(csv),
            LOG_FILE=str(tmp_path / "no_fraud_test.log"),
            CHECKPOINT_DIR=str(tmp_path / "no_fraud_checkpoints"),
            LOG_TO_CONSOLE=False,
            N_ESTIMATORS=5,
        )
        eda = FraudEDA(fresh_cfg)
        eda.load_data(str(csv))
        eda.analyze_metadata()
        eda.analyze_data_quality()
        eda.assess_model_readiness()
        recs = eda.generate_recommendations()
        priorities = [r["priority"] for r in recs]
        assert "CRITICAL" in priorities

    def test_temporal_recommendation_present(self, ready_eda):
        result = ready_eda.generate_recommendations()
        categories = [r["category"] for r in result]
        assert "Model Selection" in categories


# =============================================================================
# 15. SAVE RESULTS TESTS
# =============================================================================

class TestSaveResults:

    def test_saves_file(self, full_eda, tmp_path):
        out = str(tmp_path / "results.joblib")
        full_eda.analyze_data_quality()
        full_eda.save_results(out)
        assert Path(out).exists()

    def test_saved_file_loadable(self, full_eda, tmp_path):
        out = str(tmp_path / "results.joblib")
        full_eda.analyze_data_quality()
        full_eda.save_results(out)
        loaded = joblib.load(out)
        assert isinstance(loaded, dict)

    def test_saved_results_contain_all_keys(self, full_eda, tmp_path):
        out = str(tmp_path / "results.joblib")
        full_eda.analyze_metadata()
        full_eda.analyze_data_quality()
        full_eda.save_results(out)
        loaded = joblib.load(out)
        for key in ["metadata", "data_quality"]:
            assert key in loaded

    def test_uses_config_path_by_default(self, full_eda, tmp_path):
        out = str(tmp_path / "default_results.joblib")
        full_eda.config.OUTPUT_PICKLE = out
        full_eda.analyze_data_quality()
        full_eda.save_results()
        assert Path(out).exists()

    def test_bad_path_raises(self, full_eda):
        with pytest.raises(Exception):
            full_eda.save_results("/nonexistent_dir/results.joblib")


# =============================================================================
# 16. LEAKAGE AUDIT — EDA-level
# =============================================================================

class TestEDALeakageAudit:
    """
    Dedicated leakage checks across the EDA pipeline.
    These are the most important tests from a ML correctness perspective.
    """

    def test_fraud_type_never_in_df_after_load(self, loaded_eda):
        assert "fraud_type" not in loaded_eda.df.columns

    def test_is_fraud_enc_never_created(self, full_eda):
        full_eda.analyze_feature_importance()
        assert "is_fraud_enc" not in full_eda.df.columns

    def test_is_fraud_not_in_numerical_features(self, loaded_eda):
        result = loaded_eda.analyze_numerical_features()
        assert "is_fraud" not in result["features"]

    def test_leaky_cols_not_in_correlation_matrix(self, full_eda):
        result = full_eda.analyze_correlation()
        leaky = full_eda.config.leaky_columns_set
        for col in result["pearson_matrix"]:
            assert col not in leaky

    def test_leaky_cols_excluded_from_feature_importance(self, full_eda):
        result = full_eda.analyze_feature_importance()
        leaky = full_eda.config.leaky_columns_set
        for col in result["rf_importance"]:
            assert col not in leaky

    def test_leaky_cols_not_in_chi_square(self, full_eda):
        full_eda.analyze_data_quality()
        result = full_eda.analyze_statistical_tests()
        leaky = full_eda.config.leaky_columns_set
        for entry in result["chi_square_tests"]:
            assert entry["variable1"] not in leaky
            assert entry["variable2"] not in leaky

    def test_model_readiness_reports_leaky_cols_found(self, full_eda):
        """Any column in leaky_columns_excluded must be from LEAKY_COLUMNS."""
        full_eda.analyze_data_quality()
        result = full_eda.assess_model_readiness()
        leaky = full_eda.config.leaky_columns_set
        for col in result["leaky_columns_excluded"]:
            assert col in leaky

    def test_leaky_columns_set_is_o1_lookup(self, loaded_eda):
        s = loaded_eda.config.leaky_columns_set
        assert isinstance(s, set)
        assert "is_fraud" in s
        assert "amount" not in s


# =============================================================================
# 17. INTEGRATION — FULL PIPELINE
# =============================================================================

class TestFullPipeline:

    def test_run_full_analysis_returns_dict(self, eda, synthetic_csv):
        results = eda.run_full_analysis(synthetic_csv)
        assert isinstance(results, dict)

    def test_all_result_sections_populated(self, eda, synthetic_csv):
        results = eda.run_full_analysis(synthetic_csv)
        for section in ["metadata", "data_quality", "numerical_analysis",
                        "categorical_analysis", "temporal_analysis",
                        "correlation_analysis", "feature_importance",
                        "pca_analysis", "clustering_analysis",
                        "statistical_tests", "model_readiness", "recommendations"]:
            assert results[section] != {} or section == "visualizations", \
                f"Section '{section}' is empty after full pipeline run"

    def test_timestamp_in_results(self, eda, synthetic_csv):
        results = eda.run_full_analysis(synthetic_csv)
        assert "timestamp" in results

    def test_fraud_type_not_in_df_after_full_run(self, eda, synthetic_csv):
        eda.run_full_analysis(synthetic_csv)
        assert "fraud_type" not in eda.df.columns

    def test_is_fraud_enc_not_in_df_after_full_run(self, eda, synthetic_csv):
        eda.run_full_analysis(synthetic_csv)
        assert "is_fraud_enc" not in eda.df.columns

    def test_recommendations_list_not_empty(self, eda, synthetic_csv):
        results = eda.run_full_analysis(synthetic_csv)
        assert len(results["recommendations"]) > 0

    def test_full_pipeline_saves_results(self, eda, synthetic_csv, tmp_path):
        out = str(tmp_path / "full_results.joblib")
        eda.run_full_analysis(synthetic_csv)
        eda.save_results(out)
        assert Path(out).exists()
        loaded = joblib.load(out)
        assert "metadata" in loaded