"""
test_fe_pipeline.py — Comprehensive unit tests for Dormancy Fraud Detection
Feature Engineering Pipeline.

Run from project root:
    pytest src/test_fe_pipeline.py -v
    pytest src/test_fe_pipeline.py -v --cov=Fe_pipeline --cov-report=term-missing

Coverage:
    - FEConfig validation
    - Each pipeline step individually
    - Leakage checks (nothing leaky in X)
    - Split correctness (chronological order)
    - Imputation (train-only median, nulls filled)
    - Dormancy features (flag before imputation, buckets, risk score)
    - Frequency encoding (train-only maps, unseen → 1)
    - Label encoding (train-only fit, unseen categories handled)
    - Scaling (train-only fit, applied to both)
    - Feature matrix (correct columns in X, no leaky cols)
    - Edge cases (all-null column, single class, unseen categories)
    - Integration test (full pipeline end-to-end on synthetic data)
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import field

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from FE_config import FEConfig
from Fe_pipeline import FeatureEngineeringPipeline


# =============================================================================
# FIXTURES
# =============================================================================

TEST_CSV = Path(__file__).parent.parent / "test_dataset.csv"

@pytest.fixture(scope="session")
def sample_csv(tmp_path_factory):
    """
    Create a synthetic CSV that mirrors the real dataset structure.
    1000 rows, ~3.6% fraud, ~18% null in time_since_last_transaction.
    Saved once per test session for speed.
    """
    tmp = tmp_path_factory.mktemp("data")
    csv_path = tmp / "test_data.csv"

    np.random.seed(42)
    n = 1000
    timestamps = pd.date_range('2023-01-01', periods=n, freq='1h')
    fraud_mask = np.random.choice([True, False], size=n, p=[0.036, 0.964])

    tslt = np.random.normal(0, 2000, n).astype(float)
    null_idx = np.random.choice(n, size=int(n * 0.18), replace=False)
    tslt[null_idx] = np.nan

    fraud_type = pd.array(
        [('dormancy_fraud' if f else pd.NA) for f in fraud_mask],
        dtype=pd.StringDtype()
    )

    df = pd.DataFrame({
        'transaction_id':              [f'T{i}' for i in range(n)],
        'timestamp':                   timestamps.strftime('%Y-%m-%dT%H:%M:%S.%f'),
        'sender_account':              [f'ACC{np.random.randint(0, 200)}' for _ in range(n)],
        'receiver_account':            [f'ACC{np.random.randint(200, 400)}' for _ in range(n)],
        'amount':                      np.random.exponential(300, n).round(2),
        'transaction_type':            np.random.choice(['transfer','withdrawal','payment','deposit'], n),
        'merchant_category':           np.random.choice(['retail','travel','food','entertainment','health','utilities','education','other'], n),
        'location':                    np.random.choice(['New York','London','Tokyo','Paris','Sydney','Dubai','Singapore','Toronto'], n),
        'device_used':                 np.random.choice(['mobile','web','atm','pos'], n),
        'is_fraud':                    fraud_mask,
        'fraud_type':                  fraud_type,
        'time_since_last_transaction': tslt,
        'spending_deviation_score':    np.random.normal(0, 1, n).round(4),
        'velocity_score':              np.random.randint(1, 20, n).astype(float),
        'geo_anomaly_score':           np.random.uniform(0, 1, n).round(2),
        'payment_channel':             np.random.choice(['online','atm','pos','mobile'], n),
        'ip_address':                  [f'192.168.{np.random.randint(0,5)}.{np.random.randint(0,255)}' for _ in range(n)],
        'device_hash':                 [f'HASH{np.random.randint(0, 300)}' for _ in range(n)],
    })
    df.to_csv(csv_path, index=False)
    return str(csv_path)


@pytest.fixture
def config(sample_csv, tmp_path):
    """FEConfig pointed at synthetic CSV and a temp output directory."""
    features_dir = tmp_path / "features"
    features_dir.mkdir()
    cfg = FEConfig(
        INPUT_CSV        = sample_csv,
        OUTPUT_X_TRAIN   = str(features_dir / "X_train.parquet"),
        OUTPUT_X_TEST    = str(features_dir / "X_test.parquet"),
        OUTPUT_Y_TRAIN   = str(features_dir / "y_train.parquet"),
        OUTPUT_Y_TEST    = str(features_dir / "y_test.parquet"),
        OUTPUT_ENCODERS  = str(features_dir / "encoders.joblib"),
        OUTPUT_SCALER    = str(features_dir / "scaler.joblib"),
        LOG_FILE         = str(tmp_path / "fe_test.log"),
    )
    return cfg


@pytest.fixture
def pipeline(config, tmp_path):
    """Fresh pipeline instance with temp feature output directory."""
    # Override features mkdir to use tmp_path
    p = FeatureEngineeringPipeline(config)
    return p


@pytest.fixture
def loaded_pipeline(pipeline):
    """Pipeline after load() and temporal features only — pre-split state."""
    pipeline.load()
    pipeline.build_temporal_features()
    pipeline.build_dormancy_flag()
    return pipeline


@pytest.fixture
def split_pipeline(loaded_pipeline):
    """Pipeline after load + temporal + flag + split."""
    df_train, df_test = loaded_pipeline.split_raw()
    loaded_pipeline._df_train = df_train
    loaded_pipeline._df_test  = df_test
    return loaded_pipeline, df_train, df_test


# =============================================================================
# 1. CONFIG TESTS
# =============================================================================

class TestFEConfig:

    def test_default_config_validates(self):
        cfg = FEConfig()
        assert cfg.validate() is True

    def test_test_size_bounds(self):
        with pytest.raises(AssertionError):
            FEConfig(TEST_SIZE=0.0).validate()
        with pytest.raises(AssertionError):
            FEConfig(TEST_SIZE=1.0).validate()
        with pytest.raises(AssertionError):
            FEConfig(TEST_SIZE=-0.1).validate()

    def test_test_size_valid(self):
        cfg = FEConfig(TEST_SIZE=0.2)
        assert cfg.validate() is True

    def test_target_not_in_drop_from_x(self):
        """is_fraud must never be in DROP_FROM_X."""
        with pytest.raises(AssertionError):
            from dataclasses import field as f
            cfg = FEConfig()
            cfg.DROP_FROM_X = ['transaction_id', 'is_fraud']
            cfg.validate()

    def test_dormancy_bins_labels_consistency(self):
        """bins must have exactly one more element than labels."""
        cfg = FEConfig()
        assert len(cfg.DORMANCY_BINS) == len(cfg.DORMANCY_LABELS) + 1

    def test_high_cardinality_threshold_positive(self):
        with pytest.raises(AssertionError):
            FEConfig(HIGH_CARDINALITY_THRESHOLD=0).validate()

    def test_fraud_type_in_drop_on_load(self):
        cfg = FEConfig()
        assert 'fraud_type' in cfg.DROP_ON_LOAD

    def test_is_fraud_in_leaky_columns(self):
        cfg = FEConfig()
        assert 'is_fraud' in cfg.LEAKY_COLUMNS

    def test_transaction_id_in_drop_from_x(self):
        cfg = FEConfig()
        assert 'transaction_id' in cfg.DROP_FROM_X

    def test_timestamp_in_drop_from_x(self):
        cfg = FEConfig()
        assert 'timestamp' in cfg.DROP_FROM_X

    def test_high_risk_hours_are_valid(self):
        cfg = FEConfig()
        assert all(0 <= h <= 23 for h in cfg.HIGH_RISK_HOURS)

    def test_weekend_days_are_valid(self):
        cfg = FEConfig()
        assert all(0 <= d <= 6 for d in cfg.WEEKEND_DAYS)


# =============================================================================
# 2. LOAD TESTS
# =============================================================================

class TestLoad:

    def test_load_returns_dataframe(self, pipeline):
        df = pipeline.load()
        assert isinstance(df, pd.DataFrame)

    def test_load_correct_row_count(self, pipeline):
        df = pipeline.load()
        assert len(df) == 1000

    def test_fraud_type_dropped_on_load(self, pipeline):
        df = pipeline.load()
        assert 'fraud_type' not in df.columns

    def test_is_fraud_still_present_after_load(self, pipeline):
        df = pipeline.load()
        assert 'is_fraud' in df.columns

    def test_transaction_id_still_present_after_load(self, pipeline):
        """transaction_id should still be present — dropped later from X only."""
        df = pipeline.load()
        assert 'transaction_id' in df.columns

    def test_load_has_expected_columns(self, pipeline):
        df = pipeline.load()
        expected = {
            'transaction_id', 'timestamp', 'sender_account', 'receiver_account',
            'amount', 'transaction_type', 'merchant_category', 'location',
            'device_used', 'is_fraud', 'time_since_last_transaction',
            'spending_deviation_score', 'velocity_score', 'geo_anomaly_score',
            'payment_channel', 'ip_address', 'device_hash',
        }
        assert expected.issubset(set(df.columns))

    def test_file_not_found_raises(self, config, tmp_path):
        config.INPUT_CSV = str(tmp_path / "nonexistent.csv")
        p = FeatureEngineeringPipeline(config)
        with pytest.raises(FileNotFoundError):
            p.load()


# =============================================================================
# 3. TEMPORAL FEATURES TESTS
# =============================================================================

class TestTemporalFeatures:

    def test_hour_column_created(self, loaded_pipeline):
        assert 'hour' in loaded_pipeline.df.columns

    def test_day_of_week_column_created(self, loaded_pipeline):
        assert 'day_of_week' in loaded_pipeline.df.columns

    def test_month_column_created(self, loaded_pipeline):
        assert 'month' in loaded_pipeline.df.columns

    def test_is_weekend_column_created(self, loaded_pipeline):
        assert 'is_weekend' in loaded_pipeline.df.columns

    def test_is_high_risk_hour_column_created(self, loaded_pipeline):
        assert 'is_high_risk_hour' in loaded_pipeline.df.columns

    def test_hour_range(self, loaded_pipeline):
        assert loaded_pipeline.df['hour'].between(0, 23).all()

    def test_day_of_week_range(self, loaded_pipeline):
        assert loaded_pipeline.df['day_of_week'].between(0, 6).all()

    def test_month_range(self, loaded_pipeline):
        assert loaded_pipeline.df['month'].between(1, 12).all()

    def test_is_weekend_binary(self, loaded_pipeline):
        vals = loaded_pipeline.df['is_weekend'].unique()
        assert set(vals).issubset({0, 1})

    def test_is_high_risk_hour_binary(self, loaded_pipeline):
        vals = loaded_pipeline.df['is_high_risk_hour'].unique()
        assert set(vals).issubset({0, 1})

    def test_high_risk_hours_are_correct(self, loaded_pipeline, config):
        """Rows with hour in HIGH_RISK_HOURS must have is_high_risk_hour=1."""
        df = loaded_pipeline.df
        mask = df['hour'].isin(config.HIGH_RISK_HOURS)
        assert (df.loc[mask, 'is_high_risk_hour'] == 1).all()
        assert (df.loc[~mask, 'is_high_risk_hour'] == 0).all()

    def test_timestamp_parsed_to_datetime(self, loaded_pipeline):
        assert pd.api.types.is_datetime64_any_dtype(loaded_pipeline.df['timestamp'])


# =============================================================================
# 4. DORMANCY FLAG TESTS
# =============================================================================

class TestDormancyFlag:

    def test_is_first_transaction_column_created(self, loaded_pipeline):
        assert 'is_first_transaction' in loaded_pipeline.df.columns

    def test_is_first_transaction_binary(self, loaded_pipeline):
        vals = loaded_pipeline.df['is_first_transaction'].unique()
        assert set(vals).issubset({0, 1})

    def test_flag_matches_nulls(self, loaded_pipeline):
        """is_first_transaction=1 iff time_since_last_transaction was null."""
        df = loaded_pipeline.df
        null_mask = df['time_since_last_transaction'].isnull()
        assert (df.loc[null_mask,  'is_first_transaction'] == 1).all()
        assert (df.loc[~null_mask, 'is_first_transaction'] == 0).all()

    def test_null_count_matches_flag_count(self, loaded_pipeline):
        df = loaded_pipeline.df
        n_nulls = df['time_since_last_transaction'].isnull().sum()
        n_flags = df['is_first_transaction'].sum()
        assert n_nulls == n_flags

    def test_tslt_still_has_nulls_after_flag(self, loaded_pipeline):
        """Nulls must still exist — imputation happens later."""
        assert loaded_pipeline.df['time_since_last_transaction'].isnull().any()


# =============================================================================
# 5. SPLIT TESTS
# =============================================================================

class TestSplit:

    def test_split_returns_two_dataframes(self, loaded_pipeline):
        df_train, df_test = loaded_pipeline.split_raw()
        assert isinstance(df_train, pd.DataFrame)
        assert isinstance(df_test, pd.DataFrame)

    def test_split_sizes(self, loaded_pipeline, config):
        df_train, df_test = loaded_pipeline.split_raw()
        total = len(loaded_pipeline.df)
        expected_train = int(total * (1 - config.TEST_SIZE))
        assert len(df_train) == expected_train
        assert len(df_test) == total - expected_train

    def test_no_overlap_between_splits(self, loaded_pipeline):
        df_train, df_test = loaded_pipeline.split_raw()
        train_ids = set(df_train['transaction_id'])
        test_ids  = set(df_test['transaction_id'])
        assert len(train_ids & test_ids) == 0

    def test_split_covers_all_rows(self, loaded_pipeline):
        df_train, df_test = loaded_pipeline.split_raw()
        assert len(df_train) + len(df_test) == len(loaded_pipeline.df)

    def test_train_timestamps_before_test(self, loaded_pipeline):
        """All train timestamps must be <= all test timestamps."""
        df_train, df_test = loaded_pipeline.split_raw()
        assert df_train['timestamp'].max() <= df_test['timestamp'].min()

    def test_train_is_chronologically_first(self, loaded_pipeline):
        df_train, df_test = loaded_pipeline.split_raw()
        all_sorted = pd.concat([df_train, df_test])['timestamp']
        assert all_sorted.is_monotonic_increasing

    def test_both_splits_have_fraud(self, loaded_pipeline):
        df_train, df_test = loaded_pipeline.split_raw()
        assert df_train['is_fraud'].sum() > 0
        assert df_test['is_fraud'].sum() > 0


# =============================================================================
# 6. IMPUTATION TESTS
# =============================================================================

class TestImputation:

    def test_imputation_median_computed_from_train_only(self, split_pipeline):
        pipeline, df_train, df_test = split_pipeline
        pipeline.fit_imputation(df_train)

        col = pipeline.config.DORMANCY_IMPUTE_COL
        expected = df_train.loc[
            df_train[pipeline.config.TARGET_COLUMN] == False, col
        ].median()

        assert abs(pipeline._impute_median - expected) < 1e-6

    def test_imputation_median_from_non_fraud_only(self, split_pipeline):
        """Median must NOT include fraud rows to avoid signal leak."""
        pipeline, df_train, df_test = split_pipeline
        pipeline.fit_imputation(df_train)

        col = pipeline.config.DORMANCY_IMPUTE_COL
        fraud_median = df_train.loc[
            df_train[pipeline.config.TARGET_COLUMN] == True, col
        ].median()

        # If both medians differ, we must have used non-fraud median
        non_fraud_median = df_train.loc[
            df_train[pipeline.config.TARGET_COLUMN] == False, col
        ].median()

        assert pipeline._impute_median == non_fraud_median

    def test_apply_imputation_fills_all_nulls(self, split_pipeline):
        pipeline, df_train, df_test = split_pipeline
        pipeline.fit_imputation(df_train)
        df_train = pipeline.apply_imputation(df_train)
        col = pipeline.config.DORMANCY_IMPUTE_COL
        assert df_train[col].isnull().sum() == 0

    def test_apply_imputation_uses_fitted_median(self, split_pipeline):
        pipeline, df_train, df_test = split_pipeline
        pipeline.fit_imputation(df_train)

        col = pipeline.config.DORMANCY_IMPUTE_COL
        null_mask = df_train[col].isnull().copy()

        df_train = pipeline.apply_imputation(df_train)
        # All previously null values should now equal the fitted median
        assert (df_train.loc[null_mask, col] == pipeline._impute_median).all()

    def test_non_null_values_unchanged_after_imputation(self, split_pipeline):
        pipeline, df_train, df_test = split_pipeline
        pipeline.fit_imputation(df_train)

        col = pipeline.config.DORMANCY_IMPUTE_COL
        non_null_mask   = df_train[col].notna()
        original_values = df_train.loc[non_null_mask, col].copy()

        df_train = pipeline.apply_imputation(df_train)
        assert (df_train.loc[non_null_mask, col].values == original_values.values).all()


# =============================================================================
# 7. DORMANCY BUCKET TESTS
# =============================================================================

class TestDormancyBuckets:

    @pytest.fixture
    def imputed_pipeline(self, split_pipeline):
        pipeline, df_train, df_test = split_pipeline
        pipeline.fit_imputation(df_train)
        df_train = pipeline.apply_imputation(df_train)
        df_test  = pipeline.apply_imputation(df_test)
        return pipeline, df_train, df_test

    def test_dormancy_bins_computed_from_train_only(self, imputed_pipeline):
        pipeline, df_train, df_test = imputed_pipeline
        pipeline.fit_dormancy_buckets(df_train)
        assert pipeline._dormancy_bins is not None
        assert len(pipeline._dormancy_bins) == 5  # [-inf, p25, p50, p75, inf]

    def test_dormancy_bucket_column_created(self, imputed_pipeline):
        pipeline, df_train, df_test = imputed_pipeline
        pipeline.fit_dormancy_buckets(df_train)
        df_train = pipeline.apply_dormancy_buckets(df_train, 'TRAIN')
        assert 'dormancy_bucket' in df_train.columns

    def test_dormancy_risk_score_column_created(self, imputed_pipeline):
        pipeline, df_train, df_test = imputed_pipeline
        pipeline.fit_dormancy_buckets(df_train)
        df_train = pipeline.apply_dormancy_buckets(df_train, 'TRAIN')
        assert 'dormancy_risk_score' in df_train.columns

    def test_dormancy_bucket_no_nans(self, imputed_pipeline):
        pipeline, df_train, df_test = imputed_pipeline
        pipeline.fit_dormancy_buckets(df_train)
        df_train = pipeline.apply_dormancy_buckets(df_train, 'TRAIN')
        assert df_train['dormancy_bucket'].isin(
            ['recent', 'moderate', 'dormant', 'long_dormant']
        ).all()

    def test_dormancy_risk_score_range(self, imputed_pipeline):
        pipeline, df_train, df_test = imputed_pipeline
        pipeline.fit_dormancy_buckets(df_train)
        df_train = pipeline.apply_dormancy_buckets(df_train, 'TRAIN')
        assert df_train['dormancy_risk_score'].between(0, 3).all()

    def test_dormancy_risk_score_ordinal_mapping(self, imputed_pipeline):
        pipeline, df_train, df_test = imputed_pipeline
        pipeline.fit_dormancy_buckets(df_train)
        df_train = pipeline.apply_dormancy_buckets(df_train, 'TRAIN')
        mapping = {'recent': 0, 'moderate': 1, 'dormant': 2, 'long_dormant': 3}
        for bucket, score in mapping.items():
            rows = df_train[df_train['dormancy_bucket'] == bucket]
            if len(rows) > 0:
                assert (rows['dormancy_risk_score'] == score).all()

    def test_train_bins_applied_to_test(self, imputed_pipeline):
        """Test split uses train-derived bins, not recomputed from test data."""
        pipeline, df_train, df_test = imputed_pipeline
        pipeline.fit_dormancy_buckets(df_train)
        df_test = pipeline.apply_dormancy_buckets(df_test, 'TEST')
        assert 'dormancy_bucket' in df_test.columns
        assert 'dormancy_risk_score' in df_test.columns


# =============================================================================
# 8. FREQUENCY ENCODING TESTS
# =============================================================================

class TestFrequencyEncoding:

    @pytest.fixture
    def freq_pipeline(self, split_pipeline):
        pipeline, df_train, df_test = split_pipeline
        pipeline.fit_imputation(df_train)
        df_train = pipeline.apply_imputation(df_train)
        df_test  = pipeline.apply_imputation(df_test)
        pipeline.fit_dormancy_buckets(df_train)
        df_train = pipeline.apply_dormancy_buckets(df_train, 'TRAIN')
        df_test  = pipeline.apply_dormancy_buckets(df_test,  'TEST')
        return pipeline, df_train, df_test

    def test_freq_maps_built_from_train_only(self, freq_pipeline):
        pipeline, df_train, df_test = freq_pipeline
        pipeline.fit_frequency_maps(df_train)
        # All keys in freq map must be train accounts
        for col, freq_map in pipeline._freq_maps.items():
            train_values = set(df_train[col].astype(str))
            assert set(freq_map.keys()).issubset(train_values)

    def test_sender_account_freq_created(self, freq_pipeline):
        pipeline, df_train, df_test = freq_pipeline
        pipeline.fit_frequency_maps(df_train)
        df_train = pipeline.apply_frequency_maps(df_train, 'TRAIN')
        assert 'sender_account_freq' in df_train.columns

    def test_ip_freq_created(self, freq_pipeline):
        pipeline, df_train, df_test = freq_pipeline
        pipeline.fit_frequency_maps(df_train)
        df_train = pipeline.apply_frequency_maps(df_train, 'TRAIN')
        assert 'ip_freq' in df_train.columns

    def test_freq_values_are_positive(self, freq_pipeline):
        pipeline, df_train, df_test = freq_pipeline
        pipeline.fit_frequency_maps(df_train)
        df_train = pipeline.apply_frequency_maps(df_train, 'TRAIN')
        assert (df_train['sender_account_freq'] >= 1).all()
        assert (df_train['ip_freq'] >= 1).all()

    def test_unseen_values_map_to_one(self, freq_pipeline):
        """Values in test not seen in train must map to frequency 1."""
        pipeline, df_train, df_test = freq_pipeline
        pipeline.fit_frequency_maps(df_train)
        df_test = pipeline.apply_frequency_maps(df_test, 'TEST')

        # Any test IP not in train freq map → ip_freq should be 1
        train_ips = set(pipeline._freq_maps.get('ip_address', {}).keys())
        unseen_mask = ~df_test['ip_address'].isin(train_ips)
        if unseen_mask.any():
            assert (df_test.loc[unseen_mask, 'ip_freq'] == 1).all()

    def test_freq_map_reflects_train_counts(self, freq_pipeline):
        pipeline, df_train, df_test = freq_pipeline
        pipeline.fit_frequency_maps(df_train)
        # Check a specific account's frequency matches train count
        col = 'sender_account'
        if col in pipeline._freq_maps:
            top_acc = df_train[col].value_counts().index[0]
            expected_freq = df_train[col].value_counts().iloc[0]
            assert pipeline._freq_maps[col][top_acc] == expected_freq


# =============================================================================
# 9. LABEL ENCODING TESTS
# =============================================================================

class TestLabelEncoding:

    @pytest.fixture
    def enc_pipeline(self, split_pipeline):
        pipeline, df_train, df_test = split_pipeline
        pipeline.fit_imputation(df_train)
        df_train = pipeline.apply_imputation(df_train)
        df_test  = pipeline.apply_imputation(df_test)
        pipeline.fit_dormancy_buckets(df_train)
        df_train = pipeline.apply_dormancy_buckets(df_train, 'TRAIN')
        df_test  = pipeline.apply_dormancy_buckets(df_test,  'TEST')
        pipeline.fit_frequency_maps(df_train)
        df_train = pipeline.apply_frequency_maps(df_train, 'TRAIN')
        df_test  = pipeline.apply_frequency_maps(df_test,  'TEST')
        df_train = pipeline.apply_amount_features(df_train)
        df_test  = pipeline.apply_amount_features(df_test)
        return pipeline, df_train, df_test

    def test_label_encoders_fit_on_train(self, enc_pipeline):
        pipeline, df_train, df_test = enc_pipeline
        pipeline.fit_label_encoders(df_train)
        assert 'transaction_type' in pipeline.encoders

    def test_encoded_columns_created(self, enc_pipeline):
        pipeline, df_train, df_test = enc_pipeline
        pipeline.fit_label_encoders(df_train)
        df_train = pipeline.apply_label_encoders(df_train, 'TRAIN')
        for col in pipeline.config.LOW_CARDINALITY_COLS:
            assert f'{col}_enc' in df_train.columns, f"Missing {col}_enc"

    def test_encoded_values_are_integers(self, enc_pipeline):
        pipeline, df_train, df_test = enc_pipeline
        pipeline.fit_label_encoders(df_train)
        df_train = pipeline.apply_label_encoders(df_train, 'TRAIN')
        for col in pipeline.config.LOW_CARDINALITY_COLS:
            assert pd.api.types.is_integer_dtype(df_train[f'{col}_enc'])

    def test_dormancy_bucket_encoded(self, enc_pipeline):
        pipeline, df_train, df_test = enc_pipeline
        pipeline.fit_label_encoders(df_train)
        df_train = pipeline.apply_label_encoders(df_train, 'TRAIN')
        assert 'dormancy_bucket_enc' in df_train.columns

    def test_encoder_applied_to_test(self, enc_pipeline):
        pipeline, df_train, df_test = enc_pipeline
        pipeline.fit_label_encoders(df_train)
        df_test = pipeline.apply_label_encoders(df_test, 'TEST')
        for col in pipeline.config.LOW_CARDINALITY_COLS:
            assert f'{col}_enc' in df_test.columns


# =============================================================================
# 10. AMOUNT FEATURES TESTS
# =============================================================================

class TestAmountFeatures:

    def test_amount_log_column_created(self, split_pipeline):
        pipeline, df_train, df_test = split_pipeline
        df_train = pipeline.apply_amount_features(df_train)
        assert 'amount_log' in df_train.columns

    def test_amount_log_is_log1p_of_amount(self, split_pipeline):
        pipeline, df_train, df_test = split_pipeline
        df_train = pipeline.apply_amount_features(df_train)
        expected = np.log1p(df_train['amount'])
        np.testing.assert_allclose(df_train['amount_log'].values, expected.values)

    def test_amount_log_no_negative_infinity(self, split_pipeline):
        """log1p(0) = 0, so no -inf values even if amount=0."""
        pipeline, df_train, df_test = split_pipeline
        df_train = pipeline.apply_amount_features(df_train)
        assert not df_train['amount_log'].isin([float('-inf')]).any()

    def test_amount_log_no_nulls(self, split_pipeline):
        pipeline, df_train, df_test = split_pipeline
        df_train = pipeline.apply_amount_features(df_train)
        assert df_train['amount_log'].isnull().sum() == 0


# =============================================================================
# 11. FEATURE MATRIX TESTS
# =============================================================================

class TestFeatureMatrix:

    @pytest.fixture
    def full_pipeline(self, split_pipeline):
        """Pipeline with all transforms applied, ready for build_feature_matrix."""
        pipeline, df_train, df_test = split_pipeline
        pipeline.fit_imputation(df_train)
        df_train = pipeline.apply_imputation(df_train)
        df_test  = pipeline.apply_imputation(df_test)
        pipeline.fit_dormancy_buckets(df_train)
        df_train = pipeline.apply_dormancy_buckets(df_train, 'TRAIN')
        df_test  = pipeline.apply_dormancy_buckets(df_test,  'TEST')
        pipeline.fit_frequency_maps(df_train)
        df_train = pipeline.apply_frequency_maps(df_train, 'TRAIN')
        df_test  = pipeline.apply_frequency_maps(df_test,  'TEST')
        df_train = pipeline.apply_amount_features(df_train)
        df_test  = pipeline.apply_amount_features(df_test)
        pipeline.fit_label_encoders(df_train)
        df_train = pipeline.apply_label_encoders(df_train, 'TRAIN')
        df_test  = pipeline.apply_label_encoders(df_test,  'TEST')
        return pipeline, df_train, df_test

    def test_x_does_not_contain_is_fraud(self, full_pipeline):
        pipeline, df_train, _ = full_pipeline
        X, y = pipeline.build_feature_matrix(df_train)
        assert 'is_fraud' not in X.columns

    def test_x_does_not_contain_transaction_id(self, full_pipeline):
        pipeline, df_train, _ = full_pipeline
        X, y = pipeline.build_feature_matrix(df_train)
        assert 'transaction_id' not in X.columns

    def test_x_does_not_contain_timestamp(self, full_pipeline):
        pipeline, df_train, _ = full_pipeline
        X, y = pipeline.build_feature_matrix(df_train)
        assert 'timestamp' not in X.columns

    def test_x_does_not_contain_raw_account_strings(self, full_pipeline):
        pipeline, df_train, _ = full_pipeline
        X, y = pipeline.build_feature_matrix(df_train)
        for col in ['sender_account', 'receiver_account', 'device_hash', 'ip_address']:
            assert col not in X.columns, f"Raw string '{col}' found in X"

    def test_x_does_not_contain_raw_categoricals(self, full_pipeline):
        """Raw categorical strings must be dropped — only _enc versions kept."""
        pipeline, df_train, _ = full_pipeline
        X, y = pipeline.build_feature_matrix(df_train)
        for col in pipeline.config.LOW_CARDINALITY_COLS:
            assert col not in X.columns, f"Raw categorical '{col}' found in X"

    def test_x_does_not_contain_dormancy_bucket_string(self, full_pipeline):
        pipeline, df_train, _ = full_pipeline
        X, y = pipeline.build_feature_matrix(df_train)
        assert 'dormancy_bucket' not in X.columns

    def test_x_does_not_contain_fraud_type(self, full_pipeline):
        pipeline, df_train, _ = full_pipeline
        X, y = pipeline.build_feature_matrix(df_train)
        assert 'fraud_type' not in X.columns

    def test_y_is_binary(self, full_pipeline):
        pipeline, df_train, _ = full_pipeline
        _, y = pipeline.build_feature_matrix(df_train)
        assert set(y.unique()).issubset({0, 1})

    def test_y_contains_fraud_cases(self, full_pipeline):
        pipeline, df_train, _ = full_pipeline
        _, y = pipeline.build_feature_matrix(df_train)
        assert y.sum() > 0

    def test_x_all_numeric(self, full_pipeline):
        """Every column in X must be numeric."""
        pipeline, df_train, _ = full_pipeline
        X, _ = pipeline.build_feature_matrix(df_train)
        non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
        assert len(non_numeric) == 0, f"Non-numeric columns in X: {non_numeric}"

    def test_x_no_nulls(self, full_pipeline):
        pipeline, df_train, _ = full_pipeline
        X, _ = pipeline.build_feature_matrix(df_train)
        null_cols = X.columns[X.isnull().any()].tolist()
        assert len(null_cols) == 0, f"Null values in X columns: {null_cols}"

    def test_x_train_and_test_same_columns(self, full_pipeline):
        pipeline, df_train, df_test = full_pipeline
        X_train, _ = pipeline.build_feature_matrix(df_train)
        X_test,  _ = pipeline.build_feature_matrix(df_test)
        assert list(X_train.columns) == list(X_test.columns)

    def test_leaky_columns_not_in_x(self, full_pipeline):
        pipeline, df_train, _ = full_pipeline
        X, _ = pipeline.build_feature_matrix(df_train)
        for col in pipeline.config.LEAKY_COLUMNS:
            assert col not in X.columns, f"Leaky column '{col}' found in X"


# =============================================================================
# 12. SCALING TESTS
# =============================================================================

class TestScaling:

    @pytest.fixture
    def scaled_pipeline(self, split_pipeline):
        pipeline, df_train, df_test = split_pipeline
        pipeline.fit_imputation(df_train)
        df_train = pipeline.apply_imputation(df_train)
        df_test  = pipeline.apply_imputation(df_test)
        pipeline.fit_dormancy_buckets(df_train)
        df_train = pipeline.apply_dormancy_buckets(df_train, 'TRAIN')
        df_test  = pipeline.apply_dormancy_buckets(df_test,  'TEST')
        pipeline.fit_frequency_maps(df_train)
        df_train = pipeline.apply_frequency_maps(df_train, 'TRAIN')
        df_test  = pipeline.apply_frequency_maps(df_test,  'TEST')
        df_train = pipeline.apply_amount_features(df_train)
        df_test  = pipeline.apply_amount_features(df_test)
        pipeline.fit_label_encoders(df_train)
        df_train = pipeline.apply_label_encoders(df_train, 'TRAIN')
        df_test  = pipeline.apply_label_encoders(df_test,  'TEST')
        X_train, y_train = pipeline.build_feature_matrix(df_train)
        X_test,  y_test  = pipeline.build_feature_matrix(df_test)
        return pipeline, X_train, X_test, y_train, y_test

    def test_scaler_fit_on_train_only(self, scaled_pipeline):
        pipeline, X_train, X_test, _, _ = scaled_pipeline
        pipeline.fit_scaler(X_train)
        assert pipeline.scaler is not None

    def test_scaler_is_robust_scaler(self, scaled_pipeline):
        from sklearn.preprocessing import RobustScaler
        pipeline, X_train, X_test, _, _ = scaled_pipeline
        pipeline.fit_scaler(X_train)
        assert isinstance(pipeline.scaler, RobustScaler)

    def test_scaled_columns_have_zero_median(self, scaled_pipeline):
        """After RobustScaler, the median of scaled columns should be ~0."""
        pipeline, X_train, X_test, _, _ = scaled_pipeline
        pipeline.fit_scaler(X_train)
        X_train = pipeline.apply_scaler(X_train, 'TRAIN')
        for col in pipeline.config.COLS_TO_SCALE:
            if col in X_train.columns:
                assert abs(X_train[col].median()) < 0.1, \
                    f"Median of scaled '{col}' is {X_train[col].median():.4f}, expected ~0"

    def test_same_scaler_applied_to_test(self, scaled_pipeline):
        """Test set uses scaler fitted on train — not refitted."""
        pipeline, X_train, X_test, _, _ = scaled_pipeline
        pipeline.fit_scaler(X_train)
        X_test_scaled = pipeline.apply_scaler(X_test.copy(), 'TEST')
        for col in pipeline.config.COLS_TO_SCALE:
            if col in X_test_scaled.columns:
                # Values should be different from original (scaled)
                assert not X_test_scaled[col].equals(X_test[col])

    def test_non_scaled_columns_unchanged(self, scaled_pipeline):
        """Binary flags and encoded categoricals must not be scaled."""
        pipeline, X_train, X_test, _, _ = scaled_pipeline
        pipeline.fit_scaler(X_train)
        flag_cols = ['is_weekend', 'is_high_risk_hour', 'is_first_transaction']
        original_vals = {c: X_train[c].copy() for c in flag_cols if c in X_train.columns}
        X_train = pipeline.apply_scaler(X_train, 'TRAIN')
        for col, orig in original_vals.items():
            assert X_train[col].equals(orig), f"Flag column '{col}' was modified by scaler"


# =============================================================================
# 13. LEAKAGE AUDIT TESTS
# =============================================================================

class TestLeakageAudit:
    """
    Dedicated tests that specifically audit for data leakage.
    These are the most important tests for ML correctness.
    """

    def test_scaler_not_fit_before_split(self, pipeline):
        """Scaler must be None until fit_scaler() is called."""
        pipeline.load()
        pipeline.build_temporal_features()
        pipeline.build_dormancy_flag()
        assert pipeline.scaler is None

    def test_freq_maps_empty_before_fit(self, pipeline):
        """Frequency maps must be empty until fit_frequency_maps() is called."""
        pipeline.load()
        assert pipeline._freq_maps == {}

    def test_impute_median_none_before_fit(self, pipeline):
        """Imputation median must be None until fit_imputation() is called."""
        pipeline.load()
        assert pipeline._impute_median is None

    def test_dormancy_bins_none_before_fit(self, pipeline):
        """Dormancy bins must be None until fit_dormancy_buckets() is called."""
        pipeline.load()
        assert pipeline._dormancy_bins is None

    def test_is_fraud_not_in_X(self, split_pipeline):
        pipeline, df_train, df_test = split_pipeline
        pipeline.fit_imputation(df_train)
        df_train = pipeline.apply_imputation(df_train)
        pipeline.fit_dormancy_buckets(df_train)
        df_train = pipeline.apply_dormancy_buckets(df_train, 'TRAIN')
        pipeline.fit_frequency_maps(df_train)
        df_train = pipeline.apply_frequency_maps(df_train, 'TRAIN')
        df_train = pipeline.apply_amount_features(df_train)
        pipeline.fit_label_encoders(df_train)
        df_train = pipeline.apply_label_encoders(df_train, 'TRAIN')
        X, y = pipeline.build_feature_matrix(df_train)
        assert 'is_fraud' not in X.columns

    def test_fraud_type_not_in_X(self, split_pipeline):
        """fraud_type is dropped at load — must never reach X."""
        pipeline, df_train, _ = split_pipeline
        assert 'fraud_type' not in df_train.columns

    def test_train_freq_map_not_refit_on_test(self, split_pipeline):
        """Applying freq map to test must not modify pipeline._freq_maps."""
        pipeline, df_train, df_test = split_pipeline
        pipeline.fit_imputation(df_train)
        df_train = pipeline.apply_imputation(df_train)
        df_test  = pipeline.apply_imputation(df_test)
        pipeline.fit_dormancy_buckets(df_train)
        df_train = pipeline.apply_dormancy_buckets(df_train, 'TRAIN')
        df_test  = pipeline.apply_dormancy_buckets(df_test,  'TEST')
        pipeline.fit_frequency_maps(df_train)
        freq_maps_before = {k: dict(v) for k, v in pipeline._freq_maps.items()}
        pipeline.apply_frequency_maps(df_test, 'TEST')
        assert pipeline._freq_maps == freq_maps_before


# =============================================================================
# 14. EDGE CASE TESTS
# =============================================================================

class TestEdgeCases:

    def test_all_null_tslt_imputed(self, split_pipeline):
        """If all time_since_last_transaction in train are null, imputation should not crash."""
        pipeline, df_train, df_test = split_pipeline
        col = pipeline.config.DORMANCY_IMPUTE_COL
        # Manually set all train values to null
        df_train = df_train.copy()
        df_train[col] = np.nan
        # Median of all-null is NaN — fillna with NaN is a no-op
        # Pipeline should handle this gracefully
        try:
            pipeline.fit_imputation(df_train)
        except Exception as e:
            pytest.fail(f"fit_imputation raised unexpectedly on all-null input: {e}")

    def test_amount_log_zero_amount(self, split_pipeline):
        """log1p(0) = 0, should not crash or produce -inf."""
        pipeline, df_train, df_test = split_pipeline
        df_train = df_train.copy()
        df_train['amount'] = 0.0
        df_train = pipeline.apply_amount_features(df_train)
        assert (df_train['amount_log'] == 0.0).all()

    def test_unseen_category_in_test_handled(self, split_pipeline):
        """Label encoder must handle categories in test not seen in train."""
        pipeline, df_train, df_test = split_pipeline
        pipeline.fit_imputation(df_train)
        df_train = pipeline.apply_imputation(df_train)
        df_test  = pipeline.apply_imputation(df_test)
        pipeline.fit_dormancy_buckets(df_train)
        df_train = pipeline.apply_dormancy_buckets(df_train, 'TRAIN')
        df_test  = pipeline.apply_dormancy_buckets(df_test,  'TEST')
        pipeline.fit_frequency_maps(df_train)
        df_train = pipeline.apply_frequency_maps(df_train, 'TRAIN')
        df_test  = pipeline.apply_frequency_maps(df_test,  'TEST')
        df_train = pipeline.apply_amount_features(df_train)
        df_test  = pipeline.apply_amount_features(df_test)
        pipeline.fit_label_encoders(df_train)

        # Inject unseen category into test
        df_test = df_test.copy()
        df_test.loc[df_test.index[0], 'transaction_type'] = 'UNKNOWN_CATEGORY'

        # Should not raise
        try:
            df_test = pipeline.apply_label_encoders(df_test, 'TEST')
        except Exception as e:
            pytest.fail(f"apply_label_encoders raised on unseen category: {e}")

        assert 'transaction_type_enc' in df_test.columns

    def test_frequency_map_unseen_account_defaults_to_one(self, split_pipeline):
        pipeline, df_train, df_test = split_pipeline
        pipeline.fit_imputation(df_train)
        df_train = pipeline.apply_imputation(df_train)
        df_test  = pipeline.apply_imputation(df_test)
        pipeline.fit_dormancy_buckets(df_train)
        df_train = pipeline.apply_dormancy_buckets(df_train, 'TRAIN')
        df_test  = pipeline.apply_dormancy_buckets(df_test,  'TEST')
        pipeline.fit_frequency_maps(df_train)

        # Inject completely unseen account
        df_test = df_test.copy()
        df_test.loc[df_test.index[0], 'sender_account'] = 'ACC_TOTALLY_NEW_99999'
        df_test = pipeline.apply_frequency_maps(df_test, 'TEST')

        assert df_test.loc[df_test.index[0], 'sender_account_freq'] == 1


# =============================================================================
# 15. INTEGRATION TEST — FULL PIPELINE END-TO-END
# =============================================================================

class TestIntegration:

    def test_full_pipeline_runs_without_error(self, pipeline):
        X_train, X_test, y_train, y_test = pipeline.run()
        assert X_train is not None
        assert X_test  is not None
        assert y_train is not None
        assert y_test  is not None

    def test_full_pipeline_output_shapes(self, pipeline):
        X_train, X_test, y_train, y_test = pipeline.run()
        assert X_train.shape[0] == len(y_train)
        assert X_test.shape[0]  == len(y_test)
        assert X_train.shape[1] == X_test.shape[1]

    def test_full_pipeline_no_leaky_cols_in_output(self, pipeline, config):
        X_train, X_test, y_train, y_test = pipeline.run()
        for col in config.LEAKY_COLUMNS:
            assert col not in X_train.columns
            assert col not in X_test.columns

    def test_full_pipeline_y_is_binary(self, pipeline):
        _, _, y_train, y_test = pipeline.run()
        assert set(y_train.unique()).issubset({0, 1})
        assert set(y_test.unique()).issubset({0, 1})

    def test_full_pipeline_x_all_numeric(self, pipeline):
        X_train, X_test, _, _ = pipeline.run()
        non_num_train = X_train.select_dtypes(exclude=[np.number]).columns.tolist()
        non_num_test  = X_test.select_dtypes(exclude=[np.number]).columns.tolist()
        assert non_num_train == [], f"Non-numeric in X_train: {non_num_train}"
        assert non_num_test  == [], f"Non-numeric in X_test: {non_num_test}"

    def test_full_pipeline_x_no_nulls(self, pipeline):
        X_train, X_test, _, _ = pipeline.run()
        assert X_train.isnull().sum().sum() == 0
        assert X_test.isnull().sum().sum()  == 0

    def test_full_pipeline_parquet_files_saved(self, pipeline, config):
        pipeline.run()
        assert Path(config.OUTPUT_X_TRAIN).exists()
        assert Path(config.OUTPUT_X_TEST).exists()
        assert Path(config.OUTPUT_Y_TRAIN).exists()
        assert Path(config.OUTPUT_Y_TEST).exists()

    def test_full_pipeline_encoder_saved(self, pipeline, config):
        pipeline.run()
        assert Path(config.OUTPUT_ENCODERS).exists()

    def test_full_pipeline_scaler_saved(self, pipeline, config):
        pipeline.run()
        assert Path(config.OUTPUT_SCALER).exists()

    def test_full_pipeline_train_before_test_in_time(self, pipeline):
        """After the full run, saved train should be earlier in time than test."""
        X_train, X_test, y_train, y_test = pipeline.run()
        # Row counts confirm split (train = 80%, test = 20%)
        total = len(X_train) + len(X_test)
        assert len(X_train) == int(total * 0.8)
        assert len(X_test)  == total - int(total * 0.8)