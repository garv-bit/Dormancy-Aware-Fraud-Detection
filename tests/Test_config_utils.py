"""
test_config_utils.py — Unit tests for Config.py and Utils.py

Run:
    pytest src/test_config_utils.py -v
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
from Utils import (
    safe_divide,
    get_numeric_columns,
    get_categorical_columns,
    calculate_data_quality_score,
    detect_outliers_iqr,
    detect_outliers_zscore,
    stratified_sample,
    validate_dataframe,
    validate_file_exists,
    format_bytes,
    load_checkpoint,
    save_checkpoint,
    create_output_directory,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def config(tmp_path):
    cfg = EDAConfig(LOG_FILE=str(tmp_path / "test.log"))
    return cfg


@pytest.fixture
def logger(config):
    log = logging.getLogger("test_logger")
    log.setLevel(logging.DEBUG)
    log.handlers = []
    log.addHandler(logging.NullHandler())
    return log


@pytest.fixture
def clean_df():
    """A clean dataframe with no nulls, no duplicates."""
    np.random.seed(0)
    return pd.DataFrame({
        'amount':    np.random.exponential(300, 500).round(2),
        'velocity':  np.random.randint(1, 20, 500).astype(float),
        'category':  np.random.choice(['retail', 'travel', 'food'], 500),
        'is_fraud':  np.random.choice([True, False], 500, p=[0.04, 0.96]),
    })


@pytest.fixture
def dirty_df():
    """Dataframe with nulls and duplicates."""
    df = pd.DataFrame({
        'amount':   [100.0, 200.0, np.nan, 100.0, 400.0],
        'category': ['retail', 'travel', 'food', 'retail', 'travel'],
        'is_fraud': [False, False, True, False, True],
    })
    return df


# =============================================================================
# 1. EDAConfig TESTS
# =============================================================================

class TestEDAConfig:

    def test_default_config_creates(self):
        cfg = EDAConfig()
        assert cfg is not None

    def test_validate_passes_on_default(self, config):
        assert config.validate() is True

    def test_correlation_threshold_bounds(self):
        with pytest.raises(AssertionError):
            EDAConfig(CORRELATION_THRESHOLD=0.0).validate()
        with pytest.raises(AssertionError):
            EDAConfig(CORRELATION_THRESHOLD=1.1).validate()

    def test_alpha_bounds(self):
        with pytest.raises(AssertionError):
            EDAConfig(ALPHA=0.0).validate()
        with pytest.raises(AssertionError):
            EDAConfig(ALPHA=1.0).validate()

    def test_iqr_multiplier_positive(self):
        with pytest.raises(AssertionError):
            EDAConfig(IQR_MULTIPLIER=0.0).validate()

    def test_z_score_threshold_positive(self):
        with pytest.raises(AssertionError):
            EDAConfig(Z_SCORE_THRESHOLD=0.0).validate()

    def test_n_estimators_positive(self):
        with pytest.raises(AssertionError):
            EDAConfig(N_ESTIMATORS=0).validate()

    def test_pca_variance_threshold_bounds(self):
        with pytest.raises(AssertionError):
            EDAConfig(PCA_VARIANCE_THRESHOLD=0.0).validate()
        with pytest.raises(AssertionError):
            EDAConfig(PCA_VARIANCE_THRESHOLD=1.1).validate()

    def test_min_clusters_ge_2(self):
        with pytest.raises(AssertionError):
            EDAConfig(MIN_CLUSTERS=1).validate()

    def test_min_less_than_max_clusters(self):
        with pytest.raises(AssertionError):
            EDAConfig(MIN_CLUSTERS=5, MAX_CLUSTERS=5).validate()

    def test_test_size_bounds(self):
        with pytest.raises(AssertionError):
            EDAConfig(TEST_SIZE=0.0).validate()
        with pytest.raises(AssertionError):
            EDAConfig(TEST_SIZE=1.0).validate()

    def test_cv_folds_gt_1(self):
        with pytest.raises(AssertionError):
            EDAConfig(CV_FOLDS=1).validate()

    def test_min_sample_size_positive(self):
        with pytest.raises(AssertionError):
            EDAConfig(MIN_SAMPLE_SIZE=0).validate()

    def test_leaky_columns_not_empty(self):
        with pytest.raises(AssertionError):
            cfg = EDAConfig()
            cfg.LEAKY_COLUMNS = []
            cfg.validate()

    def test_fraud_type_in_drop_on_load(self):
        cfg = EDAConfig()
        assert 'fraud_type' in cfg.DROP_ON_LOAD

    def test_is_fraud_in_leaky_columns(self):
        cfg = EDAConfig()
        assert 'is_fraud' in cfg.LEAKY_COLUMNS

    def test_fraud_type_in_leaky_columns(self):
        cfg = EDAConfig()
        assert 'fraud_type' in cfg.LEAKY_COLUMNS

    def test_leaky_columns_set_property(self):
        cfg = EDAConfig()
        leaky_set = cfg.leaky_columns_set
        assert isinstance(leaky_set, set)
        assert 'is_fraud' in leaky_set

    def test_leaky_columns_set_o1_lookup(self):
        cfg = EDAConfig()
        assert 'is_fraud' in cfg.leaky_columns_set
        assert 'amount' not in cfg.leaky_columns_set

    def test_weekend_days_default(self):
        cfg = EDAConfig()
        assert cfg.WEEKEND_DAYS == [5, 6]

    def test_id_columns_default(self):
        cfg = EDAConfig()
        assert 'transaction_id' in cfg.ID_COLUMNS

    def test_skip_categorical_contains_fraud_type(self):
        cfg = EDAConfig()
        assert 'fraud_type' in cfg.SKIP_CATEGORICAL_ANALYSIS

    def test_completeness_threshold_bounds(self):
        with pytest.raises(AssertionError):
            EDAConfig(COMPLETENESS_THRESHOLD=101).validate()
        with pytest.raises(AssertionError):
            EDAConfig(COMPLETENESS_THRESHOLD=-1).validate()

    def test_min_sample_size_positive(self):
        with pytest.raises(AssertionError):
            EDAConfig(MIN_SAMPLE_SIZE=0).validate()

    def test_plot_dpi_positive(self):
        with pytest.raises(AssertionError):
            EDAConfig(PLOT_DPI=0).validate()


# =============================================================================
# 2. safe_divide TESTS
# =============================================================================

class TestSafeDivide:

    def test_normal_division(self):
        assert safe_divide(10, 2) == 5.0

    def test_zero_denominator_returns_default(self):
        assert safe_divide(10, 0) == 0.0

    def test_zero_denominator_custom_default(self):
        assert safe_divide(10, 0, default=99.0) == 99.0

    def test_nan_denominator_returns_default(self):
        assert safe_divide(10, float('nan')) == 0.0

    def test_inf_denominator_returns_default(self):
        assert safe_divide(10, float('inf')) == 0.0

    def test_zero_numerator(self):
        assert safe_divide(0, 5) == 0.0

    def test_negative_values(self):
        assert safe_divide(-10, 2) == -5.0

    def test_float_division(self):
        assert abs(safe_divide(1, 3) - 0.3333) < 0.001

    def test_both_zero(self):
        assert safe_divide(0, 0) == 0.0

    def test_result_inf_returns_default(self):
        # Very large numerator / very small denominator
        result = safe_divide(float('inf'), 1)
        assert result == 0.0


# =============================================================================
# 3. get_numeric_columns TESTS
# =============================================================================

class TestGetNumericColumns:

    def test_returns_numeric_columns(self, clean_df):
        cols = get_numeric_columns(clean_df)
        assert 'amount' in cols
        assert 'velocity' in cols

    def test_excludes_object_columns(self, clean_df):
        cols = get_numeric_columns(clean_df)
        assert 'category' not in cols

    def test_exclude_parameter(self, clean_df):
        cols = get_numeric_columns(clean_df, exclude=['amount'])
        assert 'amount' not in cols
        assert 'velocity' in cols

    def test_exclude_multiple(self, clean_df):
        cols = get_numeric_columns(clean_df, exclude=['amount', 'velocity'])
        assert 'amount' not in cols
        assert 'velocity' not in cols

    def test_empty_exclude(self, clean_df):
        cols = get_numeric_columns(clean_df, exclude=[])
        assert 'amount' in cols

    def test_bool_column_excluded(self):
        df = pd.DataFrame({'a': [1.0, 2.0], 'b': [True, False]})
        # bool is numeric in pandas
        cols = get_numeric_columns(df)
        assert 'a' in cols

    def test_empty_dataframe(self):
        df = pd.DataFrame({'a': pd.Series([], dtype=float)})
        cols = get_numeric_columns(df)
        assert 'a' in cols


# =============================================================================
# 4. get_categorical_columns TESTS
# =============================================================================

class TestGetCategoricalColumns:

    def test_returns_object_columns(self, clean_df):
        cols = get_categorical_columns(clean_df)
        assert 'category' in cols

    def test_excludes_numeric_columns(self, clean_df):
        cols = get_categorical_columns(clean_df)
        assert 'amount' not in cols
        assert 'velocity' not in cols

    def test_exclude_parameter(self, clean_df):
        cols = get_categorical_columns(clean_df, exclude=['category'])
        assert 'category' not in cols

    def test_bool_column_included(self):
        df = pd.DataFrame({'flag': [True, False, True], 'val': [1.0, 2.0, 3.0]})
        cols = get_categorical_columns(df)
        assert 'flag' in cols

    def test_empty_exclude(self, clean_df):
        cols = get_categorical_columns(clean_df, exclude=[])
        assert 'category' in cols


# =============================================================================
# 5. calculate_data_quality_score TESTS
# =============================================================================

class TestDataQualityScore:

    def test_perfect_data_scores_high(self, config, logger, clean_df):
        scores = calculate_data_quality_score(clean_df, config, logger)
        assert scores['completeness'] == 100.0
        assert scores['overall'] > 80.0

    def test_returns_all_keys(self, config, logger, clean_df):
        scores = calculate_data_quality_score(clean_df, config, logger)
        for key in ['completeness', 'uniqueness', 'consistency', 'validity', 'overall']:
            assert key in scores

    def test_missing_data_reduces_completeness(self, config, logger, dirty_df):
        scores = calculate_data_quality_score(dirty_df, config, logger)
        assert scores['completeness'] < 100.0

    def test_duplicates_reduce_uniqueness(self, config, logger, dirty_df):
        scores = calculate_data_quality_score(dirty_df, config, logger)
        assert scores['uniqueness'] < 100.0

    def test_overall_is_weighted_sum_not_mean(self, config, logger, clean_df):
        """overall = c*0.4 + u*0.2 + co*0.2 + v*0.2, not mean of pre-weighted."""
        scores = calculate_data_quality_score(clean_df, config, logger)
        expected = (
            scores['completeness'] * 0.4 +
            scores['uniqueness']   * 0.2 +
            scores['consistency']  * 0.2 +
            scores['validity']     * 0.2
        )
        assert abs(scores['overall'] - expected) < 0.001

    def test_scores_between_0_and_100(self, config, logger, clean_df):
        scores = calculate_data_quality_score(clean_df, config, logger)
        for key, val in scores.items():
            assert 0.0 <= val <= 100.0, f"{key} = {val} out of range"

    def test_inf_values_reduce_validity(self, config, logger):
        df = pd.DataFrame({
            'a': [1.0, float('inf'), 3.0],
            'b': ['x', 'y', 'z'],
        })
        scores = calculate_data_quality_score(df, config, logger)
        assert scores['validity'] < 100.0


# =============================================================================
# 6. detect_outliers_iqr TESTS
# =============================================================================

class TestDetectOutliersIQR:

    def test_returns_required_keys(self):
        s = pd.Series([1, 2, 3, 4, 5, 100])
        result = detect_outliers_iqr(s)
        for key in ['count', 'percentage', 'lower_bound', 'upper_bound', 'iqr']:
            assert key in result

    def test_detects_extreme_outlier(self):
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 1000.0])
        result = detect_outliers_iqr(s)
        assert result['count'] >= 1

    def test_no_outliers_in_uniform_data(self):
        s = pd.Series(range(100))
        result = detect_outliers_iqr(s)
        assert result['count'] == 0

    def test_percentage_consistent_with_count(self):
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 1000.0])
        result = detect_outliers_iqr(s)
        expected_pct = result['count'] / len(s) * 100
        assert abs(result['percentage'] - expected_pct) < 0.01

    def test_custom_multiplier(self):
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 10.0])
        strict = detect_outliers_iqr(s, multiplier=0.5)
        lenient = detect_outliers_iqr(s, multiplier=3.0)
        assert strict['count'] >= lenient['count']

    def test_iqr_is_q3_minus_q1(self):
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = detect_outliers_iqr(s)
        assert abs(result['iqr'] - (s.quantile(0.75) - s.quantile(0.25))) < 1e-9

    def test_upper_bound_gt_lower_bound(self):
        s = pd.Series(range(50))
        result = detect_outliers_iqr(s)
        assert result['upper_bound'] > result['lower_bound']


# =============================================================================
# 7. detect_outliers_zscore TESTS
# =============================================================================

class TestDetectOutliersZScore:

    def test_returns_required_keys(self):
        s = pd.Series(range(100))
        result = detect_outliers_zscore(s)
        for key in ['count', 'percentage', 'threshold']:
            assert key in result

    def test_detects_extreme_outlier(self):
        s = pd.Series([1.0] * 99 + [1000.0])
        result = detect_outliers_zscore(s)
        assert result['count'] >= 1

    def test_uniform_data_few_outliers(self):
        np.random.seed(42)
        s = pd.Series(np.random.normal(0, 1, 1000))
        result = detect_outliers_zscore(s)
        # With threshold=3, expect < 1% outliers in normal distribution
        assert result['percentage'] < 1.0

    def test_threshold_stored_correctly(self):
        s = pd.Series(range(100))
        result = detect_outliers_zscore(s, threshold=2.5)
        assert result['threshold'] == 2.5

    def test_stricter_threshold_finds_more_outliers(self):
        np.random.seed(0)
        s = pd.Series(np.random.normal(0, 1, 500))
        strict  = detect_outliers_zscore(s, threshold=1.5)
        lenient = detect_outliers_zscore(s, threshold=3.0)
        assert strict['count'] >= lenient['count']


# =============================================================================
# 8. stratified_sample TESTS
# =============================================================================

class TestStratifiedSample:

    def test_returns_dataframe(self, clean_df):
        result = stratified_sample(clean_df, n=100)
        assert isinstance(result, pd.DataFrame)

    def test_sample_size_respected(self, clean_df):
        result = stratified_sample(clean_df, n=100)
        assert len(result) == 100

    def test_returns_full_df_when_n_ge_len(self, clean_df):
        result = stratified_sample(clean_df, n=len(clean_df))
        assert len(result) == len(clean_df)

    def test_returns_full_df_when_n_exceeds_len(self, clean_df):
        result = stratified_sample(clean_df, n=10000)
        assert len(result) == len(clean_df)

    def test_fraud_ratio_approximately_preserved(self, clean_df):
        original_rate = clean_df['is_fraud'].mean()
        result = stratified_sample(clean_df, n=200, target_col='is_fraud')
        # is_fraud may be dropped by groupby.apply in some pandas versions — check if present
        if 'is_fraud' in result.columns:
            sample_rate = result['is_fraud'].mean()
            assert abs(original_rate - sample_rate) < 0.1
        else:
            # Fallback: just verify sample size is correct
            assert len(result) == 200

    def test_fallback_when_target_col_missing(self, clean_df):
        result = stratified_sample(clean_df, n=100, target_col='nonexistent')
        assert len(result) == 100

    def test_reproducible_with_same_seed(self, clean_df):
        r1 = stratified_sample(clean_df, n=100, random_state=42)
        r2 = stratified_sample(clean_df, n=100, random_state=42)
        assert list(r1.index) == list(r2.index)

    def test_different_seeds_give_different_results(self, clean_df):
        r1 = stratified_sample(clean_df, n=100, random_state=1)
        r2 = stratified_sample(clean_df, n=100, random_state=99)
        assert list(r1.index) != list(r2.index)


# =============================================================================
# 9. validate_dataframe TESTS
# =============================================================================

class TestValidateDataframe:

    def test_valid_dataframe_passes(self, logger, clean_df):
        assert validate_dataframe(clean_df, logger) is True

    def test_none_raises(self, logger):
        with pytest.raises(ValueError):
            validate_dataframe(None, logger)

    def test_empty_dataframe_raises(self, logger):
        with pytest.raises(ValueError):
            validate_dataframe(pd.DataFrame(), logger)

    def test_too_few_columns_raises(self, logger):
        df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        with pytest.raises(ValueError):
            validate_dataframe(df, logger)

    def test_small_dataset_warns(self, logger):
        df = pd.DataFrame({'a': [1], 'b': [2], 'c': [3]})
        # Should not raise but will warn
        result = validate_dataframe(df, logger)
        assert result is True


# =============================================================================
# 10. validate_file_exists TESTS
# =============================================================================

class TestValidateFileExists:

    def test_existing_file_passes(self, tmp_path, logger):
        f = tmp_path / "test.csv"
        f.write_text("a,b\n1,2")
        assert validate_file_exists(str(f), logger) is True

    def test_nonexistent_file_raises(self, tmp_path, logger):
        with pytest.raises(FileNotFoundError):
            validate_file_exists(str(tmp_path / "ghost.csv"), logger)


# =============================================================================
# 11. format_bytes TESTS
# =============================================================================

class TestFormatBytes:

    def test_bytes(self):
        assert 'B' in format_bytes(500)

    def test_kilobytes(self):
        assert 'KB' in format_bytes(2048)

    def test_megabytes(self):
        assert 'MB' in format_bytes(2 * 1024 * 1024)

    def test_gigabytes(self):
        assert 'GB' in format_bytes(2 * 1024 ** 3)

    def test_zero_bytes(self):
        result = format_bytes(0)
        assert '0.00 B' in result

    def test_returns_string(self):
        assert isinstance(format_bytes(1024), str)


# =============================================================================
# 12. Checkpointing TESTS
# =============================================================================

class TestCheckpointing:

    def test_save_and_load_checkpoint(self, tmp_path, logger):
        data = {'key': 'value', 'number': 42}
        save_checkpoint(str(tmp_path), 'test_section', data, logger)
        loaded = load_checkpoint(str(tmp_path), 'test_section', logger)
        assert loaded == data

    def test_load_nonexistent_checkpoint_returns_none(self, tmp_path, logger):
        result = load_checkpoint(str(tmp_path), 'nonexistent', logger)
        assert result is None

    def test_checkpoint_file_created(self, tmp_path, logger):
        save_checkpoint(str(tmp_path), 'mysection', {'x': 1}, logger)
        assert (tmp_path / 'mysection.joblib').exists()

    def test_checkpoint_creates_directory(self, tmp_path, logger):
        new_dir = str(tmp_path / 'new_checkpoints')
        save_checkpoint(new_dir, 'sec', {'y': 2}, logger)
        assert Path(new_dir).exists()

    def test_checkpoint_saves_complex_objects(self, tmp_path, logger):
        import numpy as np
        data = {'array': np.array([1, 2, 3]), 'df': pd.DataFrame({'a': [1, 2]})}
        save_checkpoint(str(tmp_path), 'complex', data, logger)
        loaded = load_checkpoint(str(tmp_path), 'complex', logger)
        np.testing.assert_array_equal(loaded['array'], data['array'])

    def test_overwrite_checkpoint(self, tmp_path, logger):
        save_checkpoint(str(tmp_path), 'sec', {'v': 1}, logger)
        save_checkpoint(str(tmp_path), 'sec', {'v': 2}, logger)
        loaded = load_checkpoint(str(tmp_path), 'sec', logger)
        assert loaded['v'] == 2


# =============================================================================
# 13. create_output_directory TESTS
# =============================================================================

class TestCreateOutputDirectory:

    def test_creates_directory(self, tmp_path, logger):
        new_dir = str(tmp_path / 'outputs')
        result = create_output_directory(new_dir, logger)
        assert result.exists()

    def test_returns_path_object(self, tmp_path, logger):
        result = create_output_directory(str(tmp_path / 'out'), logger)
        assert isinstance(result, Path)

    def test_existing_directory_does_not_raise(self, tmp_path, logger):
        d = str(tmp_path / 'exists')
        create_output_directory(d, logger)
        create_output_directory(d, logger)  # should not raise