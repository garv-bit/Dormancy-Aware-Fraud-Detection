"""
Production-grade Financial Fraud Detection EDA Pipeline
Modular, testable, and professionally structured.

Leakage safeguards
------------------
PRIMARY:   fraud_type (and any col in config.DROP_ON_LOAD) dropped on CSV load.
SECONDARY: config.LEAKY_COLUMNS enforced in every feature matrix build.

Checkpointing
-------------
Each section saves its result to config.CHECKPOINT_DIR/<section>.joblib
after completion.  On re-run, completed sections are restored and skipped.
Delete the checkpoint folder to force a full re-run.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
import warnings
from scipy import stats
from scipy.stats import chi2_contingency, normaltest
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score
import joblib

from Config import EDAConfig
from Utils import (
    setup_logging,
    validate_dataframe,
    validate_file_exists,
    safe_divide,
    get_numeric_columns,
    get_categorical_columns,
    calculate_data_quality_score,
    detect_outliers_iqr,
    detect_outliers_zscore,
    stratified_sample,
    load_checkpoint,
    save_checkpoint,
    print_section_header,
    create_output_directory,
    format_bytes,
)

warnings.filterwarnings("ignore")


class FraudEDA:
    """
    Comprehensive EDA pipeline for financial fraud detection datasets.

    Checkpointing: each section saves to eda_checkpoints/<section>.joblib.
    Memory:        RF, GB, clustering run on a stratified sample (SAMPLE_SIZE rows).
    Leakage:       fraud_type dropped on load; LEAKY_COLUMNS excluded everywhere.
    """

    def __init__(self, config: Optional[EDAConfig] = None):
        self.config = config or EDAConfig()
        self.config.validate()
        self.logger = setup_logging(self.config)
        self.df     = None
        self.temporal_feature_cols: List[str] = []

        self.results = {
            'metadata':            {},
            'data_quality':        {},
            'numerical_analysis':  {},
            'categorical_analysis':{},
            'temporal_analysis':   {},
            'correlation_analysis':{},
            'feature_importance':  {},
            'pca_analysis':        {},
            'clustering_analysis': {},
            'statistical_tests':   {},
            'model_readiness':     {},
            'recommendations':     [],
            'visualizations':      {},
        }

        Path(self.config.CHECKPOINT_DIR).mkdir(exist_ok=True)
        plt.style.use(self.config.PLOT_STYLE)
        sns.set_palette(self.config.COLOR_PALETTE)
        self.logger.info("FraudEDA pipeline initialized")

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _leaky(self):
        return self.config.leaky_columns_set

    def _ckpt_load(self, section: str):
        return load_checkpoint(self.config.CHECKPOINT_DIR, section, self.logger)

    def _ckpt_save(self, section: str, data: Any):
        save_checkpoint(self.config.CHECKPOINT_DIR, section, data, self.logger)

    def _get_sample(self) -> pd.DataFrame:
        """Stratified sample for expensive ML steps."""
        sample = stratified_sample(
            self.df,
            n=self.config.SAMPLE_SIZE,
            target_col="is_fraud",
            random_state=self.config.RANDOM_STATE,
        )
        if len(sample) < len(self.df):
            self.logger.info(
                f"[SAMPLE] Using {len(sample):,} / {len(self.df):,} rows for ML steps"
            )
            print(
                f"\n   📉 Sampling {len(sample):,} rows for ML steps "
                f"(stratified — fraud ratio preserved)"
            )
        return sample

    # -------------------------------------------------------------------------
    # 1. Load Data
    # -------------------------------------------------------------------------

    def load_data(self, filepath: Optional[str] = None) -> pd.DataFrame:
        """
        Load dataset from CSV with validation and leakage column removal.

        Columns in config.DROP_ON_LOAD are removed immediately after reading
        so they cannot enter any downstream analysis step.
        """
        filepath = filepath or self.config.INPUT_CSV

        try:
            validate_file_exists(filepath, self.logger)
            self.logger.info(f"Loading data from: {filepath}")

            self.df = pd.read_csv(filepath)
            validate_dataframe(self.df, self.logger)

            # PRIMARY LEAKAGE SAFEGUARD
            cols_to_drop = [c for c in self.config.DROP_ON_LOAD if c in self.df.columns]
            if cols_to_drop:
                self.df = self.df.drop(columns=cols_to_drop)
                self.logger.info(
                    f"[LEAKAGE GUARD] Dropped post-event columns on load: {cols_to_drop}"
                )
                print(f"\n⚠️  Dropped post-event columns on load: {cols_to_drop}")

            self.logger.info(
                f"Data loaded: {len(self.df):,} rows × {len(self.df.columns)} cols"
            )
            return self.df

        except FileNotFoundError:
            self.logger.error(f"File not found: {filepath}")
            raise
        except pd.errors.EmptyDataError:
            raise ValueError(f"CSV file is empty: {filepath}")
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise

    # -------------------------------------------------------------------------
    # 2. Metadata
    # -------------------------------------------------------------------------

    def analyze_metadata(self) -> Dict[str, Any]:
        print_section_header("1. DATASET METADATA & PROFILING", logger=self.logger)

        cached = self._ckpt_load("metadata")
        if cached is not None:
            self.results['metadata'] = cached
            return cached

        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")

        metadata = {
            'total_rows':    len(self.df),
            'total_columns': len(self.df.columns),
            'memory_bytes':  int(self.df.memory_usage(deep=True).sum()),
            'memory_mb':     float(self.df.memory_usage(deep=True).sum() / (1024 ** 2)),
            'bytes_per_row': float(self.df.memory_usage(deep=True).sum() / len(self.df)),
            'column_info':   {},
            'dtypes_summary': {str(k): int(v)
                               for k, v in self.df.dtypes.value_counts().items()},
        }

        for col in self.df.columns:
            metadata['column_info'][col] = {
                'dtype':      str(self.df[col].dtype),
                'non_null':   int(self.df[col].count()),
                'null_count': int(self.df[col].isnull().sum()),
                'null_pct':   float(
                    safe_divide(self.df[col].isnull().sum(), len(self.df), 0) * 100
                ),
                'unique':     int(self.df[col].nunique()),
                'unique_pct': float(
                    safe_divide(self.df[col].nunique(), len(self.df), 0) * 100
                ),
            }

        print(f"\n📊 Dataset Overview:")
        print(f"   Rows     : {metadata['total_rows']:,}")
        print(f"   Columns  : {metadata['total_columns']}")
        print(f"   Memory   : {format_bytes(metadata['memory_bytes'])}")
        print(f"   Bytes/Row: {metadata['bytes_per_row']:.0f}")

        self.results['metadata'] = metadata
        self._ckpt_save("metadata", metadata)
        self.logger.info("Metadata analysis complete")
        return metadata

    # -------------------------------------------------------------------------
    # 3. Data Quality
    # -------------------------------------------------------------------------

    def analyze_data_quality(self) -> Dict[str, Any]:
        print_section_header("2. DATA QUALITY ASSESSMENT", logger=self.logger)

        cached = self._ckpt_load("data_quality")
        if cached is not None:
            self.results['data_quality'] = cached
            return cached

        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")

        # Uses corrected weighted-sum formula (see Utils.calculate_data_quality_score)
        quality_metrics = calculate_data_quality_score(self.df, self.config, self.logger)
        total_missing   = self.df.isnull().sum().sum()
        duplicates      = self.df.duplicated().sum()

        data_quality = {
            'total_missing':    int(total_missing),
            'completeness_pct': float(quality_metrics['completeness']),
            'duplicate_rows':   int(duplicates),
            'duplicate_pct':    float(safe_divide(duplicates, len(self.df), 0) * 100),
            'quality_metrics':  quality_metrics,
            'overall_quality':  float(quality_metrics['overall']),
            'missing_by_column': {k: int(v) for k, v in self.df.isnull().sum().items()},
        }

        print(f"\n✅ Quality Metrics:")
        for k in ['completeness', 'uniqueness', 'consistency', 'validity', 'overall']:
            print(f"   {k.capitalize():<14}: {quality_metrics[k]:.2f}%")
        print(f"\n   Duplicates : {duplicates} ({data_quality['duplicate_pct']:.2f}%)")

        # Soft-leak audit: time_since_last_transaction null rate vs fraud label
        if 'time_since_last_transaction' in self.df.columns and 'is_fraud' in self.df.columns:
            # Store as plain dict immediately to avoid calling .values() on a numpy array
            null_by_fraud_dict = (
                self.df.groupby('is_fraud')['time_since_last_transaction']
                .apply(lambda x: round(x.isnull().mean() * 100, 2))
                .to_dict()
            )
            data_quality['time_since_null_by_fraud'] = null_by_fraud_dict

            print("\n  time_since_last_transaction null rate by label:")
            for label, pct in null_by_fraud_dict.items():
                tag = "fraud" if label else "non-fraud"
                print(f"   {tag}: {pct:.2f}% missing")

            vals = list(null_by_fraud_dict.values())
            if len(vals) == 2 and abs(vals[0] - vals[1]) > 5:
                diff = abs(vals[0] - vals[1])
                self.logger.warning(
                    f"[SOFT LEAK RISK] time_since_last_transaction null rate "
                    f"differs by {diff:.1f}pp between classes."
                )
                print(f"   *** {diff:.1f}pp difference - consider is_first_transaction flag.")


        if quality_metrics['overall'] < self.config.COMPLETENESS_THRESHOLD:
            self.logger.warning(f"Low overall quality: {quality_metrics['overall']:.2f}%")
        if duplicates > 0:
            self.logger.warning(f"Found {duplicates} duplicate rows")

        self.results['data_quality'] = data_quality
        self._ckpt_save("data_quality", data_quality)
        self.logger.info("Data quality assessment complete")
        return data_quality

    # -------------------------------------------------------------------------
    # 4. Numerical Features
    # -------------------------------------------------------------------------

    def analyze_numerical_features(self) -> Dict[str, Any]:
        print_section_header("3. NUMERICAL FEATURES ANALYSIS", logger=self.logger)

        cached = self._ckpt_load("numerical_analysis")
        if cached is not None:
            self.results['numerical_analysis'] = cached
            return cached

        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")

        leaky        = self._leaky()
        exclude      = ['is_fraud'] + list(leaky)
        numerical_cols = get_numeric_columns(self.df, exclude=exclude)

        print(f"\n📊 Analyzing {len(numerical_cols)} numerical features...")
        numerical_analysis = {'feature_count': len(numerical_cols), 'features': {}}

        for col in numerical_cols:
            if self.df[col].notna().sum() == 0:
                continue
            try:
                data = self.df[col].dropna()
                if len(data) < 3:
                    continue

                Q1, Q3 = data.quantile(0.25), data.quantile(0.75)

                try:
                    _, normality_p = normaltest(data)
                    is_normal      = bool(normality_p > self.config.NORMALITY_ALPHA)
                except Exception:
                    normality_p, is_normal = None, None

                numerical_analysis['features'][col] = {
                    'count':    int(data.count()),
                    'mean':     float(data.mean()),
                    'median':   float(data.median()),
                    'std':      float(data.std()),
                    'min':      float(data.min()),
                    'max':      float(data.max()),
                    'q1':       float(Q1),
                    'q3':       float(Q3),
                    'iqr':      float(Q3 - Q1),
                    'range':    float(data.max() - data.min()),
                    'skewness': float(data.skew()),
                    'kurtosis': float(data.kurtosis()),
                    'cv':       float(safe_divide(data.std(), data.mean(), 0) * 100),
                    'outliers_iqr':    detect_outliers_iqr(data, self.config.IQR_MULTIPLIER),
                    'outliers_zscore': detect_outliers_zscore(data, self.config.Z_SCORE_THRESHOLD),
                    'normality_p_value': float(normality_p) if normality_p is not None else None,
                    'is_normal': is_normal,
                }

                fs = numerical_analysis['features'][col]
                print(f"\n{col}:")
                print(f"   Mean: {fs['mean']:.4f}, Median: {fs['median']:.4f}")
                print(f"   Skewness: {fs['skewness']:.4f}, Kurtosis: {fs['kurtosis']:.4f}")
                print(f"   Outliers (IQR): {fs['outliers_iqr']['count']} "
                      f"({fs['outliers_iqr']['percentage']:.2f}%)")
                print(f"   Normal distribution: {is_normal}")

            except Exception as e:
                self.logger.error(f"Error analyzing {col}: {str(e)}")

        self.results['numerical_analysis'] = numerical_analysis
        self._ckpt_save("numerical_analysis", numerical_analysis)
        self.logger.info(
            f"Numerical analysis complete for {len(numerical_analysis['features'])} features"
        )
        return numerical_analysis

    # -------------------------------------------------------------------------
    # 5. Categorical Features
    # -------------------------------------------------------------------------

    def analyze_categorical_features(self) -> Dict[str, Any]:
        print_section_header("4. CATEGORICAL FEATURES ANALYSIS", logger=self.logger)

        cached = self._ckpt_load("categorical_analysis")
        if cached is not None:
            self.results['categorical_analysis'] = cached
            return cached

        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")

        categorical_cols = get_categorical_columns(
            self.df, exclude=self.config.SKIP_CATEGORICAL_ANALYSIS
        )

        print(f"\n📊 Analyzing {len(categorical_cols)} categorical features...")
        categorical_analysis = {'feature_count': len(categorical_cols), 'features': {}}

        for col in categorical_cols:
            try:
                value_counts = self.df[col].value_counts()
                if len(value_counts) == 0:
                    continue

                try:
                    entropy_value = float(stats.entropy(value_counts))
                except Exception:
                    entropy_value = None

                unique_ratio = safe_divide(self.df[col].nunique(), len(self.df), 0)
                cardinality  = (
                    "high"   if unique_ratio > 0.5
                    else "medium" if unique_ratio > 0.1
                    else "low"
                )

                categorical_analysis['features'][col] = {
                    'unique_count': int(self.df[col].nunique()),
                    'top_value':    str(value_counts.index[0]),
                    'top_count':    int(value_counts.iloc[0]),
                    'top_pct':      float(
                        safe_divide(value_counts.iloc[0], len(self.df), 0) * 100
                    ),
                    'entropy':      entropy_value,
                    'cardinality':  cardinality,
                    'value_counts': {
                        str(k): int(v)
                        for k, v in value_counts.head(self.config.TOP_N_VALUES).items()
                    },
                }

                fi = categorical_analysis['features'][col]
                print(f"\n{col}: {fi['unique_count']} unique values ({cardinality} cardinality)")
                if fi['unique_count'] <= self.config.MAX_CATEGORICAL_DISPLAY:
                    for val, cnt in list(value_counts.items())[:self.config.TOP_N_VALUES]:
                        pct = safe_divide(cnt, len(self.df), 0) * 100
                        print(f"   • {val}: {cnt} ({pct:.1f}%)")

            except Exception as e:
                self.logger.error(f"Error analyzing {col}: {str(e)}")

        self.results['categorical_analysis'] = categorical_analysis
        self._ckpt_save("categorical_analysis", categorical_analysis)
        self.logger.info(
            f"Categorical analysis complete for {len(categorical_analysis['features'])} features"
        )
        return categorical_analysis

    # -------------------------------------------------------------------------
    # 6. Temporal Features
    # -------------------------------------------------------------------------

    def analyze_temporal_features(self) -> Dict[str, Any]:
        if 'timestamp' not in self.df.columns:
            self.logger.info("No timestamp column found, skipping temporal analysis")
            return {}

        print_section_header("5. TEMPORAL ANALYSIS", logger=self.logger)

        cached = self._ckpt_load("temporal_analysis")
        if cached is not None:
            self.results['temporal_analysis'] = cached
            self._rederive_temporal_columns()
            return cached

        try:
            self.df['timestamp'] = pd.to_datetime(self.df['timestamp'], format='ISO8601')

            if self.config.TEMPORAL_FEATURES:
                self.df['hour']        = self.df['timestamp'].dt.hour
                self.df['day_of_week'] = self.df['timestamp'].dt.dayofweek
                self.df['month']       = self.df['timestamp'].dt.month
                self.df['quarter']     = self.df['timestamp'].dt.quarter
                self.df['is_weekend']  = self.df['day_of_week'].isin(self.config.WEEKEND_DAYS)
                # String columns — display/chi-square only; must NOT enter numeric pipelines
                self.df['day_name']   = self.df['timestamp'].dt.day_name()
                self.df['month_name'] = self.df['timestamp'].dt.month_name()

                self.temporal_feature_cols = [
                    'hour', 'day_of_week', 'month', 'quarter',
                    'is_weekend', 'day_name', 'month_name',
                ]

            time_span = (self.df['timestamp'].max() - self.df['timestamp'].min()).days

            temporal_analysis = {
                'min_date':    str(self.df['timestamp'].min()),
                'max_date':    str(self.df['timestamp'].max()),
                'time_span_days': int(time_span),
                'peak_hour':   int(self.df['hour'].mode().values[0])
                               if 'hour' in self.df.columns else None,
                'peak_day':    str(self.df['day_name'].mode().values[0])
                               if 'day_name' in self.df.columns else None,
                'peak_month':  str(self.df['month_name'].mode().values[0])
                               if 'month_name' in self.df.columns else None,
                'weekend_pct': float(
                    safe_divide(self.df['is_weekend'].sum(), len(self.df), 0) * 100
                ) if 'is_weekend' in self.df.columns else None,
                'hourly_distribution': {
                    int(k): int(v)
                    for k, v in self.df['hour'].value_counts().sort_index().items()
                } if 'hour' in self.df.columns else {},
                'daily_distribution': {
                    str(k): int(v)
                    for k, v in self.df['day_name'].value_counts().items()
                } if 'day_name' in self.df.columns else {},
                'monthly_distribution': {
                    str(k): int(v)
                    for k, v in self.df['month_name'].value_counts().items()
                } if 'month_name' in self.df.columns else {},
                'derived_columns': self.temporal_feature_cols,
            }

            print(f"\n📅 Temporal Coverage: {time_span} days")
            print(f"   From: {temporal_analysis['min_date']}")
            print(f"   To:   {temporal_analysis['max_date']}")
            print(f"\n⏰ Activity Patterns:")
            print(f"   • Peak Hour  : {temporal_analysis['peak_hour']}:00")
            print(f"   • Peak Day   : {temporal_analysis['peak_day']}")
            print(f"   • Peak Month : {temporal_analysis['peak_month']}")
            print(f"   • Weekend    : {temporal_analysis['weekend_pct']:.1f}%")

            self.results['temporal_analysis'] = temporal_analysis
            self._ckpt_save("temporal_analysis", temporal_analysis)
            self.logger.info("Temporal analysis complete")
            return temporal_analysis

        except Exception as e:
            self.logger.error(f"Error in temporal analysis: {str(e)}")
            return {}

    def _rederive_temporal_columns(self):
        """Re-create temporal columns after a checkpoint restore."""
        if 'timestamp' not in self.df.columns:
            return
        try:
            self.df['timestamp']   = pd.to_datetime(self.df['timestamp'], format='ISO8601')
            self.df['hour']        = self.df['timestamp'].dt.hour
            self.df['day_of_week'] = self.df['timestamp'].dt.dayofweek
            self.df['month']       = self.df['timestamp'].dt.month
            self.df['quarter']     = self.df['timestamp'].dt.quarter
            self.df['is_weekend']  = self.df['day_of_week'].isin(self.config.WEEKEND_DAYS)
            self.df['day_name']    = self.df['timestamp'].dt.day_name()
            self.df['month_name']  = self.df['timestamp'].dt.month_name()
            self.temporal_feature_cols = [
                'hour', 'day_of_week', 'month', 'quarter',
                'is_weekend', 'day_name', 'month_name',
            ]
        except Exception as e:
            self.logger.warning(f"Could not re-derive temporal columns: {e}")


# ---------------------------------------------------------------------------
# Attach methods from part 2 and part 3
# ---------------------------------------------------------------------------

from Fraud_eda_part2 import (
    analyze_correlation,
    analyze_feature_importance,
    analyze_pca,
    analyze_clustering,
)

from Fraud_eda_part3 import (
    analyze_statistical_tests,
    assess_model_readiness,
    generate_recommendations,
    run_full_analysis,
    save_results,
)

FraudEDA.analyze_correlation      = analyze_correlation
FraudEDA.analyze_feature_importance = analyze_feature_importance
FraudEDA.analyze_pca              = analyze_pca
FraudEDA.analyze_clustering       = analyze_clustering
FraudEDA.analyze_statistical_tests = analyze_statistical_tests
FraudEDA.assess_model_readiness   = assess_model_readiness
FraudEDA.generate_recommendations = generate_recommendations
FraudEDA.run_full_analysis        = run_full_analysis
FraudEDA.save_results             = save_results