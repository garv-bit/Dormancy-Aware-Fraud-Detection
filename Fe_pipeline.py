"""
FE_Pipeline.py — Feature Engineering Pipeline for Dormancy Fraud Detection

Transforms raw CSV into a clean, model-ready feature matrix.

Correct pipeline order (no leakage)
-------------------------------------
1.  Load & drop leaky/post-event columns
2.  Parse timestamp → extract temporal features
3.  Dormancy flag (is_first_transaction) — BEFORE imputation
4.  Time-based train/test split on RAW data
5.  Fit imputation median on train only → apply to both
6.  Fit dormancy buckets on train only → apply to both
7.  Fit IP + account frequency maps on train only → apply to both
8.  Amount log-transform (no fitting needed)
9.  Fit LabelEncoders on train only → apply to both
10. Build feature matrices (drop raw strings, IDs, leaky cols)
11. Fit RobustScaler on train only → apply to both
12. Save outputs

Leakage safeguards
------------------
- fraud_type dropped on load (post-event label)
- is_fraud only used as y, never enters X
- ALL encoders/scalers fit on train split only
- Frequency maps built from train only
- Imputation median computed from train non-fraud rows only
- Dormancy percentiles computed from train only
- timestamp dropped from X after features extracted and split performed
- transaction_id dropped from X (pure identifier)
- Raw categorical strings dropped after encoding
"""

import sys
import io
import logging
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Any

from sklearn.preprocessing import LabelEncoder, RobustScaler

from FE_config import FEConfig

# Force UTF-8 stdout on Windows
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True
    )


# ==============================================================================
# LOGGING
# ==============================================================================

def setup_logging(config: FEConfig) -> logging.Logger:
    logger = logging.getLogger('FE_Pipeline')
    logger.setLevel(logging.INFO)
    logger.handlers = []
    fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(config.LOG_FILE, encoding='utf-8')
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# ==============================================================================
# FEATURE ENGINEERING PIPELINE
# ==============================================================================

class FeatureEngineeringPipeline:
    """
    End-to-end feature engineering pipeline for dormancy fraud detection.

    Split-first design: train/test split happens on raw data immediately
    after temporal feature extraction. All encoders, frequency maps,
    imputation values, and scalers are fit exclusively on the train split
    and then applied to both train and test. This eliminates all forms
    of preprocessing leakage.
    """

    def __init__(self, config: FEConfig = None):
        self.config   = config or FEConfig()
        self.config.validate()
        self.logger   = setup_logging(self.config)
        self.df       = None
        self.encoders: Dict[str, Any] = {}
        self.scaler   = None

        # Fitted imputation / bucketing values (train only)
        self._impute_median   = None
        self._dormancy_bins   = None   # [p25, p50, p75] from train
        self._freq_maps: Dict[str, Dict] = {}

        Path("features").mkdir(exist_ok=True)
        self.logger.info("FeatureEngineeringPipeline initialised")

    # =========================================================================
    # STEP 1 — LOAD
    # =========================================================================

    def load(self) -> pd.DataFrame:
        """Load CSV and immediately drop post-event leaky columns."""
        self.logger.info(f"Loading: {self.config.INPUT_CSV}")
        self.df = pd.read_csv(self.config.INPUT_CSV)
        self.logger.info(f"Loaded: {len(self.df):,} rows x {len(self.df.columns)} cols")

        # PRIMARY LEAKAGE SAFEGUARD — drop before anything else
        to_drop = [c for c in self.config.DROP_ON_LOAD if c in self.df.columns]
        if to_drop:
            self.df.drop(columns=to_drop, inplace=True)
            self.logger.info(f"[LEAKAGE GUARD] Dropped on load: {to_drop}")

        return self.df

    # =========================================================================
    # STEP 2 — TEMPORAL FEATURES  (no fitting — pure derivation)
    # =========================================================================

    def build_temporal_features(self) -> pd.DataFrame:
        """
        Parse timestamp and extract temporal features.
        Applied to the full dataframe before split because:
        - These are pure derivations (no statistics computed)
        - timestamp is needed to perform the chronological split
        """
        self.logger.info("Building temporal features...")
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'], format='ISO8601')

        self.df['hour']             = self.df['timestamp'].dt.hour
        self.df['day_of_week']      = self.df['timestamp'].dt.dayofweek
        self.df['month']            = self.df['timestamp'].dt.month
        self.df['is_weekend']       = self.df['day_of_week'].isin(
                                          self.config.WEEKEND_DAYS).astype(int)
        self.df['is_high_risk_hour'] = self.df['hour'].isin(
                                          self.config.HIGH_RISK_HOURS).astype(int)

        self.logger.info(
            "Temporal features: hour, day_of_week, month, is_weekend, is_high_risk_hour"
        )
        return self.df

    # =========================================================================
    # STEP 3 — DORMANCY NULL FLAG  (no fitting — pure derivation)
    # =========================================================================

    def build_dormancy_flag(self) -> pd.DataFrame:
        """
        Create is_first_transaction flag from null pattern.

        MUST happen before imputation AND before split so the flag
        is available in both train and test without any data leakage
        (it is purely derived from whether the value is null — no
        statistics from either split are needed).
        """
        col = self.config.DORMANCY_IMPUTE_COL
        self.df[self.config.DORMANCY_NULL_FLAG_COL] = (
            self.df[col].isnull().astype(int)
        )
        n_first = self.df[self.config.DORMANCY_NULL_FLAG_COL].sum()
        self.logger.info(
            f"is_first_transaction: {n_first:,} "
            f"({n_first/len(self.df)*100:.2f}% of transactions)"
        )
        return self.df

    # =========================================================================
    # STEP 4 — TIME-BASED SPLIT ON RAW DATA
    # =========================================================================

    def split_raw(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split the raw dataframe chronologically BEFORE any fitting.

        Sorting by timestamp ensures the first 80% of transactions
        (chronologically) form the training set and the last 20% form
        the test set. This mirrors real deployment: train on past,
        evaluate on future.

        All subsequent fitting steps operate on df_train only.
        """
        self.logger.info("Performing time-based split on raw data...")

        sorted_idx = self.df['timestamp'].argsort()
        df_sorted  = self.df.iloc[sorted_idx].reset_index(drop=True)

        split_idx  = int(len(df_sorted) * (1 - self.config.TEST_SIZE))
        df_train   = df_sorted.iloc[:split_idx].copy()
        df_test    = df_sorted.iloc[split_idx:].copy()

        self.logger.info(
            f"Train: {len(df_train):,} rows | "
            f"{df_train[self.config.TARGET_COLUMN].sum():,} fraud "
            f"({df_train[self.config.TARGET_COLUMN].mean()*100:.2f}%)"
        )
        self.logger.info(
            f"Test:  {len(df_test):,} rows  | "
            f"{df_test[self.config.TARGET_COLUMN].sum():,} fraud "
            f"({df_test[self.config.TARGET_COLUMN].mean()*100:.2f}%)"
        )
        return df_train, df_test

    # =========================================================================
    # STEP 5 — FIT IMPUTATION ON TRAIN, APPLY TO BOTH
    # =========================================================================

    def fit_imputation(self, df_train: pd.DataFrame) -> None:
        """
        Compute non-fraud median from train set only.
        Using non-fraud median avoids introducing fraud-class signal
        into the imputed values.
        """
        col = self.config.DORMANCY_IMPUTE_COL
        self._impute_median = (
            df_train.loc[df_train[self.config.TARGET_COLUMN] == False, col]
            .median()
        )
        self.logger.info(
            f"[TRAIN] Imputation median (non-fraud): {self._impute_median:.4f}"
        )

    def apply_imputation(self, df: pd.DataFrame) -> pd.DataFrame:
        col = self.config.DORMANCY_IMPUTE_COL
        df[col] = df[col].fillna(self._impute_median)
        return df

    # =========================================================================
    # STEP 6 — FIT DORMANCY BUCKETS ON TRAIN, APPLY TO BOTH
    # =========================================================================

    def fit_dormancy_buckets(self, df_train: pd.DataFrame) -> None:
        """
        Compute percentile bin edges from train set only.
        Percentile-based so bins are data-driven regardless of units.
        """
        col = self.config.DORMANCY_IMPUTE_COL
        p25 = df_train[col].quantile(0.25)
        p50 = df_train[col].quantile(0.50)
        p75 = df_train[col].quantile(0.75)
        self._dormancy_bins = [-float('inf'), p25, p50, p75, float('inf')]
        self.logger.info(
            f"[TRAIN] Dormancy percentiles — "
            f"p25={p25:.4f}, p50={p50:.4f}, p75={p75:.4f}"
        )

    def apply_dormancy_buckets(self, df: pd.DataFrame, split_name: str) -> pd.DataFrame:
        col    = self.config.DORMANCY_IMPUTE_COL
        labels = ['recent', 'moderate', 'dormant', 'long_dormant']
        risk_map = {'recent': 0, 'moderate': 1, 'dormant': 2, 'long_dormant': 3}

        df['dormancy_bucket'] = pd.cut(
            df[col],
            bins=self._dormancy_bins,
            labels=labels,
            include_lowest=True,
        ).astype(str)

        df['dormancy_risk_score'] = (
            df['dormancy_bucket'].map(risk_map).fillna(0).astype(int)
        )

        dist = df['dormancy_bucket'].value_counts()
        self.logger.info(f"[{split_name}] Dormancy bucket distribution:\n{dist}")
        return df

    # =========================================================================
    # STEP 7 — FIT FREQUENCY MAPS ON TRAIN, APPLY TO BOTH
    # =========================================================================

    def fit_frequency_maps(self, df_train: pd.DataFrame) -> None:
        """
        Build frequency maps from train set only.
        At inference time, unseen values map to 1 (rare/unknown).
        """
        freq_cols = [c for c in self.config.HIGH_CARDINALITY_COLS
                     if c in df_train.columns]
        for col in freq_cols:
            self._freq_maps[col] = df_train[col].value_counts().to_dict()
            self.logger.info(
                f"[TRAIN] Frequency map '{col}': "
                f"{df_train[col].nunique():,} unique values"
            )

    def apply_frequency_maps(self, df: pd.DataFrame, split_name: str) -> pd.DataFrame:
        for col, freq_map in self._freq_maps.items():
            if col not in df.columns:
                continue
            out_col = (self.config.IP_FREQ_COL
                       if col == 'ip_address' else f'{col}_freq')
            df[out_col] = df[col].map(freq_map).fillna(1).astype(int)
            self.logger.info(
                f"[{split_name}] '{col}' → '{out_col}' "
                f"range: {df[out_col].min()} to {df[out_col].max()}"
            )
        return df

    # =========================================================================
    # STEP 8 — AMOUNT LOG-TRANSFORM  (no fitting needed)
    # =========================================================================

    def apply_amount_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df['amount_log'] = np.log1p(df['amount'])
        return df

    # =========================================================================
    # STEP 9 — FIT LABEL ENCODERS ON TRAIN, APPLY TO BOTH
    # =========================================================================

    def fit_label_encoders(self, df_train: pd.DataFrame) -> None:
        """Fit LabelEncoders on train categories only."""
        for col in self.config.LOW_CARDINALITY_COLS:
            if col not in df_train.columns:
                continue
            le = LabelEncoder()
            le.fit(df_train[col].astype(str))
            self.encoders[col] = le
            self.logger.info(
                f"[TRAIN] LabelEncoder '{col}': {len(le.classes_)} classes"
            )

        # dormancy_bucket encoder
        le_bucket = LabelEncoder()
        le_bucket.fit(['recent', 'moderate', 'dormant', 'long_dormant'])
        self.encoders['dormancy_bucket'] = le_bucket

    def apply_label_encoders(self, df: pd.DataFrame, split_name: str) -> pd.DataFrame:
        for col, le in self.encoders.items():
            if col == 'dormancy_bucket':
                df['dormancy_bucket_enc'] = df['dormancy_bucket'].map(
                    {c: i for i, c in enumerate(le.classes_)}
                ).fillna(0).astype(int)
            else:
                if col not in df.columns:
                    continue
                # Handle unseen categories gracefully
                known = set(le.classes_)
                df[col] = df[col].astype(str).apply(
                    lambda x: x if x in known else le.classes_[0]
                )
                df[f'{col}_enc'] = le.transform(df[col].astype(str))
        return df

    # =========================================================================
    # STEP 10 — BUILD FEATURE MATRIX
    # =========================================================================

    def build_feature_matrix(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Drop raw strings, identifiers, leaky columns.
        Return X (features) and y (target).
        """
        raw_drop = [
            'timestamp',        # datetime — all signal extracted
            'ip_address',       # raw string → ip_freq
            'sender_account',   # raw string → sender_account_freq
            'receiver_account', # raw string → receiver_account_freq
            'device_hash',      # raw string → device_hash_freq
            'dormancy_bucket',  # raw string → dormancy_bucket_enc
            # Drop raw categoricals — encoded versions kept
            'transaction_type',
            'merchant_category',
            'location',
            'device_used',
            'payment_channel',
        ]
        drop_cols = (
            self.config.DROP_FROM_X   # transaction_id, timestamp
            + raw_drop
            + self.config.LEAKY_COLUMNS
        )

        X = df.drop(columns=[c for c in drop_cols if c in df.columns])
        y = df[self.config.TARGET_COLUMN].astype(int)

        return X, y

    # =========================================================================
    # STEP 11 — FIT SCALER ON TRAIN, APPLY TO BOTH
    # =========================================================================

    def fit_scaler(self, X_train: pd.DataFrame) -> None:
        """Fit RobustScaler on train features only."""
        if not self.config.SCALE_FEATURES:
            return
        cols = [c for c in self.config.COLS_TO_SCALE if c in X_train.columns]
        self.scaler = RobustScaler()
        self.scaler.fit(X_train[cols])
        self.logger.info(f"[TRAIN] RobustScaler fit on: {cols}")

    def apply_scaler(self, X: pd.DataFrame, split_name: str) -> pd.DataFrame:
        if not self.config.SCALE_FEATURES or self.scaler is None:
            return X
        cols = [c for c in self.config.COLS_TO_SCALE if c in X.columns]
        X[cols] = self.scaler.transform(X[cols])
        self.logger.info(f"[{split_name}] Scaled: {cols}")
        return X

    # =========================================================================
    # STEP 12 — SAVE
    # =========================================================================

    def save(self, X_train, X_test, y_train, y_test):
        self.logger.info("Saving outputs...")
        X_train.to_parquet(self.config.OUTPUT_X_TRAIN, index=False)
        X_test.to_parquet(self.config.OUTPUT_X_TEST,   index=False)
        y_train.to_frame().to_parquet(self.config.OUTPUT_Y_TRAIN, index=False)
        y_test.to_frame().to_parquet(self.config.OUTPUT_Y_TEST,   index=False)
        joblib.dump(self.encoders,    self.config.OUTPUT_ENCODERS)
        joblib.dump(self._freq_maps,  'features/freq_maps.joblib')
        joblib.dump(self._dormancy_bins, 'features/dormancy_bins.joblib')
        joblib.dump(self._impute_median, 'features/impute_median.joblib')
        if self.scaler:
            joblib.dump(self.scaler, self.config.OUTPUT_SCALER)
        self.logger.info(f"X_train → {self.config.OUTPUT_X_TRAIN}")
        self.logger.info(f"X_test  → {self.config.OUTPUT_X_TEST}")
        self.logger.info(f"y_train → {self.config.OUTPUT_Y_TRAIN}")
        self.logger.info(f"y_test  → {self.config.OUTPUT_Y_TEST}")
        self.logger.info(f"Encoders → {self.config.OUTPUT_ENCODERS}")
        if self.scaler:
            self.logger.info(f"Scaler  → {self.config.OUTPUT_SCALER}")

    # =========================================================================
    # ORCHESTRATOR
    # =========================================================================

    def run(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Run the full pipeline with correct split-first order.

        FITTING happens only on train. TRANSFORM applies to both.
        This eliminates all preprocessing leakage.
        """
        self.logger.info("=" * 80)
        self.logger.info("FEATURE ENGINEERING PIPELINE — START")
        self.logger.info("=" * 80)

        # ── Pure derivations (no stats computed) ──────────────────────────────
        self.load()
        self.build_temporal_features()   # extract hour/day/month etc.
        self.build_dormancy_flag()       # is_first_transaction from null pattern

        # ── SPLIT FIRST ───────────────────────────────────────────────────────
        df_train, df_test = self.split_raw()

        # ── FIT on train ──────────────────────────────────────────────────────
        self.fit_imputation(df_train)
        self.fit_dormancy_buckets(df_train)   # after imputation fit
        self.fit_frequency_maps(df_train)
        self.fit_label_encoders(df_train)

        # ── APPLY to train ────────────────────────────────────────────────────
        df_train = self.apply_imputation(df_train)
        df_train = self.apply_dormancy_buckets(df_train, 'TRAIN')
        df_train = self.apply_frequency_maps(df_train, 'TRAIN')
        df_train = self.apply_amount_features(df_train)
        df_train = self.apply_label_encoders(df_train, 'TRAIN')

        # ── APPLY to test ─────────────────────────────────────────────────────
        df_test = self.apply_imputation(df_test)
        df_test = self.apply_dormancy_buckets(df_test, 'TEST')
        df_test = self.apply_frequency_maps(df_test, 'TEST')
        df_test = self.apply_amount_features(df_test)
        df_test = self.apply_label_encoders(df_test, 'TEST')

        # ── Build feature matrices ────────────────────────────────────────────
        X_train, y_train = self.build_feature_matrix(df_train)
        X_test,  y_test  = self.build_feature_matrix(df_test)

        self.logger.info(f"X_train: {X_train.shape} | fraud: {y_train.sum():,} ({y_train.mean()*100:.2f}%)")
        self.logger.info(f"X_test:  {X_test.shape}  | fraud: {y_test.sum():,}  ({y_test.mean()*100:.2f}%)")
        self.logger.info(f"Features: {list(X_train.columns)}")

        # ── FIT scaler on train X, APPLY to both ─────────────────────────────
        self.fit_scaler(X_train)
        X_train = self.apply_scaler(X_train, 'TRAIN')
        X_test  = self.apply_scaler(X_test,  'TEST')

        # ── Save ──────────────────────────────────────────────────────────────
        self.save(X_train, X_test, y_train, y_test)

        self.logger.info("=" * 80)
        self.logger.info("FEATURE ENGINEERING PIPELINE — COMPLETE")
        self.logger.info("=" * 80)

        return X_train, X_test, y_train, y_test