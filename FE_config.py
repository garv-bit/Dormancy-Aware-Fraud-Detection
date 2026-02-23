"""
FE_Config.py — Feature Engineering Configuration
All thresholds and settings for the feature engineering pipeline.
Keeps magic numbers out of the pipeline code.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class FEConfig:
    """Configuration for the feature engineering pipeline."""

    # =========================================================================
    # FILE PATHS
    # =========================================================================
    INPUT_CSV: str         = "Dataset/financial_fraud_detection_dataset.csv"
    OUTPUT_FEATURES: str   = "features/engineered_features.parquet"
    OUTPUT_X_TRAIN: str    = "features/X_train.parquet"
    OUTPUT_X_TEST: str     = "features/X_test.parquet"
    OUTPUT_Y_TRAIN: str    = "features/y_train.parquet"
    OUTPUT_Y_TEST: str     = "features/y_test.parquet"
    OUTPUT_ENCODERS: str   = "features/encoders.joblib"
    OUTPUT_SCALER: str     = "features/scaler.joblib"
    LOG_FILE: str          = "fe_pipeline.log"

    # =========================================================================
    # MEMORY SAFETY
    # =========================================================================
    # If full in-memory read fails, pipeline falls back to chunked sampling.
    MEMORY_SAFE_MODE: bool   = True
    CHUNK_SIZE: int          = 200_000
    FALLBACK_SAMPLE_ROWS: int = 1_000_000

    # =========================================================================
    # LEAKAGE GUARD  (same source of truth as EDAConfig)
    # =========================================================================
    # fraud_type is dropped on load — it is a post-event label.
    # is_fraud is the target — never enters X.
    DROP_ON_LOAD: List[str] = field(default_factory=lambda: ["fraud_type"])
    TARGET_COLUMN: str      = "is_fraud"

    # Columns that must never appear in the feature matrix X
    LEAKY_COLUMNS: List[str] = field(default_factory=lambda: [
        "is_fraud", "is_fraud_enc", "fraud_type", "fraud_type_enc"
    ])

    # =========================================================================
    # COLUMNS TO DROP FROM FEATURE MATRIX
    # =========================================================================
    # transaction_id  — pure row identifier, zero signal
    # timestamp       — raw string; all signal extracted into derived features
    #                   keeping it would be a string column with no model value
    DROP_FROM_X: List[str] = field(default_factory=lambda: [
        "transaction_id", "timestamp"
    ])

    # =========================================================================
    # DORMANCY / TEMPORAL SETTINGS
    # =========================================================================
    # time_since_last_transaction null means this is the account's first
    # transaction ever — that itself is a strong dormancy signal.
    DORMANCY_NULL_FLAG_COL: str  = "is_first_transaction"
    DORMANCY_IMPUTE_COL: str     = "time_since_last_transaction"

    # Log-transform amount — EDA showed right skew (skewness=1.65)
    LOG_TRANSFORM_AMOUNT: bool = True

    # Dormancy risk thresholds (days) — used to bucket time_since_last_transaction
    # 0-30 days   : recent      (low dormancy risk)
    # 30-180 days : moderate    (medium dormancy risk)
    # 180-365 days: dormant     (high dormancy risk)
    # 365+ days   : long_dormant(very high dormancy risk)
    DORMANCY_BINS: List[float] = field(default_factory=lambda: [
        0, 30, 180, 365, float('inf')
    ])
    DORMANCY_LABELS: List[str] = field(default_factory=lambda: [
        "recent", "moderate", "dormant", "long_dormant"
    ])

    # Weekend days (Mon=0 … Sun=6)
    WEEKEND_DAYS: List[int] = field(default_factory=lambda: [5, 6])

    # High-risk hours for fraud — late night / early morning
    HIGH_RISK_HOURS: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4, 5, 22, 23])

    # =========================================================================
    # IP ADDRESS FEATURES
    # =========================================================================
    # We keep ip_address and derive a frequency-based feature from it
    # rather than label-encoding the raw string (too high cardinality)
    IP_FREQ_COL: str = "ip_freq"   # how often this IP appears in the dataset

    # =========================================================================
    # ENCODING SETTINGS
    # =========================================================================
    # Low-cardinality cols (<=10 unique) — Label Encode
    # High-cardinality cols (>10 unique) — Frequency Encode
    HIGH_CARDINALITY_THRESHOLD: int = 10

    LOW_CARDINALITY_COLS: List[str] = field(default_factory=lambda: [
        "transaction_type",
        "merchant_category",
        "location",
        "device_used",
        "payment_channel",
    ])

    HIGH_CARDINALITY_COLS: List[str] = field(default_factory=lambda: [
        "sender_account",
        "receiver_account",
        "device_hash",
        "ip_address",
    ])

    # =========================================================================
    # SCALING
    # =========================================================================
    # RobustScaler used instead of StandardScaler because:
    #   - amount has heavy outliers (8.25% IQR outliers from EDA)
    #   - time_since_last_transaction has extreme std (3576 vs mean 1.53)
    # RobustScaler uses median + IQR so outliers don't distort the scale.
    SCALE_FEATURES: bool = True

    # These numeric cols are scaled — derived categoricals / flags are not
    COLS_TO_SCALE: List[str] = field(default_factory=lambda: [
        "amount",
        "amount_log",
        "time_since_last_transaction",
        "spending_deviation_score",
        "velocity_score",
        "geo_anomaly_score",
        "ip_freq",
    ])

    # =========================================================================
    # TRAIN / TEST SPLIT
    # =========================================================================
    # TimeSeriesSplit — must split on time, NOT randomly.
    # Random split would leak future transactions into training.
    SPLIT_BY_TIME: bool   = True
    TEST_SIZE: float       = 0.2
    RANDOM_STATE: int      = 42

    def validate(self):
        assert 0 < self.TEST_SIZE < 1, "TEST_SIZE must be between 0 and 1"
        assert self.HIGH_CARDINALITY_THRESHOLD > 0
        assert len(self.DORMANCY_BINS) == len(self.DORMANCY_LABELS) + 1
        assert self.TARGET_COLUMN not in self.DROP_FROM_X
        return True
