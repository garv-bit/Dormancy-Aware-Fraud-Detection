"""
Configuration file for Financial Fraud Detection EDA
All magic numbers, thresholds, and leakage guards live here.
"""

from dataclasses import dataclass
from typing import List, Set
import logging


@dataclass
class EDAConfig:
    """Configuration class for EDA parameters"""

    # ============================================================================
    # FILE PATHS
    # ============================================================================
    INPUT_CSV: str = "Dataset/financial_fraud_detection_dataset.csv"
    OUTPUT_PICKLE: str = "eda_results.joblib"
    OUTPUT_PDF: str = "eda_report.pdf"
    LOG_FILE: str = "eda_analysis.log"
    CHECKPOINT_DIR: str = "eda_checkpoints"  # per-section save folder

    # ============================================================================
    # STATISTICAL THRESHOLDS
    # ============================================================================
    CORRELATION_THRESHOLD: float = 0.5
    ALPHA: float = 0.05
    IQR_MULTIPLIER: float = 1.5
    Z_SCORE_THRESHOLD: float = 3.0
    NORMALITY_ALPHA: float = 0.05

    # ============================================================================
    # MACHINE LEARNING PARAMETERS
    # ============================================================================
    N_ESTIMATORS: int = 100
    RANDOM_STATE: int = 42
    N_JOBS: int = -1
    PCA_VARIANCE_THRESHOLD: float = 0.95
    MIN_CLUSTERS: int = 2
    MAX_CLUSTERS: int = 11
    TEST_SIZE: float = 0.2
    CV_FOLDS: int = 5

    # Max rows used for RF, GB, clustering, and silhouette.
    # Stratified sampling preserves fraud/non-fraud ratio.
    # Keeps RAM usage manageable on 5M-row datasets.
    SAMPLE_SIZE: int = 200_000
    SILHOUETTE_SAMPLE: int = 10_000   # subsample for silhouette — euclidean metric avoids full pairwise matrix

    # ============================================================================
    # DATA QUALITY THRESHOLDS
    # ============================================================================
    COMPLETENESS_THRESHOLD: float = 90.0
    MIN_SAMPLE_SIZE: int = 1000
    MIN_FRAUD_RATE: float = 0.001
    MAX_FRAUD_RATE: float = 0.99
    MIN_FEATURES: int = 5

    # ============================================================================
    # VISUALIZATION PARAMETERS
    # ============================================================================
    PLOT_DPI: int = 300
    PLOT_STYLE: str = "seaborn-v0_8-darkgrid"
    COLOR_PALETTE: str = "husl"
    FIGURE_WIDTH: int = 16
    SUBPLOT_HEIGHT: int = 4
    COLS_PER_ROW: int = 4
    HISTOGRAM_BINS: int = 30

    # ============================================================================
    # DISPLAY PARAMETERS
    # ============================================================================
    MAX_CATEGORICAL_UNIQUE: int = 20
    MAX_CATEGORICAL_DISPLAY: int = 10
    TOP_N_VALUES: int = 5
    TOP_N_FEATURES: int = 15
    MAX_CHI_SQUARE_TESTS: int = 10

    # ============================================================================
    # MEMORY OPTIMIZATION
    # ============================================================================
    STORE_RAW_DATA: bool = False
    SAMPLE_SIZE_FOR_STORAGE: int = 100
    MAX_PCA_POINTS: int = 1000
    MAX_CLUSTER_POINTS: int = 1000

    # ============================================================================
    # LOGGING CONFIGURATION
    # ============================================================================
    LOG_LEVEL: int = logging.INFO
    LOG_FORMAT: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    LOG_TO_FILE: bool = True
    LOG_TO_CONSOLE: bool = True

    # ============================================================================
    # FEATURE ENGINEERING
    # ============================================================================
    TEMPORAL_FEATURES: bool = True
    WEEKEND_DAYS: List[int] = None

    # ============================================================================
    # EXCLUSION LISTS
    # ============================================================================
    ID_COLUMNS: List[str] = None
    SKIP_CATEGORICAL_ANALYSIS: List[str] = None

    # ============================================================================
    # DATA LEAKAGE GUARD  —  SINGLE SOURCE OF TRUTH
    # ============================================================================
    # LEAKY_COLUMNS — must NEVER appear in any feature matrix X:
    #
    #   is_fraud / is_fraud_enc
    #       The target and its encoded copy. Direct leak.
    #
    #   fraud_type / fraud_type_enc
    #       Post-event label. Only non-null when is_fraud=True (96.4% missing).
    #       Assigned after fraud is confirmed — unavailable at inference time.
    #       Proxy leak.
    #
    # DROP_ON_LOAD — dropped from the raw dataframe immediately after CSV read.
    # Primary safeguard; LEAKY_COLUMNS exclusions are the secondary guard.
    LEAKY_COLUMNS: List[str] = None
    DROP_ON_LOAD: List[str] = None

    def __post_init__(self):
        if self.WEEKEND_DAYS is None:
            self.WEEKEND_DAYS = [5, 6]

        if self.ID_COLUMNS is None:
            self.ID_COLUMNS = ["transaction_id", "ip_address"]

        if self.LEAKY_COLUMNS is None:
            self.LEAKY_COLUMNS = [
                "is_fraud",
                "is_fraud_enc",
                "fraud_type",
                "fraud_type_enc",
            ]

        if self.DROP_ON_LOAD is None:
            # fraud_type is a post-event label — drop it the moment the CSV is read
            self.DROP_ON_LOAD = ["fraud_type"]

        if self.SKIP_CATEGORICAL_ANALYSIS is None:
            self.SKIP_CATEGORICAL_ANALYSIS = [
                "transaction_id",
                "timestamp",
                "ip_address",
                "fraud_type",   # post-event label
                "day_name",     # derived temporal string — high cardinality
                "month_name",   # derived temporal string — high cardinality
            ]

    @property
    def leaky_columns_set(self) -> Set[str]:
        """Return LEAKY_COLUMNS as a set for O(1) membership tests."""
        return set(self.LEAKY_COLUMNS)

    def validate(self):
        """Validate configuration parameters."""
        assert 0 < self.CORRELATION_THRESHOLD <= 1,   "Correlation threshold must be (0, 1]"
        assert 0 < self.ALPHA < 1,                    "Alpha must be (0, 1)"
        assert self.IQR_MULTIPLIER > 0,               "IQR multiplier must be positive"
        assert self.Z_SCORE_THRESHOLD > 0,            "Z-score threshold must be positive"
        assert self.N_ESTIMATORS > 0,                 "N_ESTIMATORS must be positive"
        assert 0 < self.PCA_VARIANCE_THRESHOLD <= 1,  "PCA variance threshold must be (0, 1]"
        assert self.MIN_CLUSTERS >= 2,                "MIN_CLUSTERS must be >= 2"
        assert self.MIN_CLUSTERS < self.MAX_CLUSTERS, "MIN_CLUSTERS must be < MAX_CLUSTERS"
        assert 0 < self.TEST_SIZE < 1,                "TEST_SIZE must be (0, 1)"
        assert self.CV_FOLDS > 1,                     "CV_FOLDS must be > 1"
        assert 0 <= self.COMPLETENESS_THRESHOLD <= 100
        assert self.MIN_SAMPLE_SIZE > 0
        assert self.PLOT_DPI > 0
        assert self.SAMPLE_SIZE > 0,                  "SAMPLE_SIZE must be positive"
        assert self.SILHOUETTE_SAMPLE > 0,            "SILHOUETTE_SAMPLE must be positive"
        assert len(self.LEAKY_COLUMNS) > 0,           "LEAKY_COLUMNS must not be empty"
        return True


# Default configuration instance
config = EDAConfig()