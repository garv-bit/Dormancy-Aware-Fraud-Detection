"""
Utility functions for EDA pipeline.
Includes logging setup, validation, sampling, checkpointing, and common operations.
"""

import logging
import sys
import joblib
import numpy as np
import pandas as pd

from pathlib import Path
from typing import Optional, List, Dict, Any
from Config import EDAConfig


# ==============================================================================
# LOGGING
# ==============================================================================

def setup_logging(config: EDAConfig) -> logging.Logger:
    """
    Configure logging with file and console handlers.
    Forces UTF-8 on both handlers so emoji / special characters in log
    messages do not crash on Windows (cp1252) terminals or log files.
    """
    import io

    logger = logging.getLogger('FraudEDA')
    logger.setLevel(config.LOG_LEVEL)
    logger.handlers = []

    formatter = logging.Formatter(config.LOG_FORMAT)

    if config.LOG_TO_CONSOLE:
        # Wrap stdout in a UTF-8 writer so emoji don't raise UnicodeEncodeError
        utf8_stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True
        )
        ch = logging.StreamHandler(utf8_stdout)
        ch.setLevel(config.LOG_LEVEL)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    if config.LOG_TO_FILE:
        # Explicitly open log file as UTF-8
        fh = logging.FileHandler(config.LOG_FILE, encoding='utf-8')
        fh.setLevel(config.LOG_LEVEL)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


# ==============================================================================
# VALIDATION
# ==============================================================================

def validate_dataframe(df: pd.DataFrame, logger: logging.Logger) -> bool:
    """Validate input dataframe meets basic requirements."""
    if df is None or df.empty:
        raise ValueError("DataFrame is None or empty")
    if len(df.columns) < 3:
        raise ValueError(f"Too few columns: {len(df.columns)}")
    if len(df) < 10:
        logger.warning(f"Very small dataset: {len(df)} rows")
    logger.info(f"DataFrame validated: {len(df):,} rows × {len(df.columns)} cols")
    return True


def validate_file_exists(filepath: str, logger: logging.Logger) -> bool:
    """Check if file exists before attempting to read."""
    path = Path(filepath)
    if not path.exists():
        logger.error(f"File not found: {filepath}")
        raise FileNotFoundError(f"File not found: {filepath}")
    logger.info(f"File found: {filepath}")
    return True


# ==============================================================================
# MATH HELPERS
# ==============================================================================

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default on zero / NaN / Inf."""
    try:
        if denominator == 0 or np.isnan(denominator) or np.isinf(denominator):
            return default
        result = numerator / denominator
        return default if (np.isnan(result) or np.isinf(result)) else result
    except (ZeroDivisionError, TypeError, ValueError):
        return default


# ==============================================================================
# COLUMN HELPERS
# ==============================================================================

def get_numeric_columns(df: pd.DataFrame, exclude: Optional[List[str]] = None) -> List[str]:
    """Return numeric columns, minus any in exclude."""
    exclude = exclude or []
    return [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]


def get_categorical_columns(df: pd.DataFrame, exclude: Optional[List[str]] = None) -> List[str]:
    """Return object/bool/category columns, minus any in exclude."""
    exclude = exclude or []
    return [
        c for c in df.select_dtypes(include=['object', 'bool', 'category']).columns
        if c not in exclude
    ]


# ==============================================================================
# DATA QUALITY
# ==============================================================================

def calculate_data_quality_score(
    df: pd.DataFrame, config: EDAConfig, logger: logging.Logger
) -> Dict[str, float]:
    """
    Calculate comprehensive data quality metrics.

    FIX: overall is a proper weighted sum, not np.mean() of pre-weighted values.
    Previous formula: np.mean([c*0.4, u*0.2, co*0.2, v*0.2]) ≈ 25% even when all=100%.
    Correct formula:  c*0.4 + u*0.2 + co*0.2 + v*0.2              ≈ 100% when all=100%.
    """
    total_cells   = len(df) * len(df.columns)
    missing_cells = df.isnull().sum().sum()
    completeness  = safe_divide(total_cells - missing_cells, total_cells, 0) * 100

    duplicates  = df.duplicated().sum()
    uniqueness  = safe_divide(len(df) - duplicates, len(df), 0) * 100

    inconsistent = 0
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                pd.to_numeric(df[col].dropna(), errors='raise')
                inconsistent += 1
            except (ValueError, TypeError):
                pass
    consistency = (1 - inconsistent / len(df.columns)) * 100 if len(df.columns) > 0 else 100.0

    numeric_cols = get_numeric_columns(df)
    invalid      = sum(1 for c in numeric_cols if np.isinf(df[c]).any())
    validity     = (1 - invalid / len(numeric_cols)) * 100 if numeric_cols else 100.0

    # Weighted sum (not mean of pre-weighted values)
    overall = (
        completeness * 0.4
        + uniqueness  * 0.2
        + consistency * 0.2
        + validity    * 0.2
    )

    logger.info(f"Data quality — Overall: {overall:.2f}%")
    return {
        'completeness': completeness,
        'uniqueness':   uniqueness,
        'consistency':  consistency,
        'validity':     validity,
        'overall':      overall,
    }


# ==============================================================================
# OUTLIER DETECTION
# ==============================================================================

def detect_outliers_iqr(series: pd.Series, multiplier: float = 1.5) -> Dict[str, Any]:
    """Detect outliers using the IQR method."""
    Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
    IQR = Q3 - Q1
    lb, ub = Q1 - multiplier * IQR, Q3 + multiplier * IQR
    mask = (series < lb) | (series > ub)
    return {
        'count':       int(mask.sum()),
        'percentage':  float(safe_divide(mask.sum(), len(series), 0) * 100),
        'lower_bound': float(lb),
        'upper_bound': float(ub),
        'iqr':         float(IQR),
    }


def detect_outliers_zscore(series: pd.Series, threshold: float = 3.0) -> Dict[str, Any]:
    """Detect outliers using the Z-score method."""
    from scipy import stats
    z    = np.abs(stats.zscore(series.dropna()))
    mask = z > threshold
    return {
        'count':      int(mask.sum()),
        'percentage': float(safe_divide(mask.sum(), len(series), 0) * 100),
        'threshold':  float(threshold),
    }


# ==============================================================================
# SAMPLING
# ==============================================================================

def stratified_sample(
    df: pd.DataFrame,
    n: int,
    target_col: str = "is_fraud",
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Return a stratified sample of at most n rows.

    Preserves the fraud / non-fraud ratio so ML steps run on a
    representative subset without needing all 5M rows in RAM.
    Falls back to plain sampling if target_col is absent or n >= len(df).
    """
    if n >= len(df):
        return df
    if target_col not in df.columns:
        return df.sample(n=n, random_state=random_state)

    frac    = n / len(df)
    sampled = (
        df.groupby(target_col, group_keys=False)
          .apply(lambda x: x.sample(frac=frac, random_state=random_state))
    )
    return sampled.sample(n=min(n, len(sampled)), random_state=random_state)


# ==============================================================================
# CHECKPOINTING
# ==============================================================================

def checkpoint_path(checkpoint_dir: str, section: str) -> Path:
    return Path(checkpoint_dir) / f"{section}.joblib"


def load_checkpoint(
    checkpoint_dir: str, section: str, logger: logging.Logger
) -> Any:
    """Return cached result if checkpoint exists, else None."""
    path = checkpoint_path(checkpoint_dir, section)
    if path.exists():
        try:
            data = joblib.load(path)
            logger.info(f"[CHECKPOINT] Loaded '{section}' from {path}")
            print(f"   ✅ [{section}] loaded from checkpoint — skipping.")
            return data
        except Exception as e:
            logger.warning(f"[CHECKPOINT] Failed to load '{section}': {e} — re-running.")
    return None


def save_checkpoint(
    checkpoint_dir: str, section: str, data: Any, logger: logging.Logger
):
    """Persist a section result to disk immediately after completion."""
    path = checkpoint_path(checkpoint_dir, section)
    Path(checkpoint_dir).mkdir(exist_ok=True)
    try:
        joblib.dump(data, path)
        logger.info(f"[CHECKPOINT] Saved '{section}' -> {path}")
    except Exception as e:
        logger.warning(f"[CHECKPOINT] Could not save '{section}': {e}")


# ==============================================================================
# FORMATTING / OUTPUT HELPERS
# ==============================================================================

def format_bytes(b: int) -> str:
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if b < 1024.0:
            return f"{b:.2f} {unit}"
        b /= 1024.0
    return f"{b:.2f} PB"


def print_section_header(
    title: str, width: int = 120, logger: Optional[logging.Logger] = None
):
    sep     = "=" * width
    message = f"\n{sep}\n{title}\n{sep}"
    if logger:
        logger.info(title)
    print(message)


def create_output_directory(
    output_dir: str = "eda_outputs", logger: Optional[logging.Logger] = None
) -> Path:
    path = Path(output_dir)
    path.mkdir(exist_ok=True)
    if logger:
        logger.info(f"Output directory: {path.absolute()}")
    return path