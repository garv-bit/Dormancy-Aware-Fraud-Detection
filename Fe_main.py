"""
FE_Main.py — Entry point for the Feature Engineering Pipeline

Run from the project root:
    python src/FE_Main.py

Outputs (written to features/ folder):
    X_train.parquet       — training feature matrix
    X_test.parquet        — test feature matrix
    y_train.parquet       — training labels
    y_test.parquet        — test labels
    encoders.joblib       — fitted LabelEncoders + frequency maps
    scaler.joblib         — fitted RobustScaler
"""

import sys
import io

if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True
    )

from FE_config import FEConfig
from Fe_pipeline import FeatureEngineeringPipeline


def print_header(config: FEConfig):
    width = 80
    print("=" * width)
    print(" " * 18 + "DORMANCY FRAUD DETECTION")
    print(" " * 16 + "Feature Engineering Pipeline")
    print("=" * width)
    print(f"\n  Input CSV      : {config.INPUT_CSV}")
    print(f"  Output dir     : features/")
    print(f"  Target column  : {config.TARGET_COLUMN}")
    print(f"  Test size      : {config.TEST_SIZE*100:.0f}%  (time-based split)")
    print(f"  Scaler         : RobustScaler")
    print(f"  Dormancy bins  : {config.DORMANCY_BINS}")
    print("=" * width + "\n")


def print_summary(X_train, X_test, y_train, y_test):
    width = 80
    print("\n" + "=" * width)
    print("SUMMARY")
    print("=" * width)
    print(f"\n  Train set : {len(X_train):,} rows  |  "
          f"{y_train.sum():,} fraud  ({y_train.mean()*100:.2f}%)")
    print(f"  Test set  : {len(X_test):,} rows  |  "
          f"{y_test.sum():,} fraud  ({y_test.mean()*100:.2f}%)")
    print(f"\n  Features  : {X_train.shape[1]}")
    print(f"\n  Feature list:")
    for col in X_train.columns:
        print(f"    - {col}")
    print(f"\n  Outputs saved to: features/")
    print("=" * width + "\n")


def main():
    config = FEConfig()
    print_header(config)

    try:
        pipeline = FeatureEngineeringPipeline(config)
        X_train, X_test, y_train, y_test = pipeline.run()
        print_summary(X_train, X_test, y_train, y_test)
        return 0

    except FileNotFoundError as e:
        print(f"\nERROR: File not found — {e}")
        print("  Check that INPUT_CSV in FE_Config.py points to the correct path.")
        return 1

    except MemoryError:
        print("\nERROR: Out of memory.")
        print("  The dataset is 5M rows x 3.2GB. Close other applications and retry.")
        return 1

    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())