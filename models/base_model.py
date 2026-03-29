import joblib
import logging
import numpy as np
from abc import ABC, abstractmethod
from pathlib import Path

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """Shared interface for all fraud detection models."""

    def __init__(self, name: str, output_dir: str = "models"):
        self.name = name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model = None
        self.threshold = 0.5
        self.is_fitted = False

    @abstractmethod
    def build(self):
        """Instantiate and return the underlying model."""

    def train(self, X_train, y_train):
        logger.info(f"[{self.name}] Training started — {X_train.shape[0]:,} rows, {X_train.shape[1]} features")
        self.model = self.build()
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        logger.info(f"[{self.name}] Training complete")
        return self

    def predict_proba(self, X):
        self._check_fitted()
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X, threshold=None):
        self._check_fitted()
        t = threshold if threshold is not None else self.threshold
        return (self.predict_proba(X) >= t).astype(int)

    def set_threshold(self, threshold: float):
        self.threshold = threshold
        logger.info(f"[{self.name}] Decision threshold set to {threshold:.3f}")

    def save(self):
        self._check_fitted()
        path = self.output_dir / f"{self.name}.joblib"
        joblib.dump(self, path)
        logger.info(f"[{self.name}] Saved to {path}")
        return path

    @staticmethod
    def load(path: str):
        return joblib.load(path)

    def _check_fitted(self):
        if not self.is_fitted:
            raise RuntimeError(f"[{self.name}] Model must be trained before inference.")

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name}, threshold={self.threshold:.2f})"