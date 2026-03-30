import logging
import numpy as np
from xgboost import XGBClassifier
from models.base_model import BaseModel

logger = logging.getLogger(__name__)


class XGBoostModel(BaseModel):
    """
    XGBoost — primary production model.
    scale_pos_weight handles class imbalance: (n_negative / n_positive).
    Computed from training data at fit time.
    """

    def __init__(self, output_dir: str = "models"):
        super().__init__(name="xgboost", output_dir=output_dir)
        self._scale_pos_weight = 1.0

    def build(self):
        return XGBClassifier(
            n_estimators=500,
            max_depth=7,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=10,
            scale_pos_weight=self._scale_pos_weight,
            eval_metric="aucpr",
            random_state=42,
            n_jobs=-1,
            verbosity=0,
            tree_method="hist",
        )

    def train(self, X_train, y_train):
        n_neg = int((y_train == 0).sum())
        n_pos = int((y_train == 1).sum())
        self._scale_pos_weight = round(n_neg / n_pos, 2)
        logger.info(f"[{self.name}] scale_pos_weight = {self._scale_pos_weight} "
                    f"({n_neg:,} negatives / {n_pos:,} positives)")
        return super().train(X_train, y_train)

    def get_feature_importance(self, feature_names: list) -> dict:
        self._check_fitted()
        scores = self.model.feature_importances_
        return dict(sorted(zip(feature_names, scores), key=lambda x: x[1], reverse=True))