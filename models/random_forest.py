import logging
from sklearn.ensemble import RandomForestClassifier
from models.base_model import BaseModel

logger = logging.getLogger(__name__)


class RandomForestModel(BaseModel):
    """
    Random Forest ensemble model.
    Strong non-linear learner, naturally handles feature interactions.
    Uses class_weight='balanced_subsample' for imbalance — reweights each tree.
    """

    def __init__(self, output_dir: str = "models"):
        super().__init__(name="random_forest", output_dir=output_dir)

    def build(self):
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=20,
            min_samples_leaf=10,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
            verbose=0,
        )