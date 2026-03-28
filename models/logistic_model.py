import logging
from sklearn.linear_model import LogisticRegression
from models.base_model import BaseModel

logger = logging.getLogger(__name__)


class LogisticModel(BaseModel):
    """
    Logistic Regression baseline.
    Fast, interpretable, good for establishing a performance floor.
    Uses class_weight='balanced' to handle 3.59% fraud imbalance.
    """

    def __init__(self, output_dir: str = "models"):
        super().__init__(name="logistic_regression", output_dir=output_dir)

    def build(self):
        return LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=1000,
            solver="lbfgs",
            random_state=42,
            n_jobs=-1,
        )