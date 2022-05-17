import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from zenml.logger import get_logger
from zenml.steps import Output

logger = get_logger(__name__)


class Evaluation:
    """Class for evaluating the model"""

    def __init__(self, y_true: pd.Series, y_pred: np.ndarray):
        """Initialize the Evaluation class."""
        self.y_true = y_true
        self.y_pred = y_pred

    def get_accuracy(self) -> Output(accuracy=float):
        """Calculate accuracy of the model."""
        try:
            accuracy = accuracy_score(self.y_true, self.y_pred)
            return accuracy
        except Exception as e:
            logger.error(e)
            raise e
