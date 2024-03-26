import pandas as pd
from sklearn.base import ClassifierMixin
from zenml.logger import get_logger
from zenml.steps import Output, step

from .src.evaluator import Evaluation

logger = get_logger(__name__)


@step
def evaluation(model: ClassifierMixin, test: pd.DataFrame) -> Output(
    accuracy=float
):
    """
    Args:
        model: ClassifierMixin
        x_test: pd.DataFrame
        y_test: pd.Series
    Returns:
        accuracy: float
    """
    try:
        X = test.drop("Churn", axis=1)
        y = test["Churn"]
        prediction = model.predict(X)
        evaluation = Evaluation(y, prediction)
        accuracy = evaluation.get_accuracy()
        return accuracy
    except Exception as e:
        logger.error(e)
        raise e
