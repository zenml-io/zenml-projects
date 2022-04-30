import mlflow
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.model_selection import train_test_split
from zenml.integrations.mlflow.mlflow_step_decorator import enable_mlflow
from zenml.logger import get_logger
from zenml.steps import BaseStepConfig, Output, step

from .src.tree_based_models import TreeBasedModels

logger = get_logger(__name__)


class ModelNameConfig(BaseStepConfig):
    """Model Configurations"""

    model_name: str = "lightgbm"
    fine_tuning: bool = False


@enable_mlflow
@step
def model_trainer(train: pd.DataFrame, config: ModelNameConfig) -> Output(model=ClassifierMixin):
    """Trains a specified model."""
    try:
        x_train, x_test, y_train, y_test = train_test_split(
            train.drop("Churn", axis=1), train["Churn"], test_size=0.2
        )
        tree_based_model = TreeBasedModels(x_train, y_train, x_test, y_test)

        if config.model_name == "lightgbm":
            mlflow.lightgbm.autolog()
            lgm_model = tree_based_model.lightgbm_trainer(fine_tuning=config.fine_tuning)
            return lgm_model
        elif config.model_name == "randomforest":
            mlflow.sklearn.autolog()
            rf_model = tree_based_model.random_forest_trainer(fine_tuning=config.fine_tuning)
            return rf_model
        elif config.model_name == "xgboost":
            mlflow.xgboost.autolog()
            xgb_model = tree_based_model.xgboost_trainer(fine_tuning=config.fine_tuning)
            return xgb_model
        else:
            raise ValueError("Model name not supported")
    except Exception as e:
        logger.error(e)
        raise e
