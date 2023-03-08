import logging

import mlflow
import pandas as pd
from model.model_dev import ModelTraining
from sklearn.base import RegressorMixin
from zenml.client import Client
from zenml.steps import Output, step

from .config import ModelNameConfig

experiment_tracker = Client().active_stack.experiment_tracker


@step(experiment_tracker=experiment_tracker.name)
def train_model(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    config: ModelNameConfig,
) -> Output(model=RegressorMixin):
    """
    Args:
        x_train: pd.DataFrame
        x_test: pd.DataFrame
        y_train: pd.Series
        y_test: pd.Series
    Returns:
        model: RegressorMixin
    """
    try:
        model_training = ModelTraining(x_train, y_train, x_test, y_test)

        if config.model_name == "lightgbm":
            mlflow.lightgbm.autolog()
            lgm_model = model_training.lightgbm_trainer(
                fine_tuning=config.fine_tuning
            )
            return lgm_model
        elif config.model_name == "randomforest":
            mlflow.sklearn.autolog()
            rf_model = model_training.random_forest_trainer(
                fine_tuning=config.fine_tuning
            )
            return rf_model
        elif config.model_name == "xgboost":
            mlflow.xgboost.autolog()
            xgb_model = model_training.xgboost_trainer(
                fine_tuning=config.fine_tuning
            )
            return xgb_model
        else:
            raise ValueError("Model name not supported")
    except Exception as e:
        logging.error(e)
        raise e
