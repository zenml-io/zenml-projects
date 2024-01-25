import logging
from typing import Annotated

import mlflow
import pandas as pd

from model.model_dev import ModelTrainer
from sklearn.base import RegressorMixin
from zenml.client import Client
from zenml import step
from zenml import ArtifactConfig


experiment_tracker = Client().active_stack.experiment_tracker


@step(enable_cache=False, experiment_tracker=experiment_tracker.name)
def train_model(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    model_type: str = "lightgbm",
    do_fine_tuning: bool = True
) -> Annotated[
    RegressorMixin,
    ArtifactConfig(name="sklearn_regressor", is_model_artifact=True)
]:
    """
    Args:
        x_train: pd.DataFrame
        x_test: pd.DataFrame
        y_train: pd.Series
        y_test: pd.Series
        model_type: str - available options ["lightgbm", "randomforest", "xgboost"]
        do_fine_tuning: Should full training run or only fine tuning
    Returns:
        model: RegressorMixin
    """
    try:
        model_training = ModelTrainer(x_train, y_train, x_test, y_test)

        if model_type == "lightgbm":
            mlflow.lightgbm.autolog()
            lgm_model = model_training.lightgbm_trainer(
                fine_tuning=do_fine_tuning
            )
            return lgm_model
        elif model_type == "randomforest":
            mlflow.sklearn.autolog()
            rf_model = model_training.random_forest_trainer(
                fine_tuning=do_fine_tuning
            )
            return rf_model
        elif model_type == "xgboost":
            mlflow.xgboost.autolog()
            xgb_model = model_training.xgboost_trainer(
                fine_tuning=do_fine_tuning
            )
            return xgb_model
        else:
            raise ValueError("Model type not supported")
    except Exception as e:
        logging.error(e)
        raise e
