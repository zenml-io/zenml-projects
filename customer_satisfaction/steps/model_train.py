import logging
from zenml.steps import step, Output
from model.model_dev import ModelTraining
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from zenml.integrations.constants import MLFLOW
from zenml.integrations.mlflow.mlflow_step_decorator import enable_mlflow
import mlflow


@enable_mlflow
@step
def train_model(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> Output(
    lgbm_model=LGBMRegressor,
):
    """ 
    Args:  
        x_train: pd.DataFrame
        x_test: pd.DataFrame
        y_train: pd.Series
        y_test: pd.Series
    Returns: 
        lgm_model: LGBMRegressor
    """
    model_training = ModelTraining(x_train, y_train, x_test, y_test)
    mlflow.lightgbm.autolog()
    lgm_model = model_training.lightgbm_model(fine_tuning=False)
    logging.info("Light GBM model trained")
    return lgm_model
