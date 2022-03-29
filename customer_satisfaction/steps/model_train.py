import logging
from zenml.steps import step, Output
from model.model_dev import ModelTraining
import pandas as pd
from catboost import CatBoostRegressor
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
    rf_model=RandomForestRegressor,
    lgbm_model=LGBMRegressor,
    xgb_model=XGBRegressor,
):
    """ 
    Args:  
        x_train: pd.DataFrame
        x_test: pd.DataFrame
        y_train: pd.Series
        y_test: pd.Series
    Returns: 
        lg_model: CatBoostRegressor
        rf_model: RandomForestRegressor
        lgbm_model: LGBMRegressor
        xgb_model: XGBRegressor
    """
    model_training = ModelTraining(x_train, y_train, x_test, y_test)
    # lg_model = model_training.Catboost(fine_tuning=False)
    # logging.info("CatBoost model trained")
    rf_model = model_training.random_forest(fine_tuning=False)
    mlflow.sklearn.autolog() 
    mlflow.sklearn.save_model(rf_model, "my_model")
    logging.info("Random Forest model trained")
    lgbm_model = model_training.LightGBM(fine_tuning=False)
    mlflow.lightgbm.autolog()
    logging.info("Light GBM model trained")
    xgb_model = model_training.xgboost(fine_tuning=False)
    mlflow.xgboost.autolog()
    return rf_model, lgbm_model, xgb_model
