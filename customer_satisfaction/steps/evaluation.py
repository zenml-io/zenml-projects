from model.evaluation import Evaluation
from zenml.steps import step, Output
import logging

import pandas as pd
import numpy as np

from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from zenml.integrations.constants import MLFLOW
from zenml.integrations.mlflow.mlflow_step_decorator import enable_mlflow
import mlflow

@enable_mlflow
@step
def evaluation(
    model: CatBoostRegressor, x_test: pd.DataFrame, y_test: pd.Series
) -> Output(r2_score=float, rmse=float):
    """ 
    Args:  
        lg_model: CatBoostRegressor
        rf_model: RandomForestRegressor
        lgbm_model: LGBMRegressor
        xgb_model: XGBRegressor
        x_test: pd.DataFrame
        y_test: pd.Series
    Returns: 
        lg_score: float
        rf_score: float
        lgbm_score: float
        xgb_score: float
    """
    prediction = model.predict(x_test)
    evaluation = Evaluation()
    r2_score = evaluation.r2_score(y_test, prediction) 
    mlflow.log_metric("r2_score", r2_score)
    mse = evaluation.mean_squared_error(y_test, prediction)
    mlflow.log_metric("mse", mse)
    rmse = np.sqrt(mse)
    mlflow.log_metric("rmse", rmse)
    return r2_score, rmse

