from model.evaluation import Evaluation

import pandas as pd
import numpy as np
import logging 

from sklearn.base import RegressorMixin

from zenml.steps import step, Output
from zenml.integrations.mlflow.mlflow_step_decorator import enable_mlflow
import mlflow

@enable_mlflow
@step
def evaluation(
    model: RegressorMixin, x_test: pd.DataFrame, y_test: pd.Series
) -> Output(r2_score=float, rmse=float):
    """ 
    Args:  
        model: RegressorMixin
        x_test: pd.DataFrame
        y_test: pd.Series
    Returns: 
        r2_score: float
        rmse: float
    """ 
    try: 
        prediction = model.predict(x_test)
        evaluation = Evaluation()
        r2_score = evaluation.r2_score(y_test, prediction) 
        mlflow.log_metric("r2_score", r2_score)
        mse = evaluation.mean_squared_error(y_test, prediction)
        mlflow.log_metric("mse", mse)
        rmse = np.sqrt(mse)
        mlflow.log_metric("rmse", rmse)
        return mse, rmse
    except Exception as e: 
        logging.error(e)
        raise e
        
