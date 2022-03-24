import logging
from zenml.steps import step, Output
from model.model_dev import ModelTraining
import pandas as pd 
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

@step
def train_model(x_train:pd.DataFrame,
    x_test:pd.DataFrame,
    y_train:pd.Series,
    y_test:pd.Series) -> Output(lg_model = CatBoostRegressor, rf_model= RandomForestRegressor, lgbm_model= LGBMRegressor, xgb_model=XGBRegressor): 
    ''' 
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
    '''
    model_training = ModelTraining(x_train, x_test, y_train, y_test)
    lg_model = model_training.Catboost() 
    logging.info("CatBoost model trained")
    rf_model = model_training.random_forest() 
    logging.info("Random Forest model trained")
    lgbm_model = model_training.LightGBM()
    logging.info("Light GBM model trained")
    xgb_model = model_training.xgboost()
    return lg_model, rf_model, lgbm_model, xgb_model
    
