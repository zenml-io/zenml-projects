import pandas as pd
import numpy as np
import logging

import optuna
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from lightgbm import LGBMRegressor
import xgboost as xgb
from xgboost import XGBRegressor
from catboost import CatBoostRegressor


class Hyperparameter_Optimization:  

    ''' 
    Class for doing hyperparameter optimization.

    '''
    def __init__(self, x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame, y_test: pd.Series) -> None:
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test


    def optimize_randomforest(self, trial: optuna.Trial) -> float: 
        '''
        Method for optimizing Random Forest

        '''
        n_estimators = trial.suggest_int("n_estimators", 1, 200)
        max_depth = trial.suggest_int("max_depth", 1, 20)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
        reg = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
        )
        reg.fit(self.x_train, self.y_train)
        val_accuracy = reg.score(self.x_test, self.y_test)
        return val_accuracy

    def optimize_lightgbm(self, trial: optuna.Trial) -> float:
        '''
        Method for Optimizing LightGBM 
        '''
        n_estimators = trial.suggest_int("n_estimators", 1, 200)
        max_depth = trial.suggest_int("max_depth", 1, 20)
        learning_rate = trial.suggest_uniform("learning_rate", 0.01, 0.99)
        reg = LGBMRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
        )
        reg.fit(self.x_train, self.y_train)
        val_accuracy = reg.score(self.x_test, self.y_test)
        return val_accuracy

    def optimize_xgboost_regressor(self, trial: optuna.Trial) -> float: 
        '''
        Method for Optimizing Xgboost
        '''
        param = {
            "max_depth": trial.suggest_int("max_depth", 1, 30),
            "learning_rate": trial.suggest_loguniform(
                "learning_rate", 1e-7, 10.0
            ),
            "n_estimators": trial.suggest_int("n_estimators", 1, 200),
        }
        reg = xgb.XGBRegressor(**param)
        reg.fit(self.x_train, self.y_train)
        val_accuracy = reg.score(self.x_test, self.y_test)
        return val_accuracy


class ModelTraining: 
    '''
    Class for training models.
    '''
    def __init__(self, x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame, y_test: pd.Series) -> None:
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test


    def random_forest_model(self, fine_tuning: bool = True): 
        """
        It trains the random forest model.
        
        Args:
            fine_tuning: If True, hyperparameter optimization is performed. If False, the default
            parameters are used, defaults to True (optional)
        """
        logging.info("Entered for training Random Forest model")
        try:
            if fine_tuning:
                hyper_opt = Hyperparameter_Optimization(
                    self.x_train, self.y_train, self.x_test, self.y_test
                )
                study = optuna.create_study(direction="maximize")
                study.optimize(hyper_opt.optimize_randomforest, n_trials=100)
                trial = study.best_trial
                n_estimators = trial.params["n_estimators"]
                max_depth = trial.params["max_depth"]
                min_samples_split = trial.params["min_samples_split"]
                print("Best parameters : ", trial.params)
                reg = RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                )
                reg.fit(self.x_train, self.y_train)
                return reg
            else:
                model = RandomForestRegressor(
                    n_estimators=152, max_depth=20, min_samples_split=17
                )
                model.fit(self.x_train, self.y_train)
                return model
        except Exception as e:
            logging.error("Error in training Random Forest model")
            logging.error(e)
            return None


    def lightgbm_model(self, fine_tuning: bool = True): 
        """
        It trains the LightGBM model.
        
        Args:
            fine_tuning: If True, hyperparameter optimization is performed. If False, the default
            parameters are used, defaults to True (optional)
        """

        logging.info("Entered for training LightGBM model")
        try:
            if fine_tuning:
                hyper_opt = Hyperparameter_Optimization(
                    self.x_train, self.y_train, self.x_test, self.y_test
                )
                study = optuna.create_study(direction="maximize")
                study.optimize(hyper_opt.optimize_lightgbm, n_trials=100)
                trial = study.best_trial
                n_estimators = trial.params["n_estimators"]
                max_depth = trial.params["max_depth"]
                learning_rate = trial.params["learning_rate"]
                reg = LGBMRegressor(
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    max_depth=max_depth,
                )
                reg.fit(self.x_train, self.y_train)
                return reg
            else:
                model = LGBMRegressor(
                    n_estimators=200, learning_rate=0.01, max_depth=20
                )
                model.fit(self.x_train, self.y_train)
                return model
        except Exception as e:
            logging.error("Error in training LightGBM model")
            logging.error(e)
            return None

    def xgboost_model(self, fine_tuning: bool = True):
        """
        It trains the xgboost model.
        
        Args:
            fine_tuning: If True, hyperparameter optimization is performed. If False, the default
            parameters are used, defaults to True (optional)
        """

        logging.info("Entered for training XGBoost model")
        try:
            if fine_tuning:
                hy_opt = Hyperparameter_Optimization(
                    self.x_train, self.y_train, self.x_test, self.y_test
                )
                study = optuna.create_study(direction="maximize")
                study.optimize(hy_opt.optimize_xgboost_regressor, n_trials=100)
                trial = study.best_trial
                n_estimators = trial.params["n_estimators"]
                learning_rate = trial.params["learning_rate"]
                max_depth = trial.params["max_depth"]
                reg = xgb.XGBRegressor(
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    max_depth=max_depth,
                )
                reg.fit(self.x_train, self.y_train)
                return reg

            else:
                model = xgb.XGBRegressor(
                    n_estimators=200, learning_rate=0.01, max_depth=20
                )
                model.fit(self.x_train, self.y_train)
                return model
        except Exception as e:
            logging.error("Error in training XGBoost model")
            logging.error(e)
            return None

