import gc
from typing import Union

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from zenml.logger import get_logger
from zenml.steps import Output

from .configs import StackEnsembleConfig

logger = get_logger(__name__)
from rich import print as rprint


class StackedEnsembles:
    def __init__(
        self,
        stack_config: StackEnsembleConfig,
        train: pd.DataFrame,
        test: pd.DataFrame,
    ) -> None:
        """Initialize the Stacked Ensembles model.

        Args:
            stack_config: the configuration for the stacking model
            train: training data
            test: testing data
            y: training labels
        """
        self.stack_config = stack_config
        self.train = train
        self.test = test

    def stacking_data_loader(
        self,
        model: Union[XGBClassifier, CatBoostClassifier, LGBMClassifier],
        fold: int,
    ) -> Output(train_fold_pred=np.array, test_pred_mean=np.array):
        """Build the first level of the stacking model and outputs the predictions which will be used as features for the next level model.

        Args:
            model: the model to be used for the first level of the stacking model
            fold: the fold number
        """
        try:
            self.y = self.train.drop("Churn", axis=1)
            stk = StratifiedKFold(n_splits=fold, random_state=42, shuffle=True)

            train_fold_pred = np.zeros((self.train.shape[0], 1))
            test_pred = np.zeros((self.test.shape[0], fold))

            for counter, (train_index, valid_index) in enumerate(stk.split(self.train, self.y)):
                x_train, y_train = self.train.iloc[train_index], self.y[train_index]
                x_valid, y_valid = self.train.iloc[valid_index], self.y[valid_index]

                rprint("------------ Fold", counter + 1, "Start! ------------")
                if self.stack_config.model_name == "cat":
                    model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)])
                elif self.stack_config.model_name == "xgb":
                    model.fit(
                        x_train,
                        y_train,
                        eval_set=[(x_valid, y_valid)],
                        eval_metric="auc",
                        verbose=500,
                        early_stopping_rounds=200,
                    )
                else:
                    model.fit(
                        x_train,
                        y_train,
                        eval_set=[(x_valid, y_valid)],
                        eval_metric="auc",
                        verbose=500,
                        early_stopping_rounds=200,
                    )

                rprint("------------ Fold", counter + 1, "Done! ------------")

                train_fold_pred[valid_index, :] = model.predict_proba(x_valid)[:, 1].reshape(-1, 1)
                test_pred[:, counter] = model.predict_proba(self.test)[:, 1]

                del x_train, y_train, x_valid, y_valid
                gc.collect()

            test_pred_mean = np.mean(test_pred, axis=1).reshape(-1, 1)

            del test_pred
            gc.collect()
            logger("Done!")
            return train_fold_pred, test_pred_mean
        except Exception as e:
            logger(e)

    def stacking_model_builder(self) -> Output(stack_x_train=np.ndarray, stack_x_test=np.ndarray):
        try:
            lgb_params = {
                "objective": "binary",
                "n_estimators": 20000,
                "random_state": 42,
                "learning_rate": 8e-3,
                "subsample": 0.6,
                "subsample_freq": 1,
                "colsample_bytree": 0.4,
                "reg_alpha": 10.0,
                "reg_lambda": 1e-1,
                "min_child_weight": 256,
                "min_child_samples": 20,
                "device": "gpu",
            }

            xgb_params = {
                "n_estimators": 10000,
                "learning_rate": 0.03689407512484644,
                "max_depth": 8,
                "colsample_bytree": 0.3723914688159835,
                "subsample": 0.780714581166012,
                "eval_metric": "auc",
                "use_label_encoder": False,
                "gamma": 0,
                "reg_lambda": 50.0,
                "tree_method": "gpu_hist",
                "gpu_id": 0,
                "predictor": "gpu_predictor",
                "random_state": 42,
            }

            cat_params = {
                "iterations": 17298,
                "learning_rate": 0.03429054860458741,
                "reg_lambda": 0.3242286463210283,
                "subsample": 0.9433911589913944,
                "random_strength": 22.4849972385133,
                "depth": 8,
                "min_data_in_leaf": 4,
                "leaf_estimation_iterations": 8,
                "task_type": "GPU",
                "bootstrap_type": "Poisson",
                "verbose": 500,
                "early_stopping_rounds": 200,
                "eval_metric": "AUC",
            }

            lgbm = LGBMClassifier(**lgb_params)
            xgb = XGBClassifier(**xgb_params)
            cat = CatBoostClassifier(**cat_params)

            cat_train, cat_test = self.stacking_data_loader(cat, 5)
            del cat
            gc.collect()
            lgbm_train, lgbm_test = self.stacking_data_loader(lgbm, 5)
            del lgbm
            gc.collect()
            xgb_train, xgb_test = self.stacking_data_loader(xgb, 5)
            del xgb
            gc.collect()

            stack_x_train = np.concatenate((cat_train, lgbm_train, xgb_train), axis=1)
            stack_x_test = np.concatenate((cat_test, lgbm_test, xgb_test), axis=1)

            del cat_train, lgbm_train, xgb_train, cat_test, lgbm_test, xgb_test
            gc.collect()

            return stack_x_train, stack_x_test
        except Exception as e:
            logger(e)
