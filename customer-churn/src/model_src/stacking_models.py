import gc
from ctypes import Union

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from zenml.logger import get_logger
from zenml.steps import BaseStepConfig, Output

logger = get_logger(__name__)
from rich import print as rprint


class StackedEnsembles:
    def __init__(
        self, stack_config: BaseStepConfig, train: pd.DataFrame, test: pd.DataFrame, y: pd.Series
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
        self.y = y

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

    # def stacking_model_builder(self):
    #     cat_train, cat_test = Stacking_Data_Loader(cat 5)
    #     del cat
    #     gc.collect()

    #     lgbm_train, lgbm_test = Stacking_Data_Loader(lgbm 5)
    #     del lgbm
    #     gc.collect()

    #     xgb_train, xgb_test = Stacking_Data_Loader(xgb, 5)
    #     del xgb
    #     gc.collect()
