from typing import Any, Type, Union, List
import pickle
import os

from zenml.io import fileio
from zenml.materializers.base_materializer import BaseMaterializer
import pandas as pd

from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

DEFAULT_FILENAME = "CustomerSatisfactionEnvironment"


class cs_materializer(BaseMaterializer):
    ASSOCIATED_TYPES = [
        pd.Series,
        pd.DataFrame,
        CatBoostRegressor,
        RandomForestRegressor,
        LGBMRegressor,
        XGBRegressor,
    ]

    def handle_input(
        self, data_type: Type[Any]
    ) -> Union[
        pd.Series,
        pd.DataFrame,
        CatBoostRegressor,
        RandomForestRegressor,
        LGBMRegressor,
        XGBRegressor,
    ]:
        """Reads a base sklearn label encoder from a pickle file."""
        super().handle_input(data_type)
        filepath = os.path.join(self.artifact.uri, DEFAULT_FILENAME)
        with fileio.open(filepath, "rb") as fid:
            clf = pickle.load(fid)
        return clf

    def handle_return(
        self,
        clf: Union[
            pd.Series,
            pd.DataFrame,
            CatBoostRegressor,
            RandomForestRegressor,
            LGBMRegressor,
            XGBRegressor,
        ],
    ) -> None:
        """Creates a pickle for a sklearn label encoder.
        Args:
            clf: A sklearn label encoder.
        """
        super().handle_return(clf)
        filepath = os.path.join(self.artifact.uri, DEFAULT_FILENAME)
        with fileio.open(filepath, "wb") as fid:
            pickle.dump(clf, fid)
