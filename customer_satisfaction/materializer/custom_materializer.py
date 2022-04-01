from typing import Any, Type, Union, List
import pickle
import os

from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

import numpy as np 
import pandas as pd

from zenml.io import fileio
from zenml.materializers.base_materializer import BaseMaterializer



DEFAULT_FILENAME = "CustomerSatisfactionEnvironment"

class cs_materializer(BaseMaterializer):
    '''
    Custom materializer for the Customer Satisfaction Zenfile 
    '''
    ASSOCIATED_TYPES = [
        str,
        np.ndarray,
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
        str,
        np.ndarray,
        pd.Series,
        pd.DataFrame,
        CatBoostRegressor,
        RandomForestRegressor,
        LGBMRegressor,
        XGBRegressor,
    ]:
        """
        It loads the model from the artifact and returns it. 

        Args: 
            data_type: The type of the model to be loaded
        """
        super().handle_input(data_type)
        filepath = os.path.join(self.artifact.uri, DEFAULT_FILENAME)
        with fileio.open(filepath, "rb") as fid:
            clf = pickle.load(fid)
        return clf

    def handle_return(
        self,
        clf: Union[ 
            str,
            np.ndarray,
            pd.Series,
            pd.DataFrame,
            CatBoostRegressor,
            RandomForestRegressor,
            LGBMRegressor,
            XGBRegressor,
        ],
    ) -> None:
        """
        It saves the model to the artifact store.
        
        Args:
            clf: The model to be saved
        """

        super().handle_return(clf)
        filepath = os.path.join(self.artifact.uri, DEFAULT_FILENAME)
        with fileio.open(filepath, "wb") as fid:
            pickle.dump(clf, fid)
