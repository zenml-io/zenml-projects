import os
import pickle
from typing import Any, Type, Union

import pandas as pd
from steps.src.log_reg import LogisticRegression
from zenml.io import fileio
from zenml.materializers.base_materializer import BaseMaterializer

DEFAULT_FILENAME = "CustomerChurnEnvironment"

import numpy as np


class cs_materializer(BaseMaterializer):
    """
    Custom materializer for the Customer Satisfaction Zenfile
    """

    ASSOCIATED_TYPES = [
        np.float64,
        pd.Series,
        LogisticRegression,
    ]

    def handle_input(
        self, data_type: Type[Any]
    ) -> Union[np.float64, pd.Series, LogisticRegression]:
        """
        It loads the model from the artifact and returns it.

        Args:
            data_type: The type of the model to be loaded
        """
        super().handle_input(data_type)
        filepath = os.path.join(self.artifact.uri, DEFAULT_FILENAME)
        with fileio.open(filepath, "rb") as fid:
            obj = pickle.load(fid)
        return obj

    def handle_return(
        self,
        obj: Union[np.float64, pd.Series, LogisticRegression],
    ) -> None:
        """
        It saves the model to the artifact store.

        Args:
            model: The model to be saved
        """

        super().handle_return(obj)
        filepath = os.path.join(self.artifact.uri, DEFAULT_FILENAME)
        with fileio.open(filepath, "wb") as fid:
            pickle.dump(obj, fid)
