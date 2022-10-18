import pandas as pd
from typing import List
from sklearn import preprocessing

from zenml.steps import step
from sklearn.base import RegressorMixin

from .utils import get_label_encoder


@step
def predictor(
    model: RegressorMixin,
    data: pd.DataFrame,
    le_seasons: preprocessing.LabelEncoder,
) -> pd.DataFrame:
    """Runs predictions on next weeks NBA matches.

    Args:
        model: A sklearn regression model (e.g. RandomForestRegressor).
        data: Dataframe with data to predict on.
        le_seasons: Previously fitted LabelEncoder to encode season IDs.

    Returns:
        pd.DataFrame: [description]
    """
    feature_cols = model.feature_names_in_

    data = data[feature_cols]
    predicted_y = model.predict(data)
    data["PREDICTION"] = predicted_y

    data["SEASON_ID"] = le_seasons.inverse_transform(data["SEASON_ID"])

    return data
