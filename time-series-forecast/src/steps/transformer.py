import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from zenml.steps import Output, step


@step
def transformer(
    data: pd.DataFrame,
) -> Output(
    X_train=np.ndarray,
    X_test=np.ndarray,
    y_train=np.ndarray,
    y_test=np.ndarray,
):
    """Feature engineering by transforming cardinal directions into 2-dimensional feature vectors.

    First convert the cardinal directions (North, South, etc.) into degrees then calculate
    cosine and sine values using a unit circle, these values make your vector [x1,x2] respectivaly
    then to determine the magnitude of these new wind-direction-vectors, multiply them by the wind speed

    Args:
        data: DataFrame with training feature data and training target data

    Returns:
        X_train:np.array, X_test:np.array, y_train:np.array, y_test:np.array
    """
    df = data.copy()
    cardinal_directions = {
        "N": 0,
        "NNE": 22.5,
        "NE": 45,
        "ENE": 67.5,
        "E": 90,
        "ESE": 112.5,
        "SE": 135,
        "SSE": 157.5,
        "S": 180,
        "SSW": 202.5,
        "SW": 225,
        "WSW": 247.5,
        "W": 270,
        "WNW": 292.5,
        "NW": 315,
        "NNW": 337.5,
    }

    for direction in cardinal_directions:
        df.loc[df["Direction"] == direction, "Direction"] = cardinal_directions[
            direction
        ]

    df["Direction"] = df["Direction"].astype(float)
    df["v1"] = df["Speed"] * np.cos(np.deg2rad(np.array(df["Direction"])))
    df["v2"] = df["Speed"] * np.sin(np.deg2rad(np.array(df["Direction"])))

    df = df.drop(["Direction", "Speed"], axis=1)

    X = np.array(df[["v1", "v2"]])
    y = np.array(df["Total"])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    return X_train, X_test, y_train, y_test
