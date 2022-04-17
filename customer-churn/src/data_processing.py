import pandas as pd
from feature_engine.encoding import MeanEncoder
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import LabelEncoder
from zenml.logger import get_logger

logger = get_logger(__name__)
from zenml.steps import Output, step


def encode_categorical_columns(data: pd.DataFrame) -> Output(data=pd.DataFrame):
    """
    Encode categorical columns to numeric values using LabelEncoder.

    Args:
        data (pd.DataFrame): DataFrame containing categorical columns.
    """
    try:
        for col in data.columns:
            if data[col].dtype == "O":
                le = LabelEncoder()
                data[col] = le.fit_transform(data[col])
        return data

    except ValueError:
        logger.error(
            "Categorical columns encoding failed due to not matching the type of the input data, Recheck the type of your input data."
        )
        raise ValueError
    except Exception as e:
        logger.error(e)


def mean_encoding(data: pd.DataFrame) -> Output(data=pd.DataFrame):
    """
    Mean encoding of categorical columns. Mean encoding is a technique that is used to convert categorical values to numeric values.

    Args:
        data (pd.DataFrame): DataFrame
    """
    try:
        cat_col = []
        for col in data.columns:
            if data[col].dtype == "O":
                cat_col.append(col)
        X = data.drop("y", axis=1)
        y = data["y"]
        encoder = MeanEncoder(
            variables=cat_col,
            ignore_format=True,
        )
        encoder.fit(X, y)
        return data
    except ValueError:
        logger.error(
            "Mean encoding failed due to not matching the type of the input data, Recheck the type of your input data."
        )
        raise ValueError

    except Exception as e:
        logger.error(e)


def handle_imbalanced_data(
    data: pd.DataFrame,
) -> Output(balanced_data=pd.DataFrame, pipeline=Pipeline):
    """
    Handle imbalanced data by combining SMOTE with random undersampling of the majority class.

    Args:
        data (pd.DataFrame): DataFrame
    """
    try:
        X = data.drop("y", axis=1)
        y = data["y"]
        over = SMOTE(sampling_strategy=0.1)
        under = RandomUnderSampler(sampling_strategy=0.5)
        steps = [("o", over), ("u", under)]
        pipeline = Pipeline(steps=steps)
        X_res, y_res = pipeline.fit_resample(X, y)
        balanced_data = pd.concat([X_res, y_res], axis=1)
        return balanced_data, pipeline
    except ValueError:
        logger.error(
            "Imbalanced data handling failed due to not matching the type of the input data, Recheck the type of your input data."
        )
        raise ValueError

    except Exception as e:
        logger.error(e)
