import logging

import pandas as pd
from feature_engine.encoding import MeanEncoder
from sklearn.preprocessing import LabelEncoder
from zenml.logger import get_logger

logger = get_logger(__name__)
from zenml.steps import Output

from .utils import unique_data_detector


class DataProcessor:
    def __init__(self) -> None:
        """Initialize the DataProcessor class."""
        pass

    def encode_categorical_columns(self, data: pd.DataFrame) -> Output(data=pd.DataFrame):
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

    def mean_encoding(self, data: pd.DataFrame) -> Output(data=pd.DataFrame):
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

    # def handle_imbalanced_data(
    #     self,
    #     data: pd.DataFrame,
    # ) -> Output(balanced_data=pd.DataFrame):
    #     """
    #     Handle imbalanced data by combining SMOTE with random undersampling of the majority class.

    #     Args:
    #         data (pd.DataFrame): DataFrame
    #     """
    #     try:
    #         X = data.drop("Churn", axis=1)
    #         y = data["Churn"]
    #         over = SMOTE(sampling_strategy=0.5)
    #         under = RandomUnderSampler(sampling_strategy=0.5)
    #         steps = [("o", over), ("u", under)]
    #         pipeline = Pipeline(steps=steps)
    #         X_res, y_res = pipeline.fit_resample(X, y)
    #         balanced_data = pd.concat([X_res, y_res], axis=1)
    #         return balanced_data
    #     except ValueError:
    #         logger.error(
    #             "Imbalanced data handling failed due to not matching the type of the input data, Recheck the type of your input data."
    #         )
    #         raise ValueError

    #     except Exception as e:
    #         logger.error(e)

    def drop_columns(self, data: pd.DataFrame) -> Output(output_data=pd.DataFrame):
        """
        Drop columns from the dataframe by using several methods.

        Args:
            data (pd.DataFrame): DataFrame
        """
        try:
            if data.empty:
                logging.error("Dataframe is empty.")
            data = self.single_value_column_remover(data)
            data = self.handle_missing_values(data)
            return data

        except ValueError:
            logger.error(
                "Drop columns failed due to not matching the type of the input data, Recheck the type of your input data."
            )
            raise ValueError

        except Exception as e:
            logger.error(e)

    def single_value_column_remover(self, data: pd.DataFrame) -> pd.DataFrame:
        """Removes columns with a single value.

        Args:
            data (pd.DataFrame): Dataframe from which single column to be removed.
            threshold (int): Threshold for removing columns.

        Returns:
            pd.DataFrame: Dataframe with columns removed.
        """
        try:
            n_uniques = unique_data_detector(data)
            n_uniques = n_uniques[n_uniques <= 1]
            if n_uniques.empty:
                print(f"No columns with a single value were found.")
                return data
            else:
                data.drop(
                    n_uniques.index, axis=1, inplace=True
                )  # here we are dropping columns which have single value in them (i.e. we are removing columns with only one unique value).
                return data
        except ValueError:
            logger.error(
                "Data must be a DataFrame, Ensure the data which you're passing is a DataFrame."
            )
            raise ValueError
        except Exception as e:
            logger.error(e)

    def handle_missing_values(self, data: pd.DataFrame) -> Output(data=pd.DataFrame):
        """
        Handle missing values by filling them with mean values.

        Args:
            data (pd.DataFrame): DataFrame
        """
        try:
            strategy: str = "mean"
            if strategy == "mean":
                data = data.fillna(data.mean())
                return data
            else:
                raise ValueError
        except ValueError:
            logger.error(
                "data must be a DataFrame, Ensure the data which you're passing is a DataFrame."
            )
            raise ValueError
        except Exception as e:
            logger.error(e)
