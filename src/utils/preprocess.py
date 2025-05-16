# src/utils/preprocessing_utils.py

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class DropIDColumn(BaseEstimator, TransformerMixin):
    """Sklearn transformer to drop ID column."""

    def __init__(self, column: str = "SK_ID_CURR"):
        self.column = column

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.drop(columns=[self.column], errors="ignore")


class DeriveAgeFeatures(BaseEstimator, TransformerMixin):
    """Create AGE_YEARS and EMPLOYMENT_YEARS from DAYS_BIRTH / DAYS_EMPLOYED."""

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        # Derive AGE_YEARS and drop original column
        if "DAYS_BIRTH" in df:
            df["AGE_YEARS"] = -df["DAYS_BIRTH"] / 365.25
            df = df.drop(columns=["DAYS_BIRTH"])

        # Derive EMPLOYMENT_YEARS and drop original column
        if "DAYS_EMPLOYED" in df:
            df["EMPLOYMENT_YEARS"] = df["DAYS_EMPLOYED"].apply(
                lambda x: abs(x) / 365.25 if x < 0 else 0
            )
            df = df.drop(columns=["DAYS_EMPLOYED"])

        return df
