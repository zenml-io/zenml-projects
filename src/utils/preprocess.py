# src/utils/preprocessing_utils.py
from datetime import datetime
from typing import List

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


class SimpleScaler(BaseEstimator, TransformerMixin):
    """Sklearn transformer to scale numeric columns."""

    def __init__(self, exclude: List[str]):
        from sklearn.preprocessing import StandardScaler

        self.exclude = exclude
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        to_scale = [
            c for c in X.columns if c not in self.exclude and pd.api.types.is_numeric_dtype(X[c])
        ]
        self.scaler.fit(X[to_scale])
        self._cols = to_scale
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        df[self._cols] = self.scaler.transform(df[self._cols])
        return df


class DeriveAgeFeatures(BaseEstimator, TransformerMixin):
    """Create AGE_YEARS and EMPLOYMENT_YEARS from DAYS_BIRTH / DAYS_EMPLOYED."""

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        if "DAYS_BIRTH" in df:
            df["AGE_YEARS"] = -df["DAYS_BIRTH"] / 365.25
        if "DAYS_EMPLOYED" in df:
            df["EMPLOYMENT_YEARS"] = df["DAYS_EMPLOYED"].apply(
                lambda x: abs(x) / 365.25 if x < 0 else 0
            )
        return df
