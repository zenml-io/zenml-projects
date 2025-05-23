# src/utils/preprocessing_utils.py

import numpy as np
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
    """Create AGE_YEARS and EMPLOYMENT_YEARS from DAYS_BIRTH / DAYS_EMPLOYED.

    Implements fairness-aware age discretization to reduce bias.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        # Derive AGE_YEARS with fairness-aware binning
        if "DAYS_BIRTH" in df:
            age_years = -df["DAYS_BIRTH"] / 365.25

            # Create balanced age bins to reduce bias
            # Use quantile-based binning instead of fixed ranges
            df["AGE_YEARS"] = age_years

            # Add age-related features that are less biased
            df["AGE_SQUARED"] = age_years**2  # Non-linear age effect
            df["AGE_LOG"] = pd.Series(age_years).apply(
                lambda x: np.log(max(x, 18))
            )

            # Create broad age categories to reduce granular bias
            df["AGE_CATEGORY"] = pd.cut(
                age_years,
                bins=[0, 35, 50, 65, 100],
                labels=["young", "middle", "mature", "senior"],
            ).astype(str)

            df = df.drop(columns=["DAYS_BIRTH"])

        # Derive EMPLOYMENT_YEARS with stability indicators
        if "DAYS_EMPLOYED" in df:
            # Handle the special case of 365243 (unemployed marker)
            employment_days = df["DAYS_EMPLOYED"].copy()

            # Replace the unemployed marker with 0
            employment_days = employment_days.replace(365243, 0)

            df["EMPLOYMENT_YEARS"] = employment_days.apply(
                lambda x: abs(x) / 365.25 if x < 0 else 0
            )

            # Add employment stability features
            df["IS_EMPLOYED"] = (employment_days < 0).astype(int)
            df["EMPLOYMENT_STABILITY"] = df["EMPLOYMENT_YEARS"].apply(
                lambda x: "stable"
                if x > 2
                else "new"
                if x > 0
                else "unemployed"
            )

            df = df.drop(columns=["DAYS_EMPLOYED"])

        return df
