import pandas as pd
from zenml import step
from sklearn.model_selection import train_test_split
from typing import Tuple, Annotated


@step
def split_data_step(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify_col: str = "y",  # Stratify by target variable by default
) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"],
]:
    """Splits the data into training and testing sets.

    Args:
        df: Preprocessed Pandas DataFrame with features and target.
        test_size: Proportion of the dataset to include in the test split.
        random_state: Controls the shuffling applied to the data before splitting.
        stratify_col: Column to use for stratified splitting. Defaults to 'y'.

    Returns:
        A tuple containing X_train, X_test, y_train, y_test.
    """
    if "y" not in df.columns:
        raise ValueError("Target column 'y' not found in DataFrame.")

    X = df.drop("y", axis=1)
    y = df["y"]

    stratify_data = None
    if stratify_col and stratify_col in df.columns:
        stratify_data = df[stratify_col]
    elif stratify_col:
        print(
            f"Warning: Stratification column '{stratify_col}' not found. Proceeding without stratification."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_data,
    )
    return X_train, X_test, y_train, y_test
