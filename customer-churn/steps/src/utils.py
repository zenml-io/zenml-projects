import pandas as pd


def unique_data_detector(data: pd.DataFrame) -> pd.Series:
    """Detects unique values in a dataframe.

    Args:
        data (pd.DataFrame): Dataframe to be analyzed.

    Returns:
        pd.Series: Series of unique values in the dataframe.
    """
    try:
        n_uniques = data.nunique()
        return n_uniques
    except:
        raise ValueError(
            "Data must be a dataframe or data has some issues. Make sure you have checked your data."
        )
