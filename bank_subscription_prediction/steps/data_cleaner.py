import pandas as pd
from zenml import step

@step
def clean_data_step(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans the input DataFrame.

    Args:
        df: Pandas DataFrame to clean.

    Returns:
        Cleaned Pandas DataFrame.
    """
    # Drop rows with missing data
    data = df.dropna()

    # Convert the 'day' column type to object as 'day' is categorical
    # Ensure the column exists before trying to modify it
    if 'day' in data.columns:
        data['day'] = data['day'].astype('object')
    else:
        # Handle the case where 'day' column might be missing, perhaps log a warning
        # For now, we'll proceed without this conversion if the column isn't there
        pass 

    return data 