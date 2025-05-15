import pandas as pd
from zenml import step
from typing import List

@step
def preprocess_data_step(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses the data for model training.

    Args:
        df: Cleaned Pandas DataFrame.

    Returns:
        Preprocessed Pandas DataFrame.
    """
    data_processed = df.copy()

    # Drop the 'duration' column as it's not known before a call
    if 'duration' in data_processed.columns:
        data_processed = data_processed.drop('duration', axis=1)

    # Identify categorical columns (excluding the target 'y')
    categorical_cols: List[str] = data_processed.select_dtypes(include=['object']).columns.tolist()
    if 'y' in categorical_cols:
        categorical_cols.remove('y') # Exclude target from dummification for now
    
    # Convert categorical variables to dummy variables
    # The notebook uses pd.get_dummies which handles 'unknown' as a category
    if categorical_cols:
        data_processed = pd.get_dummies(data_processed, columns=categorical_cols, drop_first=False)

    # Convert target variable 'y' to numeric (1 for 'yes', 0 for 'no')
    if 'y' in data_processed.columns:
        data_processed['y'] = data_processed['y'].map({'yes': 1, 'no': 0})
        # Ensure it's integer type after mapping if there were no NaNs introduced by unmapped values
        if not data_processed['y'].isnull().any():
             data_processed['y'] = data_processed['y'].astype(int)
        else:
            # Handle cases where 'y' might have values other than 'yes' or 'no'
            # For now, we'll let them be NaNs, which might need further handling or raise an error
            # Depending on requirements, one might want to fill NaNs or raise an error here.
            print("Warning: 'y' column contains values other than 'yes' or 'no', resulting in NaNs after mapping.")

    return data_processed 