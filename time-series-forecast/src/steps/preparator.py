from zenml.steps import step, Output
import pandas as pd

@step
def preparator(data: pd.DataFrame) -> Output(
    df = pd.DataFrame
):
    """Cleans and prepares the dataset.

    Args:
        data: DataFrame with training feature data and target data.

    Returns:
        pd.DataFrame
    """

    df = data.drop(['Source_time','Lead_hours','ANM','Non_ANM', 'int64_field_0'],axis=1)
    df = df[df['Direction'].notna()]
    df = df[df['Total'].notna()]
    df['Speed'] =  df['Speed'].fillna(df['Speed'].median())

    return df
