import pandas as pd
import numpy as np

from sklearn.base import RegressorMixin

from zenml.steps.step_output import Output
from zenml.steps import step


@step(enable_cache=False)
def tester(
    model: RegressorMixin, test_df_x: pd.DataFrame, test_df_y: pd.DataFrame
) -> Output(mae=float):
    """Test trained model on test set

    Args:
        model: Trained model
        test_df_x: Test dataset
        test_df_y: Ground truth for test dataset

    Returns:
        mae: Mean average error
    """
    predicted_y = model.predict(test_df_x)

    # Calculate the absolute errors
    errors = abs(predicted_y - np.squeeze(test_df_y.values.T))
    mae = round(np.mean(errors), 2)
    # Print out the mean absolute error (mae)
    print("Mean Absolute Error:", round(np.mean(errors), 2), "three pointers")

    return mae
