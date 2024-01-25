from typing import Tuple

import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAXResultsWrapper, SARIMAX
from typing_extensions import Annotated
from zenml import step, ArtifactConfig


@step
def sarimax_inference_step(df: pd.DataFrame, model: SARIMAXResultsWrapper) -> Annotated[pd.DataFrame, "next_day_predictions"]:
    """Train a SARIMAX model on the provided data."""
    # Make predictions 
    pred = model.predict(len(df), len(df) + 287)
    return res_mle
