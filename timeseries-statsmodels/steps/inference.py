from typing import Tuple

import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAXResultsWrapper
from typing_extensions import Annotated
from zenml import step
from materializers.statsmodelpredictions_materializer import (
    StatsmodelPredictionMaterializer,
)


@step(output_materializers=StatsmodelPredictionMaterializer, enable_cache=False)
def sarimax_inference_step(
    df: pd.DataFrame, model: SARIMAXResultsWrapper
) -> Annotated[pd.DataFrame, "next_day_predictions"]:
    """Train a SARIMAX model on the provided data."""
    # Make predictions
    pred = model.get_forecast(steps=288).predicted_mean
    new_df = pd.DataFrame()
    new_df["DATA"] = df 
    new_df["PRED"] = pred
    return new_df
