from typing import Tuple

import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAXResultsWrapper, SARIMAX
from typing_extensions import Annotated
from zenml import step, ArtifactConfig
from materializers.statsmodel_materializer import StatsmodelMaterializer


@step(output_materializers=StatsmodelMaterializer)
def sarimax_trainer_step(
    data: pd.DataFrame,
    order: Tuple[int, int, int] = (1, 1, 1),
    seasonal_order: Tuple[int, int, int, int] = (
        1,
        1,
        1,
        288,
    ),  # Seasonal (P, D, Q, S)
    trend: str = "c",  # Trend parameter
) -> Annotated[
    SARIMAXResultsWrapper,
    ArtifactConfig(name="sarimax_statsmodel", is_model_artifact=True),
]:
    """Train a SARIMAX model on the provided data."""
    # Instantiate and fit the SARIMAX model
    model = SARIMAX(
        data, order=order, seasonal_order=seasonal_order, trend=trend
    )
    res_mle = model.fit(disp=False)

    # Return the fitted model
    return res_mle
