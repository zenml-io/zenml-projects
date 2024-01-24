from typing import Tuple

import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAXResultsWrapper, SARIMAX
from typing_extensions import Annotated
from zenml import step


@step
def sarimax_trainer_step(
    data: np.ndarray,
    order: Tuple[int, int, int] = (1, 0, 1),
    seasonal_order: Tuple[int, int, int, int] = (
        1,
        1,
        1,
        12,
    ),  # Seasonal (P, D, Q, S)
    trend: str = "c",  # Trend parameter
) -> Annotated[SARIMAXResultsWrapper, "sarimax_statsmodel"]:
    """Train a SARIMAX model on the provided data."""
    # Instantiate and fit the SARIMAX model
    model = SARIMAX(
        data, order=order, seasonal_order=seasonal_order, trend=trend
    )
    res_mle  = model.fit(disp=False)
    
    print(res_mle.summary())

    # Return the fitted model
    return res_mle 
