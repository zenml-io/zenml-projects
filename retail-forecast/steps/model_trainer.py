from typing import Dict, List

import pandas as pd
from prophet import Prophet
from zenml import step


@step
def train_model(
    train_data_dict: Dict[str, pd.DataFrame],
    series_ids: List[str],
    weekly_seasonality: bool = True,
    yearly_seasonality: bool = False,
    daily_seasonality: bool = False,
    seasonality_mode: str = "multiplicative",
) -> Dict[str, Prophet]:
    """
    Train a Prophet model for each store-item combination.

    Args:
        train_data_dict: Dictionary with training data for each series
        series_ids: List of series identifiers
        weekly_seasonality: Whether to include weekly seasonality
        yearly_seasonality: Whether to include yearly seasonality
        daily_seasonality: Whether to include daily seasonality
        seasonality_mode: 'additive' or 'multiplicative'

    Returns:
        Dictionary of trained Prophet models for each series
    """
    models = {}

    for series_id in series_ids:
        print(f"Training model for {series_id}...")
        train_data = train_data_dict[series_id]

        # Initialize Prophet model
        model = Prophet(
            weekly_seasonality=weekly_seasonality,
            yearly_seasonality=yearly_seasonality,
            daily_seasonality=daily_seasonality,
            seasonality_mode=seasonality_mode,
        )

        # Fit model
        model.fit(train_data)

        # Store trained model
        models[series_id] = model

    print(f"Successfully trained {len(models)} Prophet models")

    return models
