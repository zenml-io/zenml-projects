"""Utility functions for model prediction."""

from darts import TimeSeries
from zenml.logger import get_logger

logger = get_logger(__name__)


def iterative_predict(
    model,
    series: TimeSeries,
    horizon: int,
    model_output_chunk_length: int = None,
) -> TimeSeries:
    """Generate predictions using iterative multi-step approach for better long-term accuracy.

    Args:
        model: Trained forecasting model
        series: Input time series data for context
        horizon: Total number of time steps to forecast
        model_output_chunk_length: Model's output chunk length. If None, tries to get from model

    Returns:
        TimeSeries containing all predictions
    """
    logger.info(f"Using iterative multi-step prediction for horizon={horizon}")

    # Try to get output_chunk_length from model if not provided
    if model_output_chunk_length is None:
        if hasattr(model, "output_chunk_length"):
            model_output_chunk_length = model.output_chunk_length
        else:
            logger.warning(
                "Could not determine model output_chunk_length, defaulting to 7"
            )
            model_output_chunk_length = 7

    logger.info(
        f"Using model output_chunk_length: {model_output_chunk_length}"
    )

    # Use multiple prediction steps for better long-term accuracy
    predictions_list = []
    context_series = series

    # Predict in chunks of output_chunk_length
    remaining_steps = horizon
    while remaining_steps > 0:
        chunk_size = min(model_output_chunk_length, remaining_steps)

        # Generate prediction chunk
        chunk_pred = model.predict(n=chunk_size, series=context_series)

        predictions_list.append(chunk_pred)

        # Extend context with the prediction for next iteration
        context_series = context_series.concatenate(chunk_pred)
        remaining_steps -= chunk_size

    # Combine all predictions
    if len(predictions_list) == 1:
        predictions = predictions_list[0]
    else:
        predictions = predictions_list[0]
        for pred_chunk in predictions_list[1:]:
            predictions = predictions.concatenate(pred_chunk)

    logger.info(
        f"Generated {len(predictions)} predictions using multi-step approach"
    )
    return predictions
