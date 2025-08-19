"""
Model evaluation step for FloraCast.
"""

from typing import Annotated
from darts import TimeSeries
from zenml import step, log_metadata
from zenml.logger import get_logger

from utils.metrics import smape

logger = get_logger(__name__)


@step
def evaluate(
    model: object,
    train_series: TimeSeries,
    val_series: TimeSeries,
    horizon: int = 7,
    metric: str = "smape",
) -> Annotated[float, "evaluation_score"]:
    """
    Evaluate the trained model on validation data.

    Args:
        model: Trained forecasting model
        train_series: Training time series
        val_series: Validation time series
        horizon: Forecasting horizon
        metric: Evaluation metric name

    Returns:
        Evaluation metric score (lower is better for SMAPE)
    """

    logger.info(f"Evaluating model with horizon={horizon}, metric={metric}")

    try:
        # Generate predictions using TFT model
        # TFT requires the series parameter to generate predictions
        logger.info(f"Generating predictions for horizon {horizon}")

        # For TFT models, we need to provide the series parameter
        if hasattr(model, "predict"):
            predictions = model.predict(
                n=min(horizon, len(val_series)), series=train_series
            )
            logger.info(f"Generated {len(predictions)} predictions")

            # Truncate validation series to match prediction length
            actual = val_series[: len(predictions)]

            # Calculate metric
            if metric == "smape":
                score = smape(actual, predictions)
            else:
                raise ValueError(f"Unknown metric: {metric}")

            logger.info(f"Evaluation {metric}: {score:.4f}")

            # Log metadata to ZenML for observability
            log_metadata(
                {
                    "evaluation_metric": metric,
                    "score": float(score),
                    "horizon": horizon,
                    "num_predictions": len(predictions),
                    "actual_length": len(actual),
                    "model_type": type(model).__name__,
                }
            )

            return float(score)
        else:
            logger.error("Model does not have predict method")
            return 9999.0

    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        logger.info("This might be due to TFT model prediction requirements")
        # Return a high penalty score for failed evaluation
        return 9999.0
