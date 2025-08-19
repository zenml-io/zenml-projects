"""
Batch inference step for FloraCast using ZenML Model Control Plane.
"""

from typing import Annotated
import pandas as pd
from darts import TimeSeries
from zenml import step, get_step_context, log_metadata
from zenml.logger import get_logger
from zenml.client import Client

logger = get_logger(__name__)


@step
def batch_inference_predict(
    series: TimeSeries,
    horizon: int = 14,
) -> Annotated[pd.DataFrame, "predictions"]:
    """
    Perform batch inference using the trained model from Model Control Plane.

    Args:
        series: Time series data for forecasting
        horizon: Number of time steps to forecast

    Returns:
        DataFrame containing forecast results with columns ['ds', 'yhat']
    """
    logger.info(f"Performing batch inference with horizon: {horizon}")

    try:
        # Get the model from Model Control Plane
        context = get_step_context()
        if not context.model:
            raise ValueError(
                "No model found in step context. Make sure to run training first or specify model version in config."
            )

        logger.info(
            f"Using model: {context.model.name}, version: {context.model.version}"
        )

        # Try to get the trained model artifact
        try:
            model_artifact = context.model.get_artifact("trained_model")
            if model_artifact is None:
                raise ValueError(
                    "trained_model artifact not found in model version"
                )
        except Exception as e:
            logger.error(f"Failed to get trained_model artifact: {e}")
            # List all available artifacts for debugging
            try:
                logger.info("Available artifacts in model version:")
                # Use the correct method to get model version artifacts
                client = Client()
                model_version = client.get_model_version(
                    model_name_or_id=context.model.name,
                    model_version_name_or_number_or_id=context.model.version,
                )
                artifacts = model_version.model_artifacts
                for artifact in artifacts:
                    logger.info(f"  - {artifact.name}")
            except Exception as list_error:
                logger.warning(f"Could not list artifacts: {list_error}")
            raise

        # Load the trained model
        trained_model = model_artifact.load()
        logger.info(
            f"Loaded model from Model Control Plane: {type(trained_model).__name__}"
        )

        # Generate predictions using improved multi-step approach (same as evaluation)
        logger.info(
            f"Using iterative multi-step prediction for horizon={horizon}"
        )

        # Use multiple prediction steps for better long-term accuracy
        predictions_list = []
        context_series = series

        # Predict in chunks of output_chunk_length (14 days)
        remaining_steps = horizon
        while remaining_steps > 0:
            chunk_size = min(
                14, remaining_steps
            )  # Model's output_chunk_length
            chunk_pred = trained_model.predict(
                n=chunk_size, series=context_series
            )
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

        # Convert to DataFrame
        pred_df = predictions.pd_dataframe().reset_index()
        pred_df.columns = ["ds", "yhat"]  # Standard naming

        logger.info(f"Created predictions DataFrame with {len(pred_df)} rows")
        logger.info(
            f"Forecast period: {pred_df['ds'].min()} to {pred_df['ds'].max()}"
        )
        logger.info(
            f"Prediction stats: mean={pred_df['yhat'].mean():.2f}, std={pred_df['yhat'].std():.2f}"
        )

        # Log metadata to ZenML for observability
        log_metadata(
            {
                "horizon": horizon,
                "num_predictions": len(pred_df),
                "forecast_start": str(pred_df["ds"].min()),
                "forecast_end": str(pred_df["ds"].max()),
                "prediction_mean": float(pred_df["yhat"].mean()),
                "prediction_std": float(pred_df["yhat"].std()),
                "prediction_min": float(pred_df["yhat"].min()),
                "prediction_max": float(pred_df["yhat"].max()),
                "model_type": type(trained_model).__name__,
                "model_name": context.model.name,
                "model_version": context.model.version,
            }
        )

        return pred_df

    except Exception as e:
        logger.error(f"Batch inference failed: {str(e)}")
        raise
