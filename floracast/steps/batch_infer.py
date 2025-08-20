"""Batch inference step for FloraCast using ZenML Model Control Plane."""

from typing import Annotated, Tuple

import numpy as np
import pandas as pd
from darts import TimeSeries
from materializers.timeseries_materializer import DartsTimeSeriesMaterializer
from utils.prediction import iterative_predict
from zenml import get_step_context, log_metadata, step
from zenml.client import Client
from zenml.logger import get_logger

logger = get_logger(__name__)


@step(
    output_materializers={
        "prediction_series": DartsTimeSeriesMaterializer,
    }
)
def batch_inference_predict(
    df: pd.DataFrame,
    datetime_col: str = "ds",
    target_col: str = "y",
    freq: str = "D",
    horizon: int = 14,
) -> Tuple[
    Annotated[pd.DataFrame, "predictions"],
    Annotated[TimeSeries, "prediction_series"],
]:
    """Perform batch inference using the trained model from Model Control Plane.

    Args:
        df: Raw DataFrame with datetime and target columns
        datetime_col: Name of datetime column
        target_col: Name of target column
        freq: Frequency string for time series
        horizon: Number of time steps to forecast

    Returns:
        DataFrame containing forecast results with columns ['ds', 'yhat']
        TimeSeries containing the forecast results
    """
    logger.info(f"Performing batch inference with horizon: {horizon}")

    try:
        # Convert DataFrame to TimeSeries and cast to float32 for consistency
        logger.info("Converting DataFrame to TimeSeries")
        series = TimeSeries.from_dataframe(
            df, time_col=datetime_col, value_cols=target_col, freq=freq
        ).astype(np.float32)

        logger.info(f"Created TimeSeries with {len(series)} points")
        logger.info(
            f"Series range: {series.start_time()} to {series.end_time()}"
        )

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

        # Load the fitted scaler artifact
        fitted_scaler = None
        try:
            scaler_artifact = context.model.get_artifact("fitted_scaler")
            if scaler_artifact is None:
                raise ValueError(
                    "fitted_scaler artifact not found in model version"
                )
            fitted_scaler = scaler_artifact.load()
            logger.info("Loaded fitted scaler artifact from model version")

            # Apply scaling to the input series
            logger.info("Applying scaling to input series for inference")
            series = fitted_scaler.transform(series)
            logger.info("Scaling applied successfully")
        except Exception as scaler_error:
            logger.error(f"Failed to load or apply scaler: {scaler_error}")
            logger.warning(
                "Proceeding without scaling - predictions may be incorrect!"
            )
            # Continue without scaling for backward compatibility

        # Generate predictions using improved multi-step approach
        predictions = iterative_predict(trained_model, series, horizon)

        # Inverse transform predictions back to original scale
        if fitted_scaler is not None:
            try:
                logger.info(
                    "Inverse transforming predictions back to original scale"
                )
                predictions = fitted_scaler.inverse_transform(predictions)
                logger.info("Inverse transformation applied successfully")
            except Exception as inverse_error:
                logger.error(
                    f"Failed to inverse transform predictions: {inverse_error}"
                )
                logger.warning("Predictions remain in scaled format!")
        else:
            logger.warning(
                "No scaler available - predictions remain in original format"
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

        return pred_df, predictions

    except Exception as e:
        logger.error(f"Batch inference failed: {str(e)}")
        raise
