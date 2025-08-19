"""
Batch inference step for FloraCast using ZenML Model Control Plane.
"""

import os
from typing import Annotated
from darts import TimeSeries
from zenml import step, get_step_context, log_metadata
from zenml.logger import get_logger
from materializers import TFTModelMaterializer  # Import to register the materializer

logger = get_logger(__name__)


@step
def batch_inference_predict(
    series: Annotated[TimeSeries, "inference_series"],
    horizon: int = 14,
    output_path: str = "outputs/forecast.csv"
) -> Annotated[str, "predictions_path"]:
    """
    Perform batch inference using the trained model from Model Control Plane.
    
    Args:
        series: Time series data for forecasting
        horizon: Number of time steps to forecast
        output_path: Path to save forecast results
        
    Returns:
        Path to saved forecast results
    """
    logger.info(f"Performing batch inference with horizon: {horizon}")
    
    try:
        # Get the model from Model Control Plane
        context = get_step_context()
        if not context.model:
            raise ValueError("No model found in step context. Make sure to run training first or specify model version in config.")
        
        logger.info(f"Using model: {context.model.name}, version: {context.model.version}")
        
        # Try to get the trained model artifact
        try:
            model_artifact = context.model.get_artifact("trained_model")
            if model_artifact is None:
                raise ValueError("trained_model artifact not found in model version")
        except Exception as e:
            logger.error(f"Failed to get trained_model artifact: {e}")
            # List all available artifacts for debugging
            logger.info("Available artifacts in model version:")
            for name in context.model.artifacts:
                logger.info(f"  - {name}")
            raise
        
        # Load the trained model
        trained_model = model_artifact.load()
        logger.info(f"Loaded model from Model Control Plane: {type(trained_model).__name__}")
        
        # Generate predictions using the loaded model
        predictions = trained_model.predict(n=horizon, series=series)
        logger.info(f"Generated {len(predictions)} predictions")
        
        # Convert to DataFrame
        pred_df = predictions.pd_dataframe().reset_index()
        pred_df.columns = ['ds', 'yhat']  # Standard naming
        
        # Create output directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save predictions
        pred_df.to_csv(output_path, index=False)
        
        logger.info(f"Saved {len(pred_df)} predictions to {output_path}")
        logger.info(f"Forecast period: {pred_df['ds'].min()} to {pred_df['ds'].max()}")
        logger.info(f"Prediction stats: mean={pred_df['yhat'].mean():.2f}, std={pred_df['yhat'].std():.2f}")
        
        # Log metadata to ZenML for observability
        log_metadata({
            "horizon": horizon,
            "num_predictions": len(pred_df),
            "forecast_start": str(pred_df['ds'].min()),
            "forecast_end": str(pred_df['ds'].max()),
            "prediction_mean": float(pred_df['yhat'].mean()),
            "prediction_std": float(pred_df['yhat'].std()),
            "prediction_min": float(pred_df['yhat'].min()),
            "prediction_max": float(pred_df['yhat'].max()),
            "model_type": type(trained_model).__name__,
            "model_name": context.model.name,
            "model_version": context.model.version
        })
        
        return output_path
        
    except Exception as e:
        logger.error(f"Batch inference failed: {str(e)}")
        raise