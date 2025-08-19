"""
Training pipeline for FloraCast forecasting models.
"""

from typing import Dict, Any
from zenml import pipeline
from zenml.logger import get_logger

from steps.ingest import ingest_data
from steps.preprocess import preprocess
from steps.train import train_model
from steps.evaluate import evaluate
from steps.promote import promote_model
from materializers import DartsTimeSeriesMaterializer, TFTModelMaterializer

logger = get_logger(__name__)


@pipeline
def train_forecast_pipeline() -> None:
    """
    Training pipeline that ingests data, preprocesses it, trains a model,
    evaluates it, and promotes it via ZenML Model Control Plane.
    """
    logger.info("Starting FloraCast training pipeline")
    
    # Step 1: Ingest data
    raw_data = ingest_data()
    
    # Step 2: Preprocess data into Darts TimeSeries
    train_series, val_series = preprocess(df=raw_data)
    
    # Step 3: Train the forecasting model
    trained_model, artifact_uri, model_class = train_model(
        train_series=train_series
    )
    
    # Step 4: Evaluate the model
    score = evaluate(
        model=trained_model,
        train_series=train_series,
        val_series=val_series
    )
    
    # Step 5: Register model and promote if better
    promotion_status = promote_model(
        score=score,
        artifact_uri=artifact_uri,
        model_class=model_class
    )
    
    logger.info(f"Training pipeline completed: {promotion_status}")