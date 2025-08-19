"""
Batch inference pipeline for FloraCast forecasting models.
"""

from typing import Dict, Any
from zenml import pipeline
from zenml.logger import get_logger

from steps.ingest import ingest_data
from steps.preprocess import preprocess
from steps.batch_infer import batch_inference_predict
from materializers import DartsTimeSeriesMaterializer, TFTModelMaterializer

logger = get_logger(__name__)


@pipeline
def batch_inference_pipeline() -> None:
    """
    Batch inference pipeline that loads model from Model Control Plane and generates predictions.
    """
    logger.info("Starting FloraCast batch inference pipeline")
    
    # Step 1: Ingest data
    raw_data = ingest_data()
    
    # Step 2: Preprocess data
    inference_series, _ = preprocess(df=raw_data)
    
    # Step 3: Generate predictions using model from MCP
    output_path = batch_inference_predict(series=inference_series)
    
    logger.info(f"Batch inference completed. Results saved to: {output_path}")