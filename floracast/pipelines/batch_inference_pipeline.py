"""
Batch inference pipeline for FloraCast forecasting models.
"""

from zenml import pipeline
from zenml.logger import get_logger

from steps.ingest import ingest_data
from steps.preprocess import preprocess
from steps.batch_infer import batch_inference_predict

logger = get_logger(__name__)


@pipeline(enable_cache=False)
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
    batch_inference_predict(series=inference_series)

    logger.info("Batch inference completed. Returning predictions DataFrame.")
