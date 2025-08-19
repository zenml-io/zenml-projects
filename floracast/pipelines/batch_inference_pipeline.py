"""
Batch inference pipeline for FloraCast forecasting models.
"""

from zenml import pipeline
from zenml.logger import get_logger

from steps.ingest import ingest_data
from steps.preprocess import preprocess_for_inference
from steps.batch_infer import batch_inference_predict

logger = get_logger(__name__)


@pipeline(enable_cache=False)
def batch_inference_pipeline() -> None:
    """
    Batch inference pipeline that loads model from Model Control Plane and generates predictions.
    """
    logger.info("Starting FloraCast batch inference pipeline")

    # Step 1: Ingest data (simulate real-time data sources)
    raw_data = ingest_data(infer=True)

    # Step 2: Preprocess data (use full series for inference context)
    inference_series = preprocess_for_inference(df=raw_data)

    # Step 3: Generate predictions using model from MCP
    batch_inference_predict(series=inference_series)

    logger.info("Batch inference completed. Returning predictions DataFrame.")
