"""
Batch inference pipeline for FloraCast forecasting models.
"""

from zenml import pipeline
from zenml.logger import get_logger

from steps.ingest import ingest_data
from steps.batch_infer import batch_inference_predict

logger = get_logger(__name__)


@pipeline(enable_cache=False)
def batch_inference_pipeline() -> None:
    """
    Batch inference pipeline that loads model from Model Control Plane and generates predictions.
    """
    # Step 1: Ingest data (simulate real-time data sources)
    raw_data = ingest_data(infer=True)

    # Step 2: Generate predictions using model from MCP (with scaling handled internally)
    batch_inference_predict(df=raw_data)
