"""
Training pipeline for FloraCast forecasting models.
"""

from zenml import pipeline
from zenml.logger import get_logger

from steps.ingest import ingest_data
from steps.preprocess import preprocess_for_training
from steps.train import train_model
from steps.evaluate import evaluate
from steps.promote import promote_model

logger = get_logger(__name__)


@pipeline(enable_cache=True)
def train_forecast_pipeline() -> None:
    """
    Training pipeline that ingests data, preprocesses it, trains a model,
    evaluates it, and promotes it via ZenML Model Control Plane.
    """
    # Step 1: Ingest data
    raw_data = ingest_data()

    # Step 2: Preprocess data into Darts TimeSeries with train/val split
    train_series, val_series = preprocess_for_training(df=raw_data)

    # Step 3: Train the forecasting model
    trained_model = train_model(train_series=train_series)

    # Step 4: Evaluate the model
    evaluation_results = evaluate(
        model=trained_model, train_series=train_series, val_series=val_series
    )
    score = evaluation_results[0]

    # Step 5: Register model and promote if better
    _ = promote_model(current_score=score)
