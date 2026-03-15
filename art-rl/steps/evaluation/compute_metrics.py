"""Step to compute evaluation metrics from predictions."""

from typing import Annotated, List

from zenml import log_model_metadata, step
from zenml.logger import get_logger

logger = get_logger(__name__)


@step
def compute_metrics(
    predictions: List[dict],
) -> Annotated[dict, "evaluation_metrics"]:
    """Compute evaluation metrics from inference predictions.

    Calculates:
    - Accuracy: Fraction of scenarios answered correctly
    - Answer rate: Fraction of scenarios that got any answer
    - Source precision: How often predicted sources match expected

    Args:
        predictions: List of prediction dictionaries from run_inference.

    Returns:
        Dictionary of evaluation metrics.
    """
    total = len(predictions)
    if total == 0:
        logger.warning("No predictions to evaluate")
        return {
            "total_scenarios": 0,
            "correct": 0,
            "accuracy": 0.0,
            "answer_rate": 0.0,
            "source_precision": 0.0,
        }

    correct = sum(p["correct"] for p in predictions)
    answered = sum(1 for p in predictions if p.get("predicted_answer"))
    errors = sum(1 for p in predictions if p.get("error"))

    # Calculate source precision (did we find the right emails?)
    source_matches = 0
    source_total = 0
    for p in predictions:
        expected = set(p.get("expected_message_ids", []))
        predicted = set(p.get("predicted_source_ids", []))
        if expected:
            source_total += 1
            if expected & predicted:  # Any overlap
                source_matches += 1

    metrics = {
        "total_scenarios": total,
        "correct": correct,
        "accuracy": correct / total,
        "answered": answered,
        "answer_rate": answered / total,
        "errors": errors,
        "error_rate": errors / total,
        "source_matches": source_matches,
        "source_precision": (
            source_matches / source_total if source_total > 0 else 0.0
        ),
    }

    # Log to ZenML Model Control Plane
    log_model_metadata(metadata={"evaluation": metrics})

    logger.info(f"Evaluation metrics: {metrics}")
    return metrics
