"""
Model promotion step using ZenML Model Control Plane.
"""

from typing import Annotated, Optional

from zenml import get_step_context, log_metadata, step
from zenml.client import Client
from zenml.enums import ModelStages
from zenml.logger import get_logger

logger = get_logger(__name__)


@step
def promote_model(
    current_score: float,
    target_stage: str = ModelStages.PRODUCTION,
    metric: str = "smape",
) -> Annotated[str, "promotion_status"]:
    """
    Promote current model version to a target stage if it outperforms the
    model currently in that stage, using artifacts/metadata from the model
    context. Also logs decision metadata.

    Args:
        target_stage: The stage to promote to (default: production)
        metric: The evaluation metric to compare (default: smape)

    Returns:
        Status message about the promotion decision.
    """

    client = Client()

    try:
        # Use the model from the current step context (pattern from batch_infer)
        context_model = get_step_context().model
        if not context_model:
            raise ValueError(
                "No model found in step context. Ensure training associated a model with this pipeline."
            )

        logger.info(
            f"Evaluating promotion for model: {context_model.name} (version: {context_model.version})"
        )

        # Use the score passed from the evaluation step
        current_score = float(current_score)

        # (Removed optional artifact_uri/model_class collection as unnecessary)

        # Fetch production version score to compare with
        prod_score: Optional[float] = None
        try:
            prod_model_version = client.get_model_version(
                model_name_or_id=context_model.name,
                model_version_name_or_number_or_id=target_stage,
            )
            prod_score_artifact = prod_model_version.get_artifact(
                "evaluation_score"
            )
            if prod_score_artifact is not None:
                prod_score = float(prod_score_artifact.load())
            else:
                logger.info(
                    f"`{target_stage}` model has no evaluation_score artifact; promotion will be skipped."
                )
        except (RuntimeError, KeyError):
            logger.info(
                f"No existing `{target_stage}` model version found. Current version will be promoted by default."
            )

        # Decide promotion
        promote = False
        reason = ""
        if prod_score is None:
            promote = True
            reason = f"No {target_stage} model to compare against"
        else:
            if metric.lower() in {"smape", "mae", "rmse", "mse"}:
                # Lower is better
                promote = current_score < prod_score
                reason = "lower_is_better"
            else:
                # Higher is better for other metrics by convention
                promote = current_score > prod_score
                reason = "higher_is_better"

        if promote:
            context_model.set_stage(stage=target_stage, force=True)
            status = f"Promoted model '{context_model.name}' v{context_model.version} to '{target_stage}'."
            logger.info(status)
            decision = "promoted"
        else:
            status = (
                f"Skipped promotion to '{target_stage}': current score={current_score:.6f} "
                f"not better than {target_stage} score={prod_score:.6f}."
            )
            logger.info(status)
            decision = "skipped"

        # Log decision and comparison metadata
        log_metadata(
            {
                "decision": decision,
                "reason": reason,
                "target_stage": str(target_stage),
                "metric": metric,
                "current_model_name": context_model.name,
                "current_model_version": str(context_model.version),
                "current_score": float(current_score),
                "production_score": None
                if prod_score is None
                else float(prod_score),
            }
        )

        return status

    except Exception as e:
        error_status = f"Promotion step failed: {str(e)}"
        logger.error(error_status)
        # Best-effort metadata for failure diagnostics
        try:
            log_metadata({"decision": "error", "error": str(e)})
        except Exception:
            pass
        return error_status
