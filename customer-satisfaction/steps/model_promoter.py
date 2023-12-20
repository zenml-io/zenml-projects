from zenml import get_step_context, step, ModelVersion
from zenml.logger import get_logger

logger = get_logger(__name__)


@step
def model_promoter(
    accuracy: float,
    stage: str = "production"
) -> bool:
    """Model promotion step

    Step that conditionally promotes a model in case it has an MSE greater than
    the previous production model

    Args:
        accuracy: Accuracy of the model.
        stage: Which stage to promote the model to.

    Returns:
        Whether the model was promoted or not.
    """
    # Get the current model_version produced by the current pipeline
    model_version = get_step_context().model_version

    # Get the previous model version at the production stage
    previous_production_model = ModelVersion(
        name=model_version.name,
        version="production"
    )

    try:
        previous_production_model_version_mse = float(
            previous_production_model.get_artifact("model").run_metadata["mse"].value
        )
    except RuntimeError:
        previous_production_model_version_mse = 0.0

    if accuracy < previous_production_model_version_mse:
        logger.info(
            f"Model accuracy {accuracy*100:.2f}% is below the accuracy of "
            f"the previous production model "
            f"{previous_production_model_version_mse*100:.2f}% ! "
            f"Not promoting model."
        )
        is_promoted = False
    else:
        logger.info(f"Model promoted to {stage}!")
        is_promoted = True
        model_version = get_step_context().model_version
        model_version.set_stage(stage, force=True)

    return is_promoted
