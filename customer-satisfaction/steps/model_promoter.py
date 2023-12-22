from zenml import get_step_context, step, ModelVersion
from zenml.logger import get_logger

logger = get_logger(__name__)


@step
def model_promoter(
    mse: float,
    stage: str = "production"
) -> bool:
    """Model promotion step

    Step that conditionally promotes a model in case it has an MSE greater than
    the previous production model

    Args:
        mse: Mean-squared error of the model.
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
        # In case there already is a model version at the correct stage
        previous_production_model_version_mse = float(
            previous_production_model.get_artifact("sklearn_regressor").run_metadata["metrics"].value["mse"]
        )
    except RuntimeError:
        # In case no model version has been promoted before,
        #   default to a threshold value well above the new mse
        previous_production_model_version_mse = mse + 1000

    if mse > previous_production_model_version_mse:
        logger.info(
            f"Model mean-squared error {mse:.2f} is higher than"
            f" the mse of the previous production model "
            f"{previous_production_model_version_mse:.2f} ! "
            f"Not promoting model."
        )
        is_promoted = False
    else:
        logger.info(f"Model promoted to {stage}!")
        is_promoted = True
        model_version = get_step_context().model_version
        model_version.set_stage(stage, force=True)

    return is_promoted
