from zenml import Model, get_step_context, step
from zenml.logger import get_logger

logger = get_logger(__name__)


@step
def model_promoter(mse: float, stage: str = "production") -> bool:
    """Model promotion step

    Step that conditionally promotes a model in case it has an MSE greater than
    the previous production model

    Args:
        mse: Mean-squared error of the model.
        stage: Which stage to promote the model to.

    Returns:
        Whether the model was promoted or not.
    """
    # Get the current model produced by the current pipeline
    zenml_model = get_step_context().model

    # Get the previous model version at the production stage
    previous_production_model = Model(
        name=zenml_model.name, version="production"
    )

    try:
        # In case there already is a model version at the correct stage
        previous_production_model_mse = float(
            previous_production_model.get_artifact("sklearn_regressor")
            .run_metadata["metrics"]["mse"]
        )
    except RuntimeError:
        # In case no model version has been promoted before,
        #   default to a threshold value well above the new mse
        previous_production_model_mse = mse + 1000

    if mse > previous_production_model_mse:
        logger.info(
            f"Model mean-squared error {mse:.2f} is higher than"
            f" the mse of the previous production model "
            f"{previous_production_model_mse:.2f} ! "
            f"Not promoting model."
        )
        is_promoted = False
    else:
        logger.info(f"Model promoted to {stage}!")
        is_promoted = True
        zenml_model = get_step_context().model
        zenml_model.set_stage(stage, force=True)

    return is_promoted
