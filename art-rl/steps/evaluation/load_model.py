"""Step to load a trained model for inference."""

from typing import Annotated

from zenml import step
from zenml.logger import get_logger

logger = get_logger(__name__)


@step
def load_trained_model(
    model_config: dict,
    checkpoint_path: str,
    art_path: str = "./.art",
) -> Annotated[dict, "inference_config"]:
    """Prepare configuration for loading the trained model.

    Since the actual model loading requires GPU resources and must happen
    in the inference step, this step prepares the configuration needed
    to load the model from the checkpoint.

    Args:
        model_config: Original model configuration.
        checkpoint_path: Path to the trained checkpoint.
        art_path: Directory for ART files.

    Returns:
        Configuration dict for loading the model during inference.
    """
    inference_config = {
        **model_config,
        "checkpoint_path": checkpoint_path,
        "art_path": art_path,
    }

    logger.info(
        f"Prepared inference config from checkpoint: {checkpoint_path}"
    )
    return inference_config
