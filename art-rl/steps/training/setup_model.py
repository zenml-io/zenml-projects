"""Step to configure the ART trainable model."""

from typing import Annotated

from zenml import step
from zenml.logger import get_logger

logger = get_logger(__name__)


@step
def setup_art_model(
    model_name: str = "art-email-agent",
    project_name: str = "email-search-agent",
    base_model: str = "Qwen/Qwen2.5-7B-Instruct",
) -> Annotated[dict, "model_config"]:
    """Configure the ART model for training.

    This step creates a configuration dictionary that will be used to
    initialize the ART TrainableModel in the training step. We separate
    configuration from initialization because the actual model loading
    requires GPU resources.

    Args:
        model_name: Name for the trained model (used for checkpoints).
        project_name: Project name for organizing experiments.
        base_model: Hugging Face model ID for the base model.
            Qwen 2.5 7B Instruct is recommended for this task.

    Returns:
        Configuration dictionary for model initialization.
    """
    config = {
        "name": model_name,
        "project": project_name,
        "base_model": base_model,
    }

    logger.info(f"Model configuration: {config}")
    return config
