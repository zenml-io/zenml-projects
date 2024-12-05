# model_register.py

from typing_extensions import Annotated
from zenml import ArtifactConfig, get_step_context, step
from zenml.client import Client
from zenml.logger import get_logger

logger = get_logger(__name__)

@step(enable_cache=False)
def model_register() -> Annotated[str, ArtifactConfig(name="model_registry_uri")]:
    """Model registration step."""
    # Get the current model from the context
    current_model = get_step_context().model

    client = Client()
    model_registry = client.active_stack.model_registry
    model_version = model_registry.register_model_version(
        name=current_model.name,
        version=str(current_model.version),
        model_source_uri=current_model.get_model_artifact("sklearn_classifier").uri,
        description="ZenML model registered after promotion",
    )
    logger.info(
        f"Model version {model_version.version} registered in Model Registry"
    )
    
    return model_version.model_source_uri