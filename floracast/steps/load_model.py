"""
Step to load trained model from ZenML Model Control Plane.
"""

from typing import Dict, Any, Tuple, Annotated
from zenml import step, get_pipeline_context
from zenml.logger import get_logger

logger = get_logger(__name__)


@step
def load_latest_model(config: Dict[str, Any]) -> Tuple[
    Annotated[object, "loaded_model"],
    Annotated[str, "artifact_uri"],
    Annotated[str, "model_class"],
]:
    """
    Load the trained model from ZenML Model Control Plane.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (loaded_model, artifact_uri, model_class)
    """
    try:
        # Get the pipeline context to access the model
        context = get_pipeline_context()
        model = context.model
        
        if model is None:
            raise ValueError("No model found in pipeline context. Please run training first.")
        
        logger.info(f"Loading model from Model Control Plane: {model.name}")
        
        # Load the model artifacts
        trained_model = model.get_artifact("trained_model")
        artifact_uri_artifact = model.get_artifact("artifact_uri") 
        model_class_artifact = model.get_artifact("model_class")
        
        # Load the actual objects
        loaded_model = trained_model.load()
        artifact_uri = artifact_uri_artifact.load()
        model_class = model_class_artifact.load()
        
        logger.info(f"Successfully loaded {model_class} model from MCP")
        
        return loaded_model, artifact_uri, model_class
        
    except Exception as e:
        logger.error(f"Failed to load model from MCP: {e}")
        raise ValueError(f"Could not load trained model from Model Control Plane: {e}")


@step  
def load_production_model(config: Dict[str, Any]) -> Tuple[
    Annotated[object, "loaded_model"],
    Annotated[str, "artifact_uri"],
    Annotated[str, "model_class"],
]:
    """
    Load production model from ZenML Model Control Plane (placeholder).
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (loaded_model, artifact_uri, model_class)
    """
    # This is a placeholder for actual MCP integration
    # In a real production setup, this would load from the model registry
    logger.info("Production model loading not implemented yet")
    logger.info("Falling back to loading latest training run")
    
    return load_latest_model(config=config)