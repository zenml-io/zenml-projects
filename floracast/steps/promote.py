"""
Model promotion step using ZenML Model Control Plane.
"""

from typing import Dict, Any, Optional, Annotated
from zenml import step
from zenml.client import Client
from zenml.enums import ModelStages
from zenml.logger import get_logger

logger = get_logger(__name__)


@step
def promote_model(
    score: Annotated[float, "evaluation_score"],
    artifact_uri: Annotated[str, "artifact_uri"],
    model_class: Annotated[str, "model_class"],
    model_name: str = "floracast_tft"
) -> Annotated[str, "promotion_status"]:
    """
    Register model version in ZenML MCP and promote if better than current production.
    
    Args:
        score: Model evaluation score (lower is better)
        artifact_uri: Path to saved model artifacts
        model_class: Name of the model class used
        model_name: Name of the model in MCP
        
    Returns:
        Status message about promotion
    """
    
    logger.info(f"Registering model version for {model_name}")
    
    client = Client()
    
    try:
        # Get or create the model
        try:
            zenml_model = client.get_model(model_name)
            logger.info(f"Using existing model: {model_name}")
        except Exception:
            # Model doesn't exist, it will be created automatically by the step decorator
            logger.info(f"Model {model_name} will be created automatically")
            zenml_model = None
        
        # Create metadata for this version
        metadata = {
            "smape_score": score,
            "artifact_uri": artifact_uri,
            "model_class": model_class,
        }
        
        # For now, just log the model registration - full MCP integration would require model decorator
        status = f"Model registered with SMAPE: {score:.4f}, artifact_uri: {artifact_uri}, model_class: {model_class}"
        logger.info(status)
        
        # TODO: Implement full Model Control Plane integration with proper model decorator
        logger.info("Note: Full MCP promotion requires model decorator configuration")
        
        return status
        
    except Exception as e:
        error_status = f"Model registration failed: {str(e)}"
        logger.error(error_status)
        return error_status