"""LLM configuration for LiteLLM and Langfuse integration."""

import os
import litellm
from zenml import get_step_context
from zenml.logger import get_logger

logger = get_logger(__name__)

def initialize_litellm_langfuse():
    """Initialize LiteLLM with Langfuse integration for better compatibility."""
    # Verify Langfuse credentials are available
    if not all([
        os.getenv("LANGFUSE_PUBLIC_KEY"),
        os.getenv("LANGFUSE_SECRET_KEY")
    ]):
        logger.warning("Langfuse credentials not found. LLM calls will not be traced.")
        return False
    
    try:
        # Configure LiteLLM with Langfuse integration
        litellm.callbacks = ["langfuse"]
        
        logger.info("LiteLLM configured with Langfuse integration")
        return True
        
    except Exception as e:
        logger.error(f"Failed to configure LiteLLM with Langfuse: {e}")
        logger.warning("Continuing without Langfuse tracing. LLM calls will still work but won't be traced.")
        
        # Clear any partial callback configuration
        try:
            litellm.callbacks = []
        except:
            pass
            
        return False

def get_pipeline_run_id():
    """Get the current ZenML pipeline run ID for use as trace_id."""
    try:
        step_context = get_step_context()
        return str(step_context.step_run.pipeline_run_id)
    except Exception:
        # Fallback for when not in a ZenML step context
        import uuid
        return str(uuid.uuid4())

def get_langfuse_project_id():
    """Get the Langfuse project ID from environment variable."""
    project_id = os.getenv("LANGFUSE_PROJECT_ID")
    if not project_id:
        logger.warning("LANGFUSE_PROJECT_ID not set. Please set this environment variable to generate direct trace URLs.")
    return project_id

def generate_trace_url(trace_id: str, timestamp: str = None) -> str:
    """Generate a direct Langfuse trace URL with optional timestamp."""
    project_id = get_langfuse_project_id()
    if project_id:
        base_url = f"https://cloud.langfuse.com/project/{project_id}/traces/{trace_id}"
        if timestamp:
            return f"{base_url}?timestamp={timestamp}&display=details"
        return f"{base_url}?display=details"
    return f"https://cloud.langfuse.com/traces/{trace_id}"