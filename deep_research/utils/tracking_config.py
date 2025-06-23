"""Tracking configuration utilities for experiment tracking providers."""

import logging
import os
from typing import Optional

import litellm
from zenml import get_step_context

logger = logging.getLogger(__name__)


def configure_tracking_provider(
    tracking_provider: str = "weave",
    langfuse_project_name: Optional[str] = None,
    weave_project_name: Optional[str] = None,
) -> None:
    """Configure the tracking provider for LLM calls.
    
    Args:
        tracking_provider: The tracking provider to use ('weave', 'langfuse', or 'none')
        langfuse_project_name: Project name for Langfuse tracking
        weave_project_name: Project name for Weave tracking
    """
    logger.info(f"Configuring tracking provider: {tracking_provider}")
    
    if tracking_provider.lower() == "langfuse":
        # Clear existing callbacks and configure Langfuse tracking
        litellm.callbacks = []
        _configure_langfuse(langfuse_project_name)
    elif tracking_provider.lower() == "weave":
        # Configure Weave tracking - do NOT clear callbacks as Weave manages this
        _configure_weave(weave_project_name)
    elif tracking_provider.lower() == "none":
        # Clear all callbacks
        litellm.callbacks = []
        logger.info("No tracking provider configured - callbacks cleared")
    else:
        logger.warning(f"Unknown tracking provider: {tracking_provider}. Using Weave as default.")
        _configure_weave(weave_project_name)


def _configure_langfuse(project_name: Optional[str] = None) -> None:
    """Configure Langfuse tracking.
    
    Args:
        project_name: Langfuse project name
    """
    try:
        # Check if required environment variables are set
        required_env_vars = ["LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"]
        missing_vars = [var for var in required_env_vars if not os.getenv(var)]
        
        if missing_vars:
            logger.warning(f"Missing Langfuse environment variables: {missing_vars}. Langfuse tracking disabled.")
            return
            
        # Set project name if provided
        if project_name:
            os.environ["LANGFUSE_PROJECT"] = project_name
            
        # Add Langfuse callback
        litellm.callbacks = ["langfuse"]
        logger.info(f"Langfuse tracking configured with project: {project_name or 'default'}")
        
    except Exception as e:
        logger.error(f"Failed to configure Langfuse: {e}")


def _configure_weave(project_name: Optional[str] = None) -> None:
    """Configure Weave tracking with ZenML pipeline context.
    
    Args:
        project_name: Weave project name
    """
    try:
        import weave
        
        # Check if WANDB_API_KEY is available
        if not os.getenv("WANDB_API_KEY"):
            logger.warning("WANDB_API_KEY not found. Weave tracking may not work properly.")
            return
            
        # Get ZenML pipeline context for metadata
        pipeline_context = _get_pipeline_context()
        
        # Initialize Weave with enhanced project name including run context
        if project_name:
            if pipeline_context["run_name"]:
                # Include run name in project for better organization
                enhanced_project_name = f"{project_name}"
                weave.init(enhanced_project_name)
                logger.info(f"Weave tracking configured with project: {enhanced_project_name}")
                
                # Set global tags/metadata for this pipeline run
                weave.attributes.update({
                    "zenml_pipeline_run": pipeline_context["run_name"],
                    "zenml_pipeline_id": pipeline_context["run_id"],
                    "zenml_step_name": pipeline_context["step_name"],
                })
            else:
                weave.init(project_name)
                logger.info(f"Weave tracking configured with project: {project_name}")
        else:
            weave.init("deep-research")
            logger.info("Weave tracking configured with default project name")
            
        # Log the pipeline context for debugging
        if pipeline_context["run_name"]:
            logger.info(f"Weave configured with ZenML context: {pipeline_context}")
            
        # Weave automatically tracks LiteLLM calls when properly imported
        logger.info("Weave will automatically track LiteLLM calls")
        
    except ImportError:
        logger.error("Weave not installed. Please install with: pip install weave")
    except Exception as e:
        logger.error(f"Failed to configure Weave: {e}")


def _get_pipeline_context() -> dict:
    """Get ZenML pipeline context information."""
    try:
        context = get_step_context()
        return {
            "run_name": context.pipeline_run.name,
            "run_id": str(context.pipeline_run.id),
            "step_name": context.step_run.name,
            "pipeline_name": context.pipeline_run.pipeline.name,
        }
    except RuntimeError:
        # Not running in a step context
        return {
            "run_name": None,
            "run_id": None,
            "step_name": None,
            "pipeline_name": None,
        }


def get_tracking_metadata(
    tracking_provider: str,
    project_name: Optional[str] = None,
    tags: Optional[list] = None,
    trace_name: Optional[str] = None,
    trace_id: Optional[str] = None,
) -> dict:
    """Get tracking metadata based on the configured provider.
    
    Args:
        tracking_provider: The tracking provider ('langfuse', 'weave', or 'none')
        project_name: Project name for tracking
        tags: List of tags for tracking
        trace_name: Trace name for tracking
        trace_id: Trace ID for tracking
        
    Returns:
        Metadata dictionary for the specified tracking provider
    """
    metadata = {}
    
    if tracking_provider.lower() == "langfuse":
        metadata["project"] = project_name or "deep-research"
        if tags:
            metadata["tags"] = tags
            metadata["trace_metadata"] = {tag: True for tag in tags}
        if trace_name:
            metadata["trace_name"] = trace_name
        if trace_id:
            metadata["trace_id"] = trace_id
            
    elif tracking_provider.lower() == "weave":
        # Weave metadata structure might be different
        # This can be customized based on Weave's requirements
        metadata["project"] = project_name or "deep-research"
        if tags:
            metadata["tags"] = tags
        if trace_name:
            metadata["trace_name"] = trace_name
        if trace_id:
            metadata["trace_id"] = trace_id
    
    return metadata