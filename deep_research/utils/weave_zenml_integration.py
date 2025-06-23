"""Integration utilities for linking Weave traces to ZenML pipeline runs."""

import logging
from typing import Optional
from zenml import log_metadata, get_step_context

logger = logging.getLogger(__name__)


def log_weave_trace_to_zenml(
    operation_name: str,
    trace_id: Optional[str] = None,
    additional_metadata: Optional[dict] = None
) -> None:
    """Log Weave trace information to ZenML step metadata.
    
    Args:
        operation_name: Name of the operation being tracked
        trace_id: Weave trace ID (if available)
        additional_metadata: Additional metadata to log
    """
    try:
        # Get current step context
        context = get_step_context()
        
        # Build metadata dictionary
        metadata = {
            "weave_tracking": {
                "operation": operation_name,
                "weave_project_url": f"https://wandb.ai/zenmlcode/deep-research/weave",
                "zenml_pipeline_run": context.pipeline_run.name,
                "zenml_step_name": context.step_run.name,
                "timestamp": context.step_run.start_time.isoformat() if context.step_run.start_time else None,
            }
        }
        
        # Add trace ID if available
        if trace_id:
            metadata["weave_tracking"]["trace_id"] = trace_id
            metadata["weave_tracking"]["trace_url"] = f"https://wandb.ai/zenmlcode/deep-research/weave/traces/{trace_id}"
        
        # Add additional metadata
        if additional_metadata:
            metadata["weave_tracking"].update(additional_metadata)
        
        # Log to ZenML
        log_metadata(metadata)
        logger.info(f"Logged Weave tracking info for {operation_name} to ZenML metadata")
        
    except Exception as e:
        logger.warning(f"Failed to log Weave trace to ZenML: {e}")


def get_weave_project_url(project_name: str = "deep-research") -> str:
    """Get the Weave project URL.
    
    Args:
        project_name: Weave project name
        
    Returns:
        URL to the Weave project dashboard
    """
    return f"https://wandb.ai/zenmlcode/{project_name}/weave"


def create_weave_dashboard_link() -> dict:
    """Create a link to the Weave dashboard for this pipeline run.
    
    Returns:
        Dictionary with dashboard link information
    """
    try:
        context = get_step_context()
        return {
            "weave_dashboard": {
                "url": get_weave_project_url(),
                "description": f"View Weave traces for pipeline run: {context.pipeline_run.name}",
                "pipeline_run_name": context.pipeline_run.name,
                "pipeline_run_id": str(context.pipeline_run.id),
            }
        }
    except Exception:
        return {
            "weave_dashboard": {
                "url": get_weave_project_url(),
                "description": "View Weave traces for this pipeline",
            }
        }


def log_weave_summary_to_zenml() -> None:
    """Log a summary of Weave tracking to ZenML metadata."""
    try:
        dashboard_info = create_weave_dashboard_link()
        
        # Add summary information
        dashboard_info["weave_summary"] = {
            "tracking_enabled": True,
            "automatic_llm_tracking": True,
            "trace_collection": "All LiteLLM calls are automatically tracked",
            "dashboard_access": "Click the URL above to view detailed traces",
        }
        
        log_metadata(dashboard_info)
        logger.info("Logged Weave tracking summary to ZenML metadata")
        
    except Exception as e:
        logger.warning(f"Failed to log Weave summary to ZenML: {e}")


def add_weave_context_to_function(func_name: str, **kwargs) -> dict:
    """Add ZenML context to a Weave-tracked function call.
    
    Args:
        func_name: Name of the function being called
        **kwargs: Additional context to include
        
    Returns:
        Dictionary with context information
    """
    try:
        context = get_step_context()
        return {
            "zenml_context": {
                "pipeline_run": context.pipeline_run.name,
                "step_name": context.step_run.name,
                "function_name": func_name,
                **kwargs
            }
        }
    except Exception:
        return {
            "zenml_context": {
                "function_name": func_name,
                **kwargs
            }
        }