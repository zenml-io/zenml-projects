"""
Session management utility for Langfuse integration with ZenML pipelines.

This module provides utilities to:
1. Extract ZenML pipeline run ID and use it as Langfuse session ID
2. Generate Langfuse session URLs for pipeline metadata
3. Configure Langfuse tracing with session context
"""

import os
from typing import Optional, Dict, Any
from zenml import get_step_context
from zenml.logger import get_logger
from langfuse import get_client

logger = get_logger(__name__)


class SessionManager:
    """Manages Langfuse sessions integrated with ZenML pipeline runs."""
    
    def __init__(self):
        self._session_id: Optional[str] = None
        self._pipeline_name: Optional[str] = None
        self._run_name: Optional[str] = None
        self._langfuse_client = None
        
    @property
    def session_id(self) -> str:
        """Get the session ID, creating it from ZenML context if needed."""
        if self._session_id is None:
            self._initialize_session()
        return self._session_id
    
    @property
    def pipeline_name(self) -> Optional[str]:
        """Get the current pipeline name."""
        if self._pipeline_name is None:
            self._initialize_session()
        return self._pipeline_name
    
    @property
    def run_name(self) -> Optional[str]:
        """Get the current run name."""
        if self._run_name is None:
            self._initialize_session()
        return self._run_name
    
    def _initialize_session(self):
        """Initialize session from ZenML step context."""
        try:
            # Get ZenML step context
            step_context = get_step_context()
            
            # Use ZenML pipeline run UUID as Langfuse session ID
            self._session_id = str(step_context.pipeline_run.id)
            self._pipeline_name = step_context.pipeline.name
            self._run_name = step_context.pipeline_run.name
            
            logger.info(f"Initialized Langfuse session with ZenML run ID: {self._session_id}")
            logger.info(f"Pipeline: {self._pipeline_name}, Run: {self._run_name}")
            
        except Exception as e:
            logger.warning(f"Could not initialize session from ZenML context: {e}")
            # Fallback to a default session ID
            import uuid
            self._session_id = str(uuid.uuid4())
            self._pipeline_name = "unknown_pipeline"
            self._run_name = "unknown_run"
    
    def get_langfuse_session_url(self) -> Optional[str]:
        """Generate Langfuse session URL for the current session."""
        try:
            langfuse_host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
            langfuse_public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
            langfuse_secret_key = os.getenv("LANGFUSE_SECRET_KEY")
            
            if not langfuse_public_key or not langfuse_secret_key:
                logger.warning("LANGFUSE_PUBLIC_KEY or LANGFUSE_SECRET_KEY not set, cannot generate session URL")
                return None
            
            # Get project ID by making an API call to fetch any trace
            project_id = self._get_project_id_from_api()
            
            if project_id:
                session_url = f"{langfuse_host}/project/{project_id}/sessions/{self.session_id}"
                logger.debug(f"Generated session URL with project ID: {session_url}")
                return session_url
            else:
                logger.warning("Could not determine Langfuse project ID, using fallback URL format")
                session_url = f"{langfuse_host}/sessions/{self.session_id}"
                return session_url
            
        except Exception as e:
            logger.warning(f"Could not generate Langfuse session URL: {e}")
            return None
    
    def _get_project_id_from_api(self) -> Optional[str]:
        """Get project ID by making an API call to Langfuse."""
        try:
            client = self.configure_langfuse_client()
            
            # Try to fetch any trace to get project ID from the response
            traces_response = client.api.trace.list(limit=1)
            if traces_response.data and len(traces_response.data) > 0:
                trace = traces_response.data[0]
                if hasattr(trace, 'project_id'):
                    return trace.project_id
                elif hasattr(trace, 'projectId'):
                    return trace.projectId
            
            # If no traces exist, try creating a minimal trace to get project ID
            trace = client.trace(name="project_id_check")
            client.flush()  # Ensure trace is sent
            
            # Fetch the trace we just created
            fetched_trace = client.api.trace.get(trace.id)
            if hasattr(fetched_trace, 'project_id'):
                return fetched_trace.project_id
            elif hasattr(fetched_trace, 'projectId'):
                return fetched_trace.projectId
                
        except Exception as e:
            logger.debug(f"Could not get project ID from API: {e}")
            return None
        
        return None
    
    def get_session_metadata(self) -> Dict[str, Any]:
        """Get session metadata for ZenML pipeline metadata logging."""
        metadata = {
            "langfuse_session_id": self.session_id,
            "pipeline_name": self.pipeline_name,
            "run_name": self.run_name,
        }
        
        session_url = self.get_langfuse_session_url()
        if session_url:
            metadata["langfuse_session_url"] = session_url
            
        return metadata
    
    def configure_langfuse_client(self):
        """Configure Langfuse client with session information."""
        if self._langfuse_client is None:
            self._langfuse_client = get_client()
        return self._langfuse_client
    
    def update_current_trace_with_session(self):
        """Update the current Langfuse trace with session information."""
        try:
            client = self.configure_langfuse_client()
            client.update_current_trace(session_id=self.session_id)
            logger.debug(f"Updated current trace with session ID: {self.session_id}")
        except Exception as e:
            logger.warning(f"Could not update current trace with session: {e}")


# Global session manager instance
_session_manager = None


def get_session_manager() -> SessionManager:
    """Get the global session manager instance."""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager


def get_current_session_id() -> str:
    """Get the current Langfuse session ID (ZenML run UUID)."""
    return get_session_manager().session_id


def log_session_metadata():
    """Log session metadata to the current ZenML pipeline run."""
    try:
        from zenml import log_metadata
        
        session_manager = get_session_manager()
        metadata = session_manager.get_session_metadata()
        
        log_metadata(
            metadata={"langfuse_session": metadata}
        )
        
        logger.info(f"Logged Langfuse session metadata to ZenML pipeline")
        logger.info(f"Session URL: {metadata.get('langfuse_session_url', 'Not available')}")
        
    except Exception as e:
        logger.warning(f"Could not log session metadata: {e}")


def configure_agent_with_session(agent_class):
    """Decorator to configure agents with session management."""
    def decorator(original_method):
        def wrapper(*args, **kwargs):
            # Update current trace with session before agent execution
            session_manager = get_session_manager()
            session_manager.update_current_trace_with_session()
            
            # Execute original method
            result = original_method(*args, **kwargs)
            
            return result
        return wrapper
    
    # Apply to all @observe decorated methods
    for attr_name in dir(agent_class):
        attr = getattr(agent_class, attr_name)
        if callable(attr) and hasattr(attr, '__wrapped__'):
            # This is likely an @observe decorated method
            setattr(agent_class, attr_name, decorator(attr))
    
    return agent_class