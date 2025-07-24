"""Manual Langfuse logging for LiteLLM calls to avoid version conflicts."""

import os
from typing import Any, Dict, List, Optional

from zenml.logger import get_logger

logger = get_logger(__name__)


class ManualLangfuseLogger:
    """Manual Langfuse logger that bypasses LiteLLM's built-in integration."""

    def __init__(self):
        self.langfuse_client = None
        self.enabled = False
        self._initialize_client()

    def _initialize_client(self):
        """Initialize Langfuse client."""
        try:
            # Check for credentials
            if not all(
                [
                    os.getenv("LANGFUSE_PUBLIC_KEY"),
                    os.getenv("LANGFUSE_SECRET_KEY"),
                ]
            ):
                logger.warning(
                    "Langfuse credentials not found. Manual logging disabled."
                )
                return

            from langfuse import Langfuse

            self.langfuse_client = Langfuse(
                public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
                secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
                host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
            )
            self.enabled = True
            logger.info("Manual Langfuse logger initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize manual Langfuse logger: {e}")
            self.enabled = False

    def log_llm_call(
        self,
        messages: List[Dict[str, str]],
        response_content: str,
        metadata: Dict[str, Any],
        model: str,
        usage: Optional[Dict[str, Any]] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ):
        """Log an LLM call to Langfuse."""
        if not self.enabled:
            return

        try:
            # Extract metadata
            tags = metadata.get("tags", [])
            trace_name = metadata.get("trace_name", "llm_call")
            session_id = metadata.get("session_id")
            trace_user_id = metadata.get("trace_user_id")
            generation_name = metadata.get("generation_name", "generation")
            trace_metadata = metadata.get("trace_metadata", {})

            # Create or update trace
            trace = self.langfuse_client.trace(
                name=trace_name,
                session_id=session_id,
                user_id=trace_user_id,
                tags=tags,
                metadata=trace_metadata,
            )

            # Create generation
            generation = trace.generation(
                name=generation_name,
                model=model,
                input=messages,
                output=response_content,
                start_time=start_time,
                end_time=end_time,
                usage=usage,
                metadata=metadata,
            )

            # Flush to ensure data is sent
            self.langfuse_client.flush()

            logger.debug(f"Logged LLM call to Langfuse: {trace_name}")

        except Exception as e:
            logger.error(f"Failed to log LLM call to Langfuse: {e}")


# Global instance
_manual_logger = None


def get_manual_langfuse_logger() -> ManualLangfuseLogger:
    """Get the global manual Langfuse logger instance."""
    global _manual_logger
    if _manual_logger is None:
        _manual_logger = ManualLangfuseLogger()
    return _manual_logger
