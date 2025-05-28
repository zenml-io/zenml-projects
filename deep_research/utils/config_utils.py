"""Configuration and environment utilities for the Deep Research Agent."""

import logging
import os
from typing import Any, Dict

import yaml

logger = logging.getLogger(__name__)


def load_pipeline_config(config_path: str) -> Dict[str, Any]:
    """Load pipeline configuration from YAML file.

    This is used only for pipeline-level configuration, not for step parameters.
    Step parameters should be defined directly in the step functions.

    Args:
        config_path: Path to the configuration YAML file

    Returns:
        Pipeline configuration dictionary
    """
    # Get absolute path if relative
    if not os.path.isabs(config_path):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(base_dir, config_path)

    # Load YAML configuration
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logger.error(f"Error loading pipeline configuration: {e}")
        # Return a minimal default configuration in case of loading error
        return {
            "pipeline": {
                "name": "deep_research_pipeline",
                "enable_cache": True,
            },
            "environment": {
                "docker": {
                    "requirements": [
                        "openai>=1.0.0",
                        "tavily-python>=0.2.8",
                        "PyYAML>=6.0",
                        "click>=8.0.0",
                        "pydantic>=2.0.0",
                        "typing_extensions>=4.0.0",
                    ]
                }
            },
            "resources": {"cpu": 1, "memory": "4Gi"},
            "timeout": 3600,
        }


def check_required_env_vars(env_vars: list[str]) -> list[str]:
    """Check if required environment variables are set.

    Args:
        env_vars: List of environment variable names to check

    Returns:
        List of missing environment variables
    """
    missing_vars = []
    for var in env_vars:
        if not os.environ.get(var):
            missing_vars.append(var)
    return missing_vars
