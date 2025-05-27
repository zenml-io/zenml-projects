import json
import logging
import os
from json.decoder import JSONDecodeError
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)


def remove_reasoning_from_output(output: str) -> str:
    """Remove the reasoning portion from LLM output.

    Args:
        output: Raw output from LLM that may contain reasoning

    Returns:
        Cleaned output without the reasoning section
    """
    if not output:
        return ""

    if "</think>" in output:
        return output.split("</think>")[-1].strip()
    return output.strip()


def clean_json_tags(text: str) -> str:
    """Clean JSON markdown tags from text.

    Args:
        text: Text with potential JSON markdown tags

    Returns:
        Cleaned text without JSON markdown tags
    """
    if not text:
        return ""

    cleaned = text.replace("```json\n", "").replace("\n```", "")
    cleaned = cleaned.replace("```json", "").replace("```", "")
    return cleaned


def clean_markdown_tags(text: str) -> str:
    """Clean Markdown tags from text.

    Args:
        text: Text with potential markdown tags

    Returns:
        Cleaned text without markdown tags
    """
    if not text:
        return ""

    cleaned = text.replace("```markdown\n", "").replace("\n```", "")
    cleaned = cleaned.replace("```markdown", "").replace("```", "")
    return cleaned


def extract_html_from_content(content: str) -> str:
    """Attempt to extract HTML content from a response that might be wrapped in other formats.

    Args:
        content: The content to extract HTML from

    Returns:
        The extracted HTML, or a basic fallback if extraction fails
    """
    if not content:
        return ""

    # Try to find HTML between tags
    if "<html" in content and "</html>" in content:
        start = content.find("<html")
        end = content.find("</html>") + 7  # Include the closing tag
        return content[start:end]

    # Try to find div class="research-report"
    if '<div class="research-report"' in content and "</div>" in content:
        start = content.find('<div class="research-report"')
        # Find the last closing div
        last_div = content.rfind("</div>")
        if last_div > start:
            return content[start : last_div + 6]  # Include the closing tag

    # Look for code blocks
    if "```html" in content and "```" in content:
        start = content.find("```html") + 7
        end = content.find("```", start)
        if end > start:
            return content[start:end].strip()

    # Look for JSON with an "html" field
    try:
        parsed = json.loads(content)
        if isinstance(parsed, dict) and "html" in parsed:
            return parsed["html"]
    except:
        pass

    # If all extraction attempts fail, return the original content
    return content


def safe_json_loads(json_str: Optional[str]) -> Dict[str, Any]:
    """Safely parse JSON string.

    Args:
        json_str: JSON string to parse, can be None.

    Returns:
        Dict[str, Any]: Parsed JSON as dictionary or empty dict if parsing fails or input is None.
    """
    if json_str is None:
        # Optionally, log a warning here if None input is unexpected for certain call sites
        # logger.warning("safe_json_loads received None input.")
        return {}
    try:
        return json.loads(json_str)
    except (
        JSONDecodeError,
        TypeError,
    ):  # Catch TypeError if json_str is not a valid type for json.loads
        # Optionally, log the error and the problematic string (or its beginning)
        # logger.warning(f"Failed to decode JSON string: '{str(json_str)[:200]}...'", exc_info=True)
        return {}


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
