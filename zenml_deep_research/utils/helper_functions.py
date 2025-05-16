import json
import os
import yaml
from json.decoder import JSONDecodeError
from typing import List, Dict, Any, Optional


def remove_reasoning_from_output(output: str) -> str:
    """Remove the reasoning portion from LLM output.

    Args:
        output: Raw output from LLM that may contain reasoning

    Returns:
        Cleaned output without the reasoning section
    """
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
    cleaned = text.replace("```markdown\n", "").replace("\n```", "")
    cleaned = cleaned.replace("```markdown", "").replace("```", "")
    return cleaned


def safe_json_loads(json_str: str) -> Dict[str, Any]:
    """Safely parse JSON string.

    Args:
        json_str: JSON string to parse

    Returns:
        Parsed JSON as dictionary or empty dict if parsing fails
    """
    try:
        return json.loads(json_str)
    except JSONDecodeError:
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


def tavily_search(
    query: str,
    include_raw_content: bool = True,
    max_results: int = 3,
    cap_content_length: int = 20000,
) -> Dict[str, Any]:
    """Perform a search using the Tavily API.

    Args:
        query: Search query
        include_raw_content: Whether to include raw content in results
        max_results: Maximum number of results to return
        cap_content_length: Maximum length of content to return

    Returns:
        Search results from Tavily
    """
    try:
        from tavily import TavilyClient

        # Get API key directly from environment variables
        api_key = os.environ.get("TAVILY_API_KEY", "")
        if not api_key:
            raise ValueError("TAVILY_API_KEY environment variable not set")

        tavily_client = TavilyClient(api_key=api_key)

        results = tavily_client.search(
            query=query,
            include_raw_content=include_raw_content,
            max_results=max_results,
        )

        # Cap content length if specified
        if cap_content_length > 0 and "results" in results:
            for result in results["results"]:
                if "raw_content" in result and result["raw_content"]:
                    result["raw_content"] = result["raw_content"][
                        :cap_content_length
                    ]

        return results
    except Exception as e:
        # Return an error structure that's compatible with our expected format
        return {"query": query, "results": [], "error": str(e)}
