import logging
import os
from typing import Any, Dict, List, Optional

from utils.llm_utils import get_structured_llm_output
from utils.prompts import DEFAULT_SEARCH_QUERY_PROMPT

# Import Pydantic model instead of dataclass
from utils.pydantic_models import SearchResult

logger = logging.getLogger(__name__)


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
        Dict[str, Any]: Search results from Tavily in the following format:
            {
                "query": str,  # The original query
                "results": List[Dict],  # List of search result objects
                "error": str,  # Error message (if an error occurred, otherwise omitted)
            }

            Each result in "results" has the following structure:
            {
                "url": str,  # URL of the search result
                "raw_content": str,  # Raw content of the page (if include_raw_content=True)
                "title": str,  # Title of the page
                "snippet": str,  # Snippet of the page content
            }
    """
    try:
        from tavily import TavilyClient

        # Get API key directly from environment variables
        api_key = os.environ.get("TAVILY_API_KEY", "")
        if not api_key:
            logger.error("TAVILY_API_KEY environment variable not set")
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
        logger.error(f"Error in Tavily search: {e}")
        # Return an error structure that's compatible with our expected format
        return {"query": query, "results": [], "error": str(e)}


def extract_search_results(
    tavily_results: Dict[str, Any],
) -> List[SearchResult]:
    """Extract SearchResult objects from Tavily API response.

    Args:
        tavily_results: Results from tavily_search function

    Returns:
        List[SearchResult]: List of converted SearchResult objects with standardized fields.
          SearchResult is a Pydantic model defined in data_models.py that includes:
          - url: The URL of the search result
          - content: The raw content of the page
          - title: The title of the page
          - snippet: A brief snippet of the page content
    """
    results_list = []
    if "results" in tavily_results:
        for result in tavily_results["results"]:
            if "url" in result and "raw_content" in result:
                results_list.append(
                    SearchResult(
                        url=result["url"],
                        content=result["raw_content"],
                        title=result.get("title", ""),
                        snippet=result.get("snippet", ""),
                    )
                )
    return results_list


def generate_search_query(
    sub_question: str,
    model: str = "sambanova/Llama-4-Maverick-17B-128E-Instruct",
    system_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate an optimized search query for a sub-question.

    Uses litellm for model inference via get_structured_llm_output.

    Args:
        sub_question: The sub-question to generate a search query for
        model: Model to use (with provider prefix)
        system_prompt: System prompt for the LLM, defaults to DEFAULT_SEARCH_QUERY_PROMPT

    Returns:
        Dictionary with search query and reasoning
    """
    if system_prompt is None:
        system_prompt = DEFAULT_SEARCH_QUERY_PROMPT

    fallback_response = {"search_query": sub_question, "reasoning": ""}

    return get_structured_llm_output(
        prompt=sub_question,
        system_prompt=system_prompt,
        model=model,
        fallback_response=fallback_response,
    )


def search_and_extract_results(
    query: str,
    max_results: int = 3,
    cap_content_length: int = 20000,
) -> List[SearchResult]:
    """Perform a search and extract results in one step.

    Args:
        query: Search query
        max_results: Maximum number of results to return
        cap_content_length: Maximum length of content to return

    Returns:
        List of SearchResult objects
    """
    tavily_results = tavily_search(
        query=query,
        max_results=max_results,
        cap_content_length=cap_content_length,
    )

    return extract_search_results(tavily_results)
