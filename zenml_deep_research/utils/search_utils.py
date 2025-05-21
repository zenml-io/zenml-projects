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

        # First try with advanced search
        results = tavily_client.search(
            query=query,
            include_raw_content=include_raw_content,
            max_results=max_results,
            search_depth="advanced",  # Use advanced search for better results
            include_domains=[],  # No domain restrictions
            exclude_domains=[],  # No exclusions
            include_answer=False,  # We don't need the answer field
            include_images=False,  # We don't need images
            # Note: 'include_snippets' is not a supported parameter
        )

        # Check if we got good results (with non-None and non-empty content)
        if include_raw_content and "results" in results:
            bad_content_count = sum(
                1
                for r in results["results"]
                if "raw_content" in r
                and (
                    r["raw_content"] is None or r["raw_content"].strip() == ""
                )
            )

            # If more than half of results have bad content, try a different approach
            if bad_content_count > len(results["results"]) / 2:
                logger.warning(
                    f"{bad_content_count}/{len(results['results'])} results have None or empty content. "
                    "Trying to use 'content' field instead of 'raw_content'..."
                )

                # Try to use the 'content' field which comes by default
                for result in results["results"]:
                    if (
                        "raw_content" in result
                        and (
                            result["raw_content"] is None
                            or result["raw_content"].strip() == ""
                        )
                    ) and "content" in result:
                        result["raw_content"] = result["content"]
                        logger.info(
                            f"Using 'content' field as 'raw_content' for URL {result.get('url', 'unknown')}"
                        )

                # Re-check after our fix
                bad_content_count = sum(
                    1
                    for r in results["results"]
                    if "raw_content" in r
                    and (
                        r["raw_content"] is None
                        or r["raw_content"].strip() == ""
                    )
                )

                if bad_content_count > 0:
                    logger.warning(
                        f"Still have {bad_content_count}/{len(results['results'])} results with bad content after fixes."
                    )

                # Try alternative approach - search with 'include_answer=True'
                try:
                    # Search with include_answer=True which may give us better content
                    logger.info(
                        "Trying alternative search with include_answer=True"
                    )
                    alt_results = tavily_client.search(
                        query=query,
                        include_raw_content=include_raw_content,
                        max_results=max_results,
                        search_depth="advanced",
                        include_domains=[],
                        exclude_domains=[],
                        include_answer=True,  # Include answer this time
                        include_images=False,
                    )

                    # Check if we got any improved content
                    if "results" in alt_results:
                        # Create a merged results set taking the best content
                        for i, result in enumerate(alt_results["results"]):
                            if i < len(results["results"]):
                                if (
                                    "raw_content" in result
                                    and result["raw_content"]
                                    and (
                                        results["results"][i].get(
                                            "raw_content"
                                        )
                                        is None
                                        or results["results"][i]
                                        .get("raw_content", "")
                                        .strip()
                                        == ""
                                    )
                                ):
                                    # Replace the bad content with better content from alt_results
                                    results["results"][i]["raw_content"] = (
                                        result["raw_content"]
                                    )
                                    logger.info(
                                        f"Replaced bad content with better content from alternative search for URL {result.get('url', 'unknown')}"
                                    )

                        # If answer is available, add it as a special result
                        if "answer" in alt_results and alt_results["answer"]:
                            answer_text = alt_results["answer"]
                            answer_result = {
                                "url": "tavily-generated-answer",
                                "title": "Generated Answer",
                                "raw_content": f"Generated Answer based on search results:\n\n{answer_text}",
                                "content": answer_text,
                            }
                            results["results"].append(answer_result)
                            logger.info(
                                "Added Tavily generated answer as additional search result"
                            )

                except Exception as alt_error:
                    logger.warning(
                        f"Failed to get better results with alternative search: {alt_error}"
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
            if "url" in result:
                # Get fields with defaults
                url = result["url"]
                title = result.get("title", "")

                # Try to extract the best content available:
                # 1. First try raw_content (if we requested it)
                # 2. Then try regular content (always available)
                # 3. Then try to use snippet combined with title
                # 4. Last resort: use just title

                raw_content = result.get("raw_content", None)
                regular_content = result.get("content", "")
                snippet = result.get("snippet", "")

                # Set our final content - prioritize raw_content if available and not None
                if raw_content is not None and raw_content.strip():
                    content = raw_content
                # Next best is the regular content field
                elif regular_content and regular_content.strip():
                    content = regular_content
                    logger.info(
                        f"Using 'content' field for URL {url} because raw_content was not available"
                    )
                # Try to create a usable content from snippet and title
                elif snippet:
                    content = f"Title: {title}\n\nContent: {snippet}"
                    logger.warning(
                        f"Using title and snippet as content fallback for {url}"
                    )
                # Last resort - just use the title
                elif title:
                    content = (
                        f"Title: {title}\n\nNo content available for this URL."
                    )
                    logger.warning(
                        f"Using only title as content fallback for {url}"
                    )
                # Nothing available
                else:
                    content = ""
                    logger.warning(
                        f"No content available for URL {url}, using empty string"
                    )

                results_list.append(
                    SearchResult(
                        url=url,
                        content=content,
                        title=title,
                        snippet=snippet,
                    )
                )

    # If we got the answer, add it as a special result
    if "answer" in tavily_results and tavily_results["answer"]:
        answer_text = tavily_results["answer"]
        results_list.append(
            SearchResult(
                url="tavily-generated-answer",
                content=f"Generated Answer based on search results:\n\n{answer_text}",
                title="Tavily Generated Answer",
                snippet=answer_text[:100] + "..."
                if len(answer_text) > 100
                else answer_text,
            )
        )
        logger.info("Added Tavily generated answer as a search result")

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
    max_retries: int = 2,
) -> List[SearchResult]:
    """Perform a search and extract results in one step.

    Args:
        query: Search query
        max_results: Maximum number of results to return
        cap_content_length: Maximum length of content to return
        max_retries: Maximum number of retries in case of failure

    Returns:
        List of SearchResult objects
    """
    results = []
    retry_count = 0

    # List of alternative query formats to try if the original query fails
    # to yield good results with non-None content
    alternative_queries = [
        query,  # Original query first
        f'"{query}"',  # Try exact phrase matching
        f"about {query}",  # Try broader context
        f"research on {query}",  # Try research-oriented results
        query.replace(" OR ", " "),  # Try without OR operator
    ]

    while retry_count <= max_retries and retry_count < len(
        alternative_queries
    ):
        try:
            current_query = alternative_queries[retry_count]
            logger.info(
                f"Searching with query ({retry_count + 1}/{max_retries + 1}): {current_query}"
            )

            tavily_results = tavily_search(
                query=current_query,
                max_results=max_results,
                cap_content_length=cap_content_length,
            )

            results = extract_search_results(tavily_results)

            # Check if we got results with actual content
            if results:
                # Count results with non-empty content
                content_results = sum(1 for r in results if r.content.strip())

                if content_results >= max(1, len(results) // 2):
                    logger.info(
                        f"Found {content_results}/{len(results)} results with content"
                    )
                    return results
                else:
                    logger.warning(
                        f"Only found {content_results}/{len(results)} results with content. "
                        f"Trying alternative query..."
                    )

            # If we didn't get good results but haven't hit max retries yet, try again
            if retry_count < max_retries:
                logger.warning(
                    f"Inadequate search results. Retrying with alternative query... ({retry_count + 1}/{max_retries})"
                )
                retry_count += 1
            else:
                # If we're out of retries, return whatever we have
                logger.warning(
                    f"Out of retries. Returning best results found ({len(results)} results)."
                )
                return results

        except Exception as e:
            if retry_count < max_retries:
                logger.warning(
                    f"Search failed with error: {e}. Retrying... ({retry_count + 1}/{max_retries})"
                )
                retry_count += 1
            else:
                logger.error(f"Search failed after {max_retries} retries: {e}")
                return []

    # If we've exhausted all retries, return the best results we have
    return results
