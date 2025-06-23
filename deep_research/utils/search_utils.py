import logging
import os
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from tavily import TavilyClient

try:
    from exa_py import Exa

    EXA_AVAILABLE = True
except ImportError:
    EXA_AVAILABLE = False
    Exa = None

from utils.llm_utils import get_structured_llm_output

# Import weave for optional decorators
try:
    import weave
    WEAVE_AVAILABLE = True
except ImportError:
    WEAVE_AVAILABLE = False
from utils.prompts import DEFAULT_SEARCH_QUERY_PROMPT
from utils.pydantic_models import SearchResult

logger = logging.getLogger(__name__)


class SearchProvider(Enum):
    TAVILY = "tavily"
    EXA = "exa"
    BOTH = "both"


class SearchEngineConfig:
    """Configuration for search engines"""

    def __init__(self):
        self.tavily_api_key = os.getenv("TAVILY_API_KEY")
        self.exa_api_key = os.getenv("EXA_API_KEY")
        self.default_provider = os.getenv("DEFAULT_SEARCH_PROVIDER", "tavily")
        self.enable_parallel_search = (
            os.getenv("ENABLE_PARALLEL_SEARCH", "false").lower() == "true"
        )


def get_search_client(provider: Union[str, SearchProvider]) -> Optional[Any]:
    """Get the appropriate search client based on provider."""
    if isinstance(provider, str):
        provider = SearchProvider(provider.lower())

    config = SearchEngineConfig()

    if provider == SearchProvider.TAVILY:
        if not config.tavily_api_key:
            raise ValueError("TAVILY_API_KEY environment variable not set")
        return TavilyClient(api_key=config.tavily_api_key)

    elif provider == SearchProvider.EXA:
        if not EXA_AVAILABLE:
            raise ImportError(
                "exa-py is not installed. Please install it with: pip install exa-py"
            )
        if not config.exa_api_key:
            raise ValueError("EXA_API_KEY environment variable not set")
        return Exa(config.exa_api_key)

    return None


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
        tavily_client = get_search_client(SearchProvider.TAVILY)

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


def exa_search(
    query: str,
    max_results: int = 3,
    cap_content_length: int = 20000,
    search_mode: str = "auto",
    include_highlights: bool = False,
) -> Dict[str, Any]:
    """Perform a search using the Exa API.

    Args:
        query: Search query
        max_results: Maximum number of results to return
        cap_content_length: Maximum length of content to return
        search_mode: Search mode ("neural", "keyword", or "auto")
        include_highlights: Whether to include highlights in results

    Returns:
        Dict[str, Any]: Search results from Exa in a format compatible with Tavily
    """
    try:
        exa_client = get_search_client(SearchProvider.EXA)

        # Configure content options
        text_options = {"max_characters": cap_content_length}

        kwargs = {
            "query": query,
            "num_results": max_results,
            "type": search_mode,  # "neural", "keyword", or "auto"
            "text": text_options,
        }

        if include_highlights:
            kwargs["highlights"] = {
                "highlights_per_url": 2,
                "num_sentences": 3,
            }

        response = exa_client.search_and_contents(**kwargs)

        # Extract cost information
        exa_cost = 0.0
        if hasattr(response, "cost_dollars") and hasattr(
            response.cost_dollars, "total"
        ):
            exa_cost = response.cost_dollars.total
            logger.info(
                f"Exa search cost for query '{query}': ${exa_cost:.4f}"
            )

        # Convert to standardized format compatible with Tavily
        results = {"query": query, "results": [], "exa_cost": exa_cost}

        for r in response.results:
            result_dict = {
                "url": r.url,
                "title": r.title or "",
                "snippet": "",
                "raw_content": getattr(r, "text", ""),
                "content": getattr(r, "text", ""),
            }

            # Add highlights as snippet if available
            if hasattr(r, "highlights") and r.highlights:
                result_dict["snippet"] = " ".join(r.highlights[:1])

            # Store additional metadata
            result_dict["_metadata"] = {
                "provider": "exa",
                "score": getattr(r, "score", None),
                "published_date": getattr(r, "published_date", None),
                "author": getattr(r, "author", None),
            }

            results["results"].append(result_dict)

        return results

    except Exception as e:
        logger.error(f"Error in Exa search: {e}")
        return {"query": query, "results": [], "error": str(e)}


def unified_search(
    query: str,
    provider: Union[str, SearchProvider, None] = None,
    max_results: int = 3,
    cap_content_length: int = 20000,
    search_mode: str = "auto",
    include_highlights: bool = False,
    compare_results: bool = False,
    **kwargs,
) -> Union[List[SearchResult], Dict[str, List[SearchResult]]]:
    """Unified search interface supporting multiple providers.

    Args:
        query: Search query
        provider: Search provider to use (tavily, exa, both)
        max_results: Maximum number of results
        cap_content_length: Maximum content length
        search_mode: Search mode for Exa ("neural", "keyword", "auto")
        include_highlights: Include highlights for Exa results
        compare_results: Return results from both providers separately

    Returns:
        List[SearchResult] or Dict mapping provider to results (when compare_results=True or provider="both")
    """
    # Use default provider if not specified
    if provider is None:
        config = SearchEngineConfig()
        provider = config.default_provider

    # Convert string to enum if needed
    if isinstance(provider, str):
        provider = SearchProvider(provider.lower())

    # Handle single provider case
    if provider == SearchProvider.TAVILY:
        results = tavily_search(
            query,
            max_results=max_results,
            cap_content_length=cap_content_length,
        )
        extracted, cost = extract_search_results(results, provider="tavily")
        return extracted if not compare_results else {"tavily": extracted}

    elif provider == SearchProvider.EXA:
        results = exa_search(
            query=query,
            max_results=max_results,
            cap_content_length=cap_content_length,
            search_mode=search_mode,
            include_highlights=include_highlights,
        )
        extracted, cost = extract_search_results(results, provider="exa")
        return extracted if not compare_results else {"exa": extracted}

    elif provider == SearchProvider.BOTH:
        # Run both searches
        tavily_results = tavily_search(
            query,
            max_results=max_results,
            cap_content_length=cap_content_length,
        )
        exa_results = exa_search(
            query=query,
            max_results=max_results,
            cap_content_length=cap_content_length,
            search_mode=search_mode,
            include_highlights=include_highlights,
        )

        # Extract results from both
        tavily_extracted, tavily_cost = extract_search_results(
            tavily_results, provider="tavily"
        )
        exa_extracted, exa_cost = extract_search_results(
            exa_results, provider="exa"
        )

        if compare_results:
            return {"tavily": tavily_extracted, "exa": exa_extracted}
        else:
            # Merge results, interleaving them
            merged = []
            max_len = max(len(tavily_extracted), len(exa_extracted))
            for i in range(max_len):
                if i < len(tavily_extracted):
                    merged.append(tavily_extracted[i])
                if i < len(exa_extracted):
                    merged.append(exa_extracted[i])
            return merged[:max_results]  # Limit to requested number

    else:
        raise ValueError(f"Unknown provider: {provider}")


def extract_search_results(
    search_results: Dict[str, Any], provider: str = "tavily"
) -> tuple[List[SearchResult], float]:
    """Extract SearchResult objects from provider-specific API responses.

    Args:
        search_results: Results from search API
        provider: Which provider the results came from

    Returns:
        Tuple of (List[SearchResult], float): List of converted SearchResult objects with standardized fields
          and the search cost (0.0 if not available).
          SearchResult is a Pydantic model defined in data_models.py that includes:
          - url: The URL of the search result
          - content: The raw content of the page
          - title: The title of the page
          - snippet: A brief snippet of the page content
    """
    results_list = []
    search_cost = search_results.get(
        "exa_cost", 0.0
    )  # Extract cost if present

    if "results" in search_results:
        for result in search_results["results"]:
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

                # Create SearchResult with provider metadata
                search_result = SearchResult(
                    url=url,
                    content=content,
                    title=title,
                    snippet=snippet,
                )

                # Add provider info to metadata if available
                if "_metadata" in result:
                    search_result.metadata = result["_metadata"]
                else:
                    search_result.metadata = {"provider": provider}

                results_list.append(search_result)

    # If we got the answer (Tavily specific), add it as a special result
    if (
        provider == "tavily"
        and "answer" in search_results
        and search_results["answer"]
    ):
        answer_text = search_results["answer"]
        results_list.append(
            SearchResult(
                url="tavily-generated-answer",
                content=f"Generated Answer based on search results:\n\n{answer_text}",
                title="Tavily Generated Answer",
                snippet=answer_text[:100] + "..."
                if len(answer_text) > 100
                else answer_text,
                metadata={"provider": "tavily", "type": "generated_answer"},
            )
        )
        logger.info("Added Tavily generated answer as a search result")

    return results_list, search_cost


# Conditional weave decorator
def _weave_op_if_available(func):
    """Conditionally apply weave.op decorator if weave is available."""
    if WEAVE_AVAILABLE:
        return weave.op()(func)
    return func

@_weave_op_if_available
def generate_search_query(
    sub_question: str,
    model: str = "openrouter/google/gemini-2.0-flash-lite-001",
    system_prompt: Optional[str] = None,
    tracking_provider: str = "weave",
    project: str = "deep-research",
) -> Dict[str, Any]:
    """Generate an optimized search query for a sub-question.

    Uses litellm for model inference via get_structured_llm_output.

    Args:
        sub_question: The sub-question to generate a search query for
        model: Model to use (with provider prefix)
        system_prompt: System prompt for the LLM, defaults to DEFAULT_SEARCH_QUERY_PROMPT
        tracking_provider: Experiment tracking provider (weave, langfuse, or none)
        project: Project name for LLM tracking

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
        tracking_provider=tracking_provider,
        project=project,
    )


def search_and_extract_results(
    query: str,
    max_results: int = 3,
    cap_content_length: int = 20000,
    max_retries: int = 2,
    provider: Optional[Union[str, SearchProvider]] = None,
    search_mode: str = "auto",
    include_highlights: bool = False,
) -> tuple[List[SearchResult], float]:
    """Perform a search and extract results in one step.

    Args:
        query: Search query
        max_results: Maximum number of results to return
        cap_content_length: Maximum length of content to return
        max_retries: Maximum number of retries in case of failure
        provider: Search provider to use (tavily, exa, both)
        search_mode: Search mode for Exa ("neural", "keyword", "auto")
        include_highlights: Include highlights for Exa results

    Returns:
        Tuple of (List of SearchResult objects, search cost)
    """
    results = []
    total_cost = 0.0
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

            # Determine if we're using Exa to track costs
            using_exa = False
            if provider:
                if isinstance(provider, str):
                    using_exa = provider.lower() in ["exa", "both"]
                else:
                    using_exa = provider in [
                        SearchProvider.EXA,
                        SearchProvider.BOTH,
                    ]
            else:
                config = SearchEngineConfig()
                using_exa = config.default_provider.lower() in ["exa", "both"]

            # Perform search based on provider
            if using_exa and provider != SearchProvider.BOTH:
                # Direct Exa search
                search_results = exa_search(
                    query=current_query,
                    max_results=max_results,
                    cap_content_length=cap_content_length,
                    search_mode=search_mode,
                    include_highlights=include_highlights,
                )
                results, cost = extract_search_results(
                    search_results, provider="exa"
                )
                total_cost += cost
            elif provider == SearchProvider.BOTH:
                # Search with both providers
                tavily_results = tavily_search(
                    current_query,
                    max_results=max_results,
                    cap_content_length=cap_content_length,
                )
                exa_results = exa_search(
                    query=current_query,
                    max_results=max_results,
                    cap_content_length=cap_content_length,
                    search_mode=search_mode,
                    include_highlights=include_highlights,
                )

                # Extract results from both
                tavily_extracted, _ = extract_search_results(
                    tavily_results, provider="tavily"
                )
                exa_extracted, exa_cost = extract_search_results(
                    exa_results, provider="exa"
                )
                total_cost += exa_cost

                # Merge results
                results = []
                max_len = max(len(tavily_extracted), len(exa_extracted))
                for i in range(max_len):
                    if i < len(tavily_extracted):
                        results.append(tavily_extracted[i])
                    if i < len(exa_extracted):
                        results.append(exa_extracted[i])
                results = results[:max_results]
            else:
                # Tavily search or unified search
                results = unified_search(
                    query=current_query,
                    provider=provider,
                    max_results=max_results,
                    cap_content_length=cap_content_length,
                    search_mode=search_mode,
                    include_highlights=include_highlights,
                )

                # Handle case where unified_search returns a dict
                if isinstance(results, dict):
                    all_results = []
                    for provider_results in results.values():
                        all_results.extend(provider_results)
                    results = all_results[:max_results]

            # Check if we got results with actual content
            if results:
                # Count results with non-empty content
                content_results = sum(1 for r in results if r.content.strip())

                if content_results >= max(1, len(results) // 2):
                    logger.info(
                        f"Found {content_results}/{len(results)} results with content"
                    )
                    return results, total_cost
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
                return results, total_cost

        except Exception as e:
            if retry_count < max_retries:
                logger.warning(
                    f"Search failed with error: {e}. Retrying... ({retry_count + 1}/{max_retries})"
                )
                retry_count += 1
            else:
                logger.error(f"Search failed after {max_retries} retries: {e}")
                return [], 0.0

    # If we've exhausted all retries, return the best results we have
    return results, total_cost
