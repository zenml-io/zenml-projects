import logging
import os
import openai
import json
from typing import Annotated, Dict, Any
from zenml import step

from materializers.research_state_materializer import ResearchStateMaterializer
from utils.data_models import ResearchState, SearchResult
from utils.helper_functions import (
    remove_reasoning_from_output,
    clean_json_tags,
    safe_json_loads,
    tavily_search,
)

logger = logging.getLogger(__name__)

# System prompt for generating search queries
SEARCH_QUERY_PROMPT = """
You are a Deep Research assistant. Your task is to create an effective web search query for the given research sub-question.

A good search query should:
1. Be concise and focused
2. Use specific keywords related to the sub-question
3. Be formulated to retrieve accurate and relevant information
4. Avoid ambiguous terms or overly broad language

Consider what would most effectively retrieve information from search engines to answer the specific sub-question.

Format the output in json with the following json schema definition:

<OUTPUT JSON SCHEMA>
{
  "type": "object",
  "properties": {
    "search_query": {"type": "string"},
    "reasoning": {"type": "string"}
  }
}
</OUTPUT JSON SCHEMA>

Make sure that the output is a json object with an output json schema defined above.
Only return the json object, no explanation or additional text.
"""


@step(output_materializers=ResearchStateMaterializer)
def parallel_information_gathering_step(
    state: ResearchState,
    sambanova_base_url: str = "https://api.sambanova.ai/v1",
    llm_model: str = "Meta-Llama-3.3-70B-Instruct",
    num_results_per_search: int = 3,
    cap_search_length: int = 20000,
    system_prompt: str = SEARCH_QUERY_PROMPT,
) -> Annotated[ResearchState, "updated_state"]:
    """Fetch data for each sub-question in parallel.

    Args:
        state: The current research state
        sambanova_base_url: SambaNova API base URL
        llm_model: The model to use for search query generation
        num_results_per_search: Number of results to fetch per search
        cap_search_length: Maximum length of content to process from search results
        system_prompt: System prompt for generating search queries

    Returns:
        Updated research state with search results
    """
    logger.info(
        f"Gathering information for {len(state.sub_questions)} sub-questions"
    )

    # Get API key from environment variables
    sambanova_api_key = os.environ.get("SAMBANOVA_API_KEY", "")
    if not sambanova_api_key:
        logger.error("SAMBANOVA_API_KEY environment variable not set")
        raise ValueError("SAMBANOVA_API_KEY environment variable not set")

    # Initialize OpenAI client
    openai_client = openai.OpenAI(
        api_key=sambanova_api_key, base_url=sambanova_base_url
    )

    # Dictionary to store results
    search_results = {}

    # Process each sub-question
    for i, sub_question in enumerate(state.sub_questions):
        logger.info(
            f"Researching sub-question {i + 1}/{len(state.sub_questions)}: {sub_question}"
        )

        # Generate search query
        search_query_data = _generate_search_query(
            sub_question=sub_question,
            openai_client=openai_client,
            model=llm_model,
            system_prompt=system_prompt,
        )
        search_query = search_query_data.get(
            "search_query", f"research about {sub_question}"
        )

        # Perform search
        logger.info(f"Performing search with query: {search_query}")
        tavily_results = tavily_search(
            query=search_query,
            max_results=num_results_per_search,
            cap_content_length=cap_search_length,
        )

        # Store search results
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

        search_results[sub_question] = results_list

    logger.info(f"Completed information gathering for all sub-questions")

    # Update the state with the new search results
    state.update_search_results(search_results)

    return state


def _generate_search_query(
    sub_question: str,
    openai_client: openai.OpenAI,
    model: str,
    system_prompt: str,
) -> Dict[str, Any]:
    """Generate an optimized search query for a sub-question.

    Args:
        sub_question: The sub-question to generate a search query for
        openai_client: OpenAI client
        model: Model to use
        system_prompt: System prompt for the LLM

    Returns:
        Dictionary with search query and reasoning
    """
    try:
        response = openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": sub_question},
            ],
        )

        content = response.choices[0].message.content
        content = remove_reasoning_from_output(content)
        content = clean_json_tags(content)

        result = safe_json_loads(content)

        if not result or "search_query" not in result:
            # Fallback if parsing fails
            return {"search_query": sub_question, "reasoning": ""}

        return result

    except Exception as e:
        logger.error(f"Error generating search query: {e}")
        return {"search_query": sub_question, "reasoning": ""}
