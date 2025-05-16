import json
import logging
import os
import openai
from typing import Dict, Any, List
from zenml import step

from utils.data_models import State, Search
from utils.helper_functions import (
    remove_reasoning_from_output, 
    clean_json_tags, 
    safe_json_loads,
    tavily_search
)

logger = logging.getLogger(__name__)

# System prompts for different tasks
FIRST_SEARCH_PROMPT = """
You are a Deep Research assistant. You will be given a paragraph in a report, it's title and expected content in the following json schema definition:

<INPUT JSON SCHEMA>
{
  "type": "object",
  "properties": {
    "title": {"type": "string"},
    "content": {"type": "string"}
  }
}
</INPUT JSON SCHEMA>

You can use a web search tool that takes a 'search_query' as parameter.
Your job is to reflect on the topic and provide the most optimal web search query to enrich your current knowledge.
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

FIRST_SUMMARY_PROMPT = """
You are a Deep Research assistant. You will be given a search query, search results and the paragraph a report that you are researching following json schema definition:

<INPUT JSON SCHEMA>
{
  "type": "object",
  "properties": {
    "title": {"type": "string"},
    "content": {"type": "string"},
    "search_query": {"type": "string"},
    "search_results": {
      "type": "array",
      "items": {"type": "string"}
    }
  }
}
</INPUT JSON SCHEMA>

Your job is to write the paragraph as a researcher using the search results to align with the paragraph topic and structure it properly to be included in the report.
Format the output in json with the following json schema definition:

<OUTPUT JSON SCHEMA>
{
  "type": "object",
  "properties": {
    "paragraph_latest_state": {"type": "string"}
  }
}
</OUTPUT JSON SCHEMA>

Make sure that the output is a json object with an output json schema defined above.
Only return the json object, no explanation or additional text.
"""

REFLECTION_PROMPT = """
You are a Deep Research assistant. You are responsible for constructing comprehensive paragraphs for a research report. You will be provided paragraph title and planned content summary, also the latest state of the paragraph that you have already created all in the following json schema definition:

<INPUT JSON SCHEMA>
{
  "type": "object",
  "properties": {
    "title": {"type": "string"},
    "content": {"type": "string"},
    "paragraph_latest_state": {"type": "string"}
  }
}
</INPUT JSON SCHEMA>

You can use a web search tool that takes a 'search_query' as parameter.
Your job is to reflect on the current state of the paragraph text and think if you haven't missed some critical aspect of the topic and provide the most optimal web search query to enrich the latest state.
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

REFLECTION_SUMMARY_PROMPT = """
You are a Deep Research assistant.
You will be given a search query, search results, paragraph title and expected content for the paragraph in a report that you are researching.
You are iterating on the paragraph and the latest state of the paragraph is also provided.
The data will be in the following json schema definition:

<INPUT JSON SCHEMA>
{
  "type": "object",
  "properties": {
    "title": {"type": "string"},
    "content": {"type": "string"},
    "search_query": {"type": "string"},
    "search_results": {
      "type": "array",
      "items": {"type": "string"}
    },
    "paragraph_latest_state": {"type": "string"}
  }
}
</INPUT JSON SCHEMA>

Your job is to enrich the current latest state of the paragraph with the search results considering expected content.
Do not remove key information from the latest state and try to enrich it, only add information that is missing.
Structure the paragraph properly to be included in the report.
Format the output in json with the following json schema definition:

<OUTPUT JSON SCHEMA>
{
  "type": "object",
  "properties": {
    "updated_paragraph_latest_state": {"type": "string"}
  }
}
</OUTPUT JSON SCHEMA>

Make sure that the output is a json object with an output json schema defined above.
Only return the json object, no explanation or additional text.
"""

def _get_first_search_query(
    title: str, 
    content: str, 
    openai_client: openai.OpenAI,
    model: str,
    system_prompt: str
) -> Dict[str, Any]:
    """Get the first search query for a paragraph.
    
    Args:
        title: Paragraph title
        content: Paragraph content description
        openai_client: OpenAI client
        model: Model to use
        system_prompt: System prompt for the LLM
        
    Returns:
        Dictionary with search query and reasoning
    """
    message = json.dumps({"title": title, "content": content})
    
    try:
        response = openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message}
            ]
        )
        
        content = response.choices[0].message.content
        content = remove_reasoning_from_output(content)
        content = clean_json_tags(content)
        
        result = safe_json_loads(content)
        
        if not result or "search_query" not in result:
            # Fallback if parsing fails
            return {"search_query": f"research about {title}", "reasoning": ""}
            
        return result
    
    except Exception as e:
        logger.error(f"Error getting first search query: {e}")
        return {"search_query": f"research about {title}", "reasoning": ""}

def _summarize_search_results(
    title: str,
    content: str,
    search_query: str,
    search_results: List[str],
    openai_client: openai.OpenAI,
    model: str,
    system_prompt: str
) -> str:
    """Summarize search results into a paragraph.
    
    Args:
        title: Paragraph title
        content: Paragraph content description
        search_query: The query used for search
        search_results: List of search result content
        openai_client: OpenAI client
        model: Model to use
        system_prompt: System prompt for the LLM
        
    Returns:
        Summarized paragraph content
    """
    message = {
        "title": title,
        "content": content,
        "search_query": search_query,
        "search_results": search_results
    }
    
    try:
        response = openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(message)}
            ]
        )
        
        content = response.choices[0].message.content
        content = remove_reasoning_from_output(content)
        content = clean_json_tags(content)
        
        result = safe_json_loads(content)
        
        if not result or "paragraph_latest_state" not in result:
            # If parsing failed, try to use the response directly
            return content
            
        return result.get("paragraph_latest_state", "")
    
    except Exception as e:
        logger.error(f"Error summarizing search results: {e}")
        return f"Research on {title} based on {search_query} was inconclusive."

def _get_reflection_search_query(
    title: str,
    content: str,
    paragraph_latest_state: str,
    openai_client: openai.OpenAI,
    model: str,
    system_prompt: str
) -> Dict[str, Any]:
    """Get a search query based on reflection of current paragraph state.
    
    Args:
        title: Paragraph title
        content: Paragraph content description
        paragraph_latest_state: Current paragraph content
        openai_client: OpenAI client
        model: Model to use
        system_prompt: System prompt for the LLM
        
    Returns:
        Dictionary with search query and reasoning
    """
    message = {
        "title": title,
        "content": content,
        "paragraph_latest_state": paragraph_latest_state
    }
    
    try:
        response = openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(message)}
            ]
        )
        
        content = response.choices[0].message.content
        content = remove_reasoning_from_output(content)
        content = clean_json_tags(content)
        
        result = safe_json_loads(content)
        
        if not result or "search_query" not in result:
            # Fallback if parsing fails
            return {"search_query": f"additional information about {title}", "reasoning": ""}
            
        return result
    
    except Exception as e:
        logger.error(f"Error getting reflection search query: {e}")
        return {"search_query": f"additional information about {title}", "reasoning": ""}

def _update_paragraph_with_reflection(
    title: str,
    content: str,
    search_query: str,
    search_results: List[str],
    paragraph_latest_state: str,
    openai_client: openai.OpenAI,
    model: str,
    system_prompt: str
) -> str:
    """Update paragraph with search results from reflection.
    
    Args:
        title: Paragraph title
        content: Paragraph content description
        search_query: The query used for search
        search_results: List of search result content
        paragraph_latest_state: Current paragraph content
        openai_client: OpenAI client
        model: Model to use
        system_prompt: System prompt for the LLM
        
    Returns:
        Updated paragraph content
    """
    message = {
        "title": title,
        "content": content,
        "search_query": search_query,
        "search_results": search_results,
        "paragraph_latest_state": paragraph_latest_state
    }
    
    try:
        response = openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(message)}
            ]
        )
        
        content = response.choices[0].message.content
        content = remove_reasoning_from_output(content)
        content = clean_json_tags(content)
        
        result = safe_json_loads(content)
        
        if not result or "updated_paragraph_latest_state" not in result:
            # If parsing failed, try to use the content directly
            return content
            
        return result.get("updated_paragraph_latest_state", paragraph_latest_state)
    
    except Exception as e:
        logger.error(f"Error updating paragraph with reflection: {e}")
        return paragraph_latest_state  # Return unchanged if error occurs

@step
def paragraph_research_step(
    current_state: State,
    sambanova_base_url: str = "https://api.sambanova.ai/v1",
    llm_regular_model: str = "Meta-Llama-3.3-70B-Instruct",
    num_reflections: int = 2,
    num_results_per_search: int = 3,
    cap_search_length: int = 20000,
    first_search_prompt: str = FIRST_SEARCH_PROMPT,
    first_summary_prompt: str = FIRST_SUMMARY_PROMPT,
    reflection_prompt: str = REFLECTION_PROMPT,
    reflection_summary_prompt: str = REFLECTION_SUMMARY_PROMPT
) -> State:
    """Research each paragraph in the report.
    
    Args:
        current_state: Current state with report structure
        sambanova_base_url: SambaNova API base URL
        llm_regular_model: The model to use for regular LLM operations
        num_reflections: Number of reflection cycles to perform
        num_results_per_search: Number of results to fetch per search
        cap_search_length: Maximum length of content to process from search results
        first_search_prompt: Prompt for generating the first search query
        first_summary_prompt: Prompt for summarizing the first search results
        reflection_prompt: Prompt for generating reflection search queries
        reflection_summary_prompt: Prompt for updating paragraphs with reflection results
        
    Returns:
        Updated state with researched paragraphs
    """
    logger.info(f"Starting research for {len(current_state.paragraphs)} paragraphs")
    
    # Get API key directly from environment variables
    sambanova_api_key = os.environ.get("SAMBANOVA_API_KEY", "")
    if not sambanova_api_key:
        logger.error("SAMBANOVA_API_KEY environment variable not set")
        raise ValueError("SAMBANOVA_API_KEY environment variable not set")
    
    # Initialize OpenAI client
    openai_client = openai.OpenAI(
        api_key=sambanova_api_key,
        base_url=sambanova_base_url
    )
    
    # Make a copy of the state to avoid modifying the input state
    researched_state = State(
        report_title=current_state.report_title,
        query=current_state.query,
        paragraphs=current_state.paragraphs.copy()
    )
    
    # Process each paragraph
    for i, paragraph in enumerate(researched_state.paragraphs):
        logger.info(f"Researching paragraph {i+1}/{len(researched_state.paragraphs)}: {paragraph.title}")
        
        # Step 1: Generate first search query
        search_query_data = _get_first_search_query(
            title=paragraph.title,
            content=paragraph.content,
            openai_client=openai_client,
            model=llm_regular_model,
            system_prompt=first_search_prompt
        )
        search_query = search_query_data.get("search_query", f"research about {paragraph.title}")
        
        # Step 2: Perform first search
        logger.info(f"Performing first search with query: {search_query}")
        search_results = tavily_search(
            query=search_query,
            max_results=num_results_per_search,
            cap_content_length=cap_search_length
        )
        
        # Step 3: Add search results to search history
        if "results" in search_results:
            for result in search_results["results"]:
                if "url" in result and "raw_content" in result:
                    paragraph.research.search_history.append(
                        Search(url=result["url"], content=result["raw_content"])
                    )
        
        # Step 4: Summarize first search results
        raw_contents = [
            result.get("raw_content", "") 
            for result in search_results.get("results", [])
            if result.get("raw_content")
        ]
        
        paragraph.research.latest_summary = _summarize_search_results(
            title=paragraph.title,
            content=paragraph.content,
            search_query=search_results.get("query", search_query),
            search_results=raw_contents,
            openai_client=openai_client,
            model=llm_regular_model,
            system_prompt=first_summary_prompt
        )
        
        # Step 5: Perform reflection iterations
        for j in range(num_reflections):
            logger.info(f"Performing reflection {j+1}/{num_reflections} for paragraph {i+1}")
            
            # Step 5a: Generate reflection search query
            reflection_query_data = _get_reflection_search_query(
                title=paragraph.title,
                content=paragraph.content,
                paragraph_latest_state=paragraph.research.latest_summary,
                openai_client=openai_client,
                model=llm_regular_model,
                system_prompt=reflection_prompt
            )
            reflection_query = reflection_query_data.get("search_query", f"additional info about {paragraph.title}")
            
            # Step 5b: Perform reflection search
            logger.info(f"Performing reflection search with query: {reflection_query}")
            reflection_results = tavily_search(
                query=reflection_query,
                max_results=num_results_per_search,
                cap_content_length=cap_search_length
            )
            
            # Step 5c: Add reflection search results to search history
            if "results" in reflection_results:
                for result in reflection_results["results"]:
                    if "url" in result and "raw_content" in result:
                        paragraph.research.search_history.append(
                            Search(url=result["url"], content=result["raw_content"])
                        )
            
            # Step 5d: Update paragraph with reflection search results
            raw_contents = [
                result.get("raw_content", "") 
                for result in reflection_results.get("results", [])
                if result.get("raw_content")
            ]
            
            paragraph.research.latest_summary = _update_paragraph_with_reflection(
                title=paragraph.title,
                content=paragraph.content,
                search_query=reflection_results.get("query", reflection_query),
                search_results=raw_contents,
                paragraph_latest_state=paragraph.research.latest_summary,
                openai_client=openai_client,
                model=llm_regular_model,
                system_prompt=reflection_summary_prompt
            )
            
            paragraph.research.reflection_iteration += 1
    
    logger.info("Completed research for all paragraphs")
    return researched_state 