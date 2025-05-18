import logging
import os
import openai
import json
from typing import Annotated, Dict, List, Any, Tuple
from zenml import step
from materializers.research_state_materializer import ResearchStateMaterializer

from utils.data_models import (
    ResearchState,
    SynthesizedInfo,
    ReflectionMetadata,
)
from utils.helper_functions import (
    remove_reasoning_from_output,
    clean_json_tags,
    safe_json_loads,
    tavily_search,
)

logger = logging.getLogger(__name__)

# System prompt for self-critique and reflection
REFLECTION_PROMPT = """
You are a Deep Research assistant with the ability to critique and improve your own research. You will be given:
1. The main research query
2. The sub-questions explored so far
3. The synthesized information for each sub-question
4. Any viewpoint analysis performed

Your task is to critically evaluate this research and identify:
1. Areas where the research is incomplete or has gaps
2. Questions that are important but not yet answered
3. Aspects where additional evidence or depth would significantly improve the research
4. Potential biases or limitations in the current findings

Be constructively critical and identify the most important improvements that would substantially enhance the research.

Format the output in json with the following json schema definition:

<OUTPUT JSON SCHEMA>
{
  "type": "object",
  "properties": {
    "critique": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "area": {"type": "string"},
          "issue": {"type": "string"},
          "importance": {"type": "string", "enum": ["high", "medium", "low"]}
        }
      }
    },
    "additional_questions": {
      "type": "array",
      "items": {"type": "string"}
    },
    "recommended_search_queries": {
      "type": "array",
      "items": {"type": "string"}
    }
  }
}
</OUTPUT JSON SCHEMA>

Make sure that the output is a json object with an output json schema defined above.
Only return the json object, no explanation or additional text.
"""

# System prompt for additional information synthesis
ADDITIONAL_SYNTHESIS_PROMPT = """
You are a Deep Research assistant. You will be given:
1. The original synthesized information on a research topic
2. New information from additional research
3. A critique of the original synthesis

Your task is to enhance the original synthesis by incorporating the new information and addressing the critique.
The updated synthesis should:
1. Integrate new information seamlessly 
2. Address gaps identified in the critique
3. Maintain a balanced, comprehensive, and accurate representation
4. Preserve the strengths of the original synthesis

Format the output in json with the following json schema definition:

<OUTPUT JSON SCHEMA>
{
  "type": "object",
  "properties": {
    "enhanced_synthesis": {"type": "string"},
    "improvements_made": {
      "type": "array",
      "items": {"type": "string"}
    },
    "remaining_limitations": {"type": "string"}
  }
}
</OUTPUT JSON SCHEMA>

Make sure that the output is a json object with an output json schema defined above.
Only return the json object, no explanation or additional text.
"""


@step(output_materializers=ResearchStateMaterializer)
def iterative_reflection_step(
    state: ResearchState,
    max_additional_searches: int = 2,
    num_results_per_search: int = 3,
    cap_search_length: int = 20000,
    sambanova_base_url: str = "https://api.sambanova.ai/v1",
    llm_model: str = "DeepSeek-R1-Distill-Llama-70B",
    reflection_prompt: str = REFLECTION_PROMPT,
    additional_synthesis_prompt: str = ADDITIONAL_SYNTHESIS_PROMPT,
) -> Annotated[ResearchState, "updated_state"]:
    """Perform iterative reflection on the research, identifying gaps and improving it.

    Args:
        state: The current research state
        max_additional_searches: Maximum number of additional searches to perform
        num_results_per_search: Number of results to fetch per search
        cap_search_length: Maximum length of content to process from search results
        sambanova_base_url: SambaNova API base URL
        llm_model: The model to use for reflection
        reflection_prompt: System prompt for the reflection
        additional_synthesis_prompt: System prompt for incorporating additional information

    Returns:
        Updated research state with enhanced information and reflection metadata
    """
    logger.info("Starting iterative reflection on research")

    # Get API key from environment variables
    sambanova_api_key = os.environ.get("SAMBANOVA_API_KEY", "")
    if not sambanova_api_key:
        logger.error("SAMBANOVA_API_KEY environment variable not set")
        raise ValueError("SAMBANOVA_API_KEY environment variable not set")

    # Initialize OpenAI client
    openai_client = openai.OpenAI(
        api_key=sambanova_api_key, base_url=sambanova_base_url
    )

    # Prepare input for reflection
    synthesized_info_dict = {
        question: {
            "synthesized_answer": info.synthesized_answer,
            "key_sources": info.key_sources,
            "confidence_level": info.confidence_level,
            "information_gaps": info.information_gaps,
        }
        for question, info in state.synthesized_info.items()
    }

    viewpoint_analysis_dict = None
    if state.viewpoint_analysis:
        # Convert the viewpoint analysis to a dict for the LLM
        tension_list = []
        for tension in state.viewpoint_analysis.areas_of_tension:
            tension_list.append(
                {"topic": tension.topic, "viewpoints": tension.viewpoints}
            )

        viewpoint_analysis_dict = {
            "main_points_of_agreement": state.viewpoint_analysis.main_points_of_agreement,
            "areas_of_tension": tension_list,
            "perspective_gaps": state.viewpoint_analysis.perspective_gaps,
            "integrative_insights": state.viewpoint_analysis.integrative_insights,
        }

    reflection_input = {
        "main_query": state.main_query,
        "sub_questions": state.sub_questions,
        "synthesized_information": synthesized_info_dict,
    }

    if viewpoint_analysis_dict:
        reflection_input["viewpoint_analysis"] = viewpoint_analysis_dict

    # Get reflection critique
    try:
        logger.info(f"Generating self-critique via {llm_model}")
        response = openai_client.chat.completions.create(
            model=llm_model,
            messages=[
                {"role": "system", "content": reflection_prompt},
                {"role": "user", "content": json.dumps(reflection_input)},
            ],
        )

        content = response.choices[0].message.content
        content = remove_reasoning_from_output(content)
        content = clean_json_tags(content)

        reflection_result = safe_json_loads(content)

        if not reflection_result:
            logger.warning("Failed to parse reflection result")
            reflection_result = {
                "critique": [],
                "additional_questions": [],
                "recommended_search_queries": [],
            }

        # Make a deep copy of the synthesized info to create enhanced_info
        enhanced_info = {
            k: SynthesizedInfo(
                synthesized_answer=v.synthesized_answer,
                key_sources=v.key_sources.copy(),
                confidence_level=v.confidence_level,
                information_gaps=v.information_gaps,
                improvements=v.improvements.copy()
                if hasattr(v, "improvements")
                else [],
            )
            for k, v in state.synthesized_info.items()
        }

        # Perform additional searches based on recommendations
        search_queries = reflection_result.get(
            "recommended_search_queries", []
        )
        if max_additional_searches > 0 and search_queries:
            # Limit to max_additional_searches
            search_queries = search_queries[:max_additional_searches]

            for query in search_queries:
                logger.info(f"Performing additional search: {query}")
                # Execute the search
                search_results = tavily_search(
                    query=query,
                    max_results=num_results_per_search,
                    cap_content_length=cap_search_length,
                )

                # Extract raw contents
                raw_contents = [
                    result.get("raw_content", "")
                    for result in search_results.get("results", [])
                    if result.get("raw_content")
                ]

                # Find the most relevant sub-question for this query
                most_relevant_question = _find_most_relevant_subquestion(
                    query=query,
                    sub_questions=state.sub_questions,
                    openai_client=openai_client,
                    model=llm_model,
                )

                if (
                    most_relevant_question
                    and most_relevant_question in enhanced_info
                ):
                    # Enhance the synthesis with new information
                    enhancement_input = {
                        "original_synthesis": enhanced_info[
                            most_relevant_question
                        ].synthesized_answer,
                        "new_information": raw_contents,
                        "critique": [
                            item
                            for item in reflection_result.get("critique", [])
                            if _is_relevant_to_subquestion(
                                item.get("issue", ""), most_relevant_question
                            )
                        ],
                    }

                    enhanced_synthesis = _enhance_synthesis(
                        enhancement_input=enhancement_input,
                        openai_client=openai_client,
                        model=llm_model,
                        system_prompt=additional_synthesis_prompt,
                    )

                    if (
                        enhanced_synthesis
                        and "enhanced_synthesis" in enhanced_synthesis
                    ):
                        # Update the synthesized answer
                        enhanced_info[
                            most_relevant_question
                        ].synthesized_answer = enhanced_synthesis[
                            "enhanced_synthesis"
                        ]

                        # Add improvements
                        improvements = enhanced_synthesis.get(
                            "improvements_made", []
                        )
                        enhanced_info[
                            most_relevant_question
                        ].improvements.extend(improvements)

        # Add any additional questions as new synthesized entries
        for new_question in reflection_result.get("additional_questions", []):
            if (
                new_question not in state.sub_questions
                and new_question not in enhanced_info
            ):
                enhanced_info[new_question] = SynthesizedInfo(
                    synthesized_answer=f"This question was identified during reflection but has not yet been researched: {new_question}",
                    key_sources=[],
                    confidence_level="low",
                    information_gaps="This question requires additional research.",
                )

        # Prepare metadata about the reflection process
        reflection_metadata = ReflectionMetadata(
            critique_summary=[
                item.get("issue", "")
                for item in reflection_result.get("critique", [])
            ],
            additional_questions_identified=reflection_result.get(
                "additional_questions", []
            ),
            searches_performed=search_queries,
            improvements_made=sum(
                [len(info.improvements) for info in enhanced_info.values()]
            ),
        )

        logger.info(
            f"Completed iterative reflection with {reflection_metadata.improvements_made} improvements"
        )

        # Update the state with enhanced info and metadata
        state.update_after_reflection(enhanced_info, reflection_metadata)

        return state

    except Exception as e:
        logger.error(f"Error during iterative reflection: {e}")

        # Create error metadata
        error_metadata = ReflectionMetadata(
            error=f"Reflection failed: {str(e)}"
        )

        # Update the state with the original synthesized info as enhanced info
        # and the error metadata
        state.update_after_reflection(state.synthesized_info, error_metadata)

        return state


def _find_most_relevant_subquestion(
    query: str,
    sub_questions: List[str],
    openai_client: openai.OpenAI,
    model: str,
) -> str:
    """Find the most relevant sub-question for a search query.

    Args:
        query: The search query
        sub_questions: List of sub-questions
        openai_client: OpenAI client
        model: Model to use

    Returns:
        The most relevant sub-question, or None if relevance cannot be determined
    """
    if not sub_questions:
        return None

    # For a single sub-question, it's obviously the most relevant
    if len(sub_questions) == 1:
        return sub_questions[0]

    try:
        # Simple prompt to determine relevance
        prompt = f"""Given the search query: "{query}"
Which of the following research sub-questions is most relevant to this query?
{json.dumps(sub_questions)}

Respond with only the exact text of the most relevant sub-question."""

        response = openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a research assistant."},
                {"role": "user", "content": prompt},
            ],
        )

        answer = response.choices[0].message.content.strip()

        # Check if the answer is one of the sub-questions
        if answer in sub_questions:
            return answer

        # If not an exact match, find the closest one
        for question in sub_questions:
            if question in answer or answer in question:
                return question

        # If still no match, return the first sub-question
        return sub_questions[0]

    except Exception as e:
        logger.error(f"Error finding relevant sub-question: {e}")
        return sub_questions[0]  # Default to the first one in case of error


def _is_relevant_to_subquestion(issue: str, sub_question: str) -> bool:
    """Determine if a critique issue is relevant to a specific sub-question.

    Args:
        issue: The critique issue text
        sub_question: The sub-question text

    Returns:
        True if the issue is relevant to the sub-question, False otherwise
    """
    # Simple relevance check based on text overlap
    # A more sophisticated approach would use semantic similarity
    return sub_question.lower() in issue.lower() or any(
        word
        for word in sub_question.lower().split()
        if len(word) > 4 and word in issue.lower()
    )


def _enhance_synthesis(
    enhancement_input: Dict[str, Any],
    openai_client: openai.OpenAI,
    model: str,
    system_prompt: str,
) -> Dict[str, Any]:
    """Enhance a synthesis with new information and address critique.

    Args:
        enhancement_input: Dictionary with original synthesis, new information, and critique
        openai_client: OpenAI client
        model: Model to use
        system_prompt: System prompt for enhancement

    Returns:
        Dictionary with enhanced synthesis and metadata
    """
    try:
        response = openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(enhancement_input)},
            ],
        )

        content = response.choices[0].message.content
        content = remove_reasoning_from_output(content)
        content = clean_json_tags(content)

        result = safe_json_loads(content)

        if not result or "enhanced_synthesis" not in result:
            # Fallback if parsing fails
            return {
                "enhanced_synthesis": enhancement_input.get(
                    "original_synthesis", ""
                ),
                "improvements_made": ["Failed to enhance synthesis"],
                "remaining_limitations": "Enhancement process failed.",
            }

        return result

    except Exception as e:
        logger.error(f"Error enhancing synthesis: {e}")
        return {
            "enhanced_synthesis": enhancement_input.get(
                "original_synthesis", ""
            ),
            "improvements_made": [f"Error during enhancement: {str(e)}"],
            "remaining_limitations": "Technical error during enhancement process.",
        }
