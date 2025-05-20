import json
import logging
from typing import Annotated

from materializers.research_state_materializer import ResearchStateMaterializer
from utils.data_models import (
    ReflectionMetadata,
    ResearchState,
    SynthesizedInfo,
)
from utils.llm_utils import (
    find_most_relevant_string,
    get_sambanova_client,
    get_structured_llm_output,
    is_text_relevant,
)
from utils.search_utils import search_and_extract_results
from zenml import step

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

    # Initialize OpenAI client using the utility function
    openai_client = get_sambanova_client(base_url=sambanova_base_url)

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

        # Define fallback for reflection result
        fallback_reflection = {
            "critique": [],
            "additional_questions": [],
            "recommended_search_queries": [],
        }

        # Use utility function to get structured output
        reflection_result = get_structured_llm_output(
            prompt=json.dumps(reflection_input),
            system_prompt=reflection_prompt,
            client=openai_client,
            model=llm_model,
            fallback_response=fallback_reflection,
        )

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
                # Execute the search using the utility function
                search_results = search_and_extract_results(
                    query=query,
                    max_results=num_results_per_search,
                    cap_content_length=cap_search_length,
                )

                # Extract raw contents
                raw_contents = [result.content for result in search_results]

                # Find the most relevant sub-question for this query
                most_relevant_question = find_most_relevant_string(
                    query, state.sub_questions, openai_client, llm_model
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
                            if is_text_relevant(
                                item.get("issue", ""), most_relevant_question
                            )
                        ],
                    }

                    # Use the utility function for enhancement
                    enhanced_synthesis = get_structured_llm_output(
                        prompt=json.dumps(enhancement_input),
                        system_prompt=additional_synthesis_prompt,
                        client=openai_client,
                        model=llm_model,
                        fallback_response={
                            "enhanced_synthesis": enhanced_info[
                                most_relevant_question
                            ].synthesized_answer,
                            "improvements_made": [
                                "Failed to enhance synthesis"
                            ],
                            "remaining_limitations": "Enhancement process failed.",
                        },
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
