import json
import logging
from typing import Annotated

from utils.llm_utils import get_structured_llm_output
from utils.prompts import REFLECTION_PROMPT
from utils.pydantic_models import ReflectionOutput, ResearchState
from zenml import step

logger = logging.getLogger(__name__)


@step
def generate_reflection_step(
    state: ResearchState,
    llm_model: str = "sambanova/DeepSeek-R1-Distill-Llama-70B",
    reflection_prompt: str = REFLECTION_PROMPT,
) -> Annotated[ReflectionOutput, "reflection_output"]:
    """
    Generate reflection and recommendations WITHOUT executing searches.

    This step only analyzes the current state and produces recommendations
    for additional research that could improve the quality of the results.

    Args:
        state: The current research state
        llm_model: The model to use for reflection
        reflection_prompt: System prompt for the reflection

    Returns:
        ReflectionOutput containing the state, recommendations, and critique
    """
    logger.info("Generating reflection on research")

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
        model=llm_model,
        fallback_response=fallback_reflection,
    )

    # Return structured output for next steps
    return ReflectionOutput(
        state=state,
        recommended_queries=reflection_result.get(
            "recommended_search_queries", []
        ),
        critique_summary=reflection_result.get("critique", []),
        additional_questions=reflection_result.get("additional_questions", []),
    )
