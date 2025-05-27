import json
import logging
import time
from typing import Annotated

from materializers.reflection_output_materializer import (
    ReflectionOutputMaterializer,
)
from utils.llm_utils import get_structured_llm_output
from utils.pydantic_models import Prompt, ReflectionOutput, ResearchState
from zenml import add_tags, log_metadata, step

logger = logging.getLogger(__name__)


@step(output_materializers=ReflectionOutputMaterializer)
def generate_reflection_step(
    state: ResearchState,
    reflection_prompt: Prompt,
    llm_model: str = "sambanova/DeepSeek-R1-Distill-Llama-70B",
    langfuse_project_name: str = "deep-research",
) -> Annotated[ReflectionOutput, "reflection_output"]:
    """
    Generate reflection and recommendations WITHOUT executing searches.

    This step only analyzes the current state and produces recommendations
    for additional research that could improve the quality of the results.

    Args:
        state: The current research state
        reflection_prompt: Prompt for generating reflection
        llm_model: The model to use for reflection

    Returns:
        ReflectionOutput containing the state, recommendations, and critique
    """
    start_time = time.time()
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
        system_prompt=str(reflection_prompt),
        model=llm_model,
        fallback_response=fallback_reflection,
        project=langfuse_project_name,
    )

    # Prepare return value
    reflection_output = ReflectionOutput(
        state=state,
        recommended_queries=reflection_result.get(
            "recommended_search_queries", []
        ),
        critique_summary=reflection_result.get("critique", []),
        additional_questions=reflection_result.get("additional_questions", []),
    )

    # Calculate execution time
    execution_time = time.time() - start_time

    # Count confidence levels in synthesized info
    confidence_levels = [
        info.confidence_level for info in state.synthesized_info.values()
    ]
    confidence_distribution = {
        "high": confidence_levels.count("high"),
        "medium": confidence_levels.count("medium"),
        "low": confidence_levels.count("low"),
    }

    # Log step metadata
    log_metadata(
        metadata={
            "reflection_generation": {
                "execution_time_seconds": execution_time,
                "llm_model": llm_model,
                "num_sub_questions_analyzed": len(state.sub_questions),
                "num_synthesized_answers": len(state.synthesized_info),
                "viewpoint_analysis_included": bool(viewpoint_analysis_dict),
                "num_critique_points": len(reflection_output.critique_summary),
                "num_additional_questions": len(
                    reflection_output.additional_questions
                ),
                "num_recommended_queries": len(
                    reflection_output.recommended_queries
                ),
                "confidence_distribution": confidence_distribution,
                "has_information_gaps": any(
                    info.information_gaps
                    for info in state.synthesized_info.values()
                ),
            }
        }
    )

    # Log artifact metadata
    log_metadata(
        metadata={
            "reflection_output_characteristics": {
                "has_recommendations": bool(
                    reflection_output.recommended_queries
                ),
                "has_critique": bool(reflection_output.critique_summary),
                "has_additional_questions": bool(
                    reflection_output.additional_questions
                ),
                "total_recommendations": len(
                    reflection_output.recommended_queries
                )
                + len(reflection_output.additional_questions),
            }
        },
        infer_artifact=True,
    )

    # Add tags to the artifact
    add_tags(tags=["reflection", "critique"], artifact="reflection_output")

    return reflection_output
