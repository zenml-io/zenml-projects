import json
import logging
import time
from typing import Annotated, List, Tuple

from materializers.analysis_data_materializer import AnalysisDataMaterializer
from utils.llm_utils import get_structured_llm_output
from utils.pydantic_models import (
    AnalysisData,
    Prompt,
    QueryContext,
    ReflectionMetadata,
    SynthesisData,
)
from zenml import log_metadata, step

logger = logging.getLogger(__name__)


@step(
    output_materializers={
        "analysis_data": AnalysisDataMaterializer,
    }
)
def generate_reflection_step(
    query_context: QueryContext,
    synthesis_data: SynthesisData,
    analysis_data: AnalysisData,
    reflection_prompt: Prompt,
    llm_model: str = "openrouter/google/gemini-2.0-flash-lite-001",
    langfuse_project_name: str = "deep-research",
) -> Tuple[
    Annotated[AnalysisData, "analysis_data"],
    Annotated[List[str], "recommended_queries"],
]:
    """
    Generate reflection and recommendations WITHOUT executing searches.

    This step only analyzes the current state and produces recommendations
    for additional research that could improve the quality of the results.

    Args:
        query_context: The query context with main query and sub-questions
        synthesis_data: The synthesized information
        analysis_data: The analysis data with viewpoint analysis
        reflection_prompt: Prompt for generating reflection
        llm_model: The model to use for reflection
        langfuse_project_name: Project name for tracing

    Returns:
        Tuple of updated AnalysisData and recommended queries
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
        for question, info in synthesis_data.synthesized_info.items()
    }

    viewpoint_analysis_dict = None
    if analysis_data.viewpoint_analysis:
        # Convert the viewpoint analysis to a dict for the LLM
        tension_list = []
        for tension in analysis_data.viewpoint_analysis.areas_of_tension:
            tension_list.append(
                {"topic": tension.topic, "viewpoints": tension.viewpoints}
            )

        viewpoint_analysis_dict = {
            "main_points_of_agreement": analysis_data.viewpoint_analysis.main_points_of_agreement,
            "areas_of_tension": tension_list,
            "perspective_gaps": analysis_data.viewpoint_analysis.perspective_gaps,
            "integrative_insights": analysis_data.viewpoint_analysis.integrative_insights,
        }

    reflection_input = {
        "main_query": query_context.main_query,
        "sub_questions": query_context.sub_questions,
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

    # Extract results
    recommended_queries = reflection_result.get(
        "recommended_search_queries", []
    )
    critique_summary = reflection_result.get("critique", [])
    additional_questions = reflection_result.get("additional_questions", [])

    # Update analysis data with reflection metadata
    analysis_data.reflection_metadata = ReflectionMetadata(
        critique_summary=[
            str(c) for c in critique_summary
        ],  # Convert to strings
        additional_questions_identified=additional_questions,
        searches_performed=[],  # Will be populated by execute_approved_searches_step
        improvements_made=0.0,  # Will be updated later
    )

    # Calculate execution time
    execution_time = time.time() - start_time

    # Count confidence levels in synthesized info
    confidence_levels = [
        info.confidence_level
        for info in synthesis_data.synthesized_info.values()
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
                "num_sub_questions_analyzed": len(query_context.sub_questions),
                "num_synthesized_answers": len(
                    synthesis_data.synthesized_info
                ),
                "viewpoint_analysis_included": bool(viewpoint_analysis_dict),
                "num_critique_points": len(critique_summary),
                "num_additional_questions": len(additional_questions),
                "num_recommended_queries": len(recommended_queries),
                "confidence_distribution": confidence_distribution,
                "has_information_gaps": any(
                    info.information_gaps
                    for info in synthesis_data.synthesized_info.values()
                ),
            }
        }
    )

    # Log artifact metadata
    log_metadata(
        metadata={
            "analysis_data_characteristics": {
                "has_reflection_metadata": True,
                "has_viewpoint_analysis": analysis_data.viewpoint_analysis
                is not None,
                "num_critique_points": len(critique_summary),
                "num_additional_questions": len(additional_questions),
            }
        },
        artifact_name="analysis_data",
        infer_artifact=True,
    )

    log_metadata(
        metadata={
            "recommended_queries_characteristics": {
                "num_queries": len(recommended_queries),
                "has_recommendations": bool(recommended_queries),
            }
        },
        artifact_name="recommended_queries",
        infer_artifact=True,
    )

    # Add tags to the artifacts
    # add_tags(tags=["analysis", "reflection"], artifact_name="analysis_data", infer_artifact=True)
    # add_tags(
    #     tags=["recommendations", "queries"], artifact_name="recommended_queries", infer_artifact=True
    # )

    return analysis_data, recommended_queries
