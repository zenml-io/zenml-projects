import json
import logging
import time
from typing import Annotated, List

from materializers.pydantic_materializer import ResearchStateMaterializer
from utils.helper_functions import (
    safe_json_loads,
)
from utils.llm_utils import run_llm_completion
from utils.prompt_models import PromptsBundle
from utils.pydantic_models import (
    ResearchState,
    ViewpointAnalysis,
    ViewpointTension,
)
from zenml import log_metadata, step

logger = logging.getLogger(__name__)


@step(output_materializers=ResearchStateMaterializer)
def cross_viewpoint_analysis_step(
    state: ResearchState,
    prompts_bundle: PromptsBundle,
    llm_model: str = "sambanova/DeepSeek-R1-Distill-Llama-70B",
    viewpoint_categories: List[str] = [
        "scientific",
        "political",
        "economic",
        "social",
        "ethical",
        "historical",
    ],
    langfuse_project_name: str = "deep-research",
) -> Annotated[ResearchState, "updated_state"]:
    """Analyze synthesized information across different viewpoints.

    Args:
        state: The current research state
        prompts_bundle: Bundle containing all prompts for the pipeline
        llm_model: The model to use for viewpoint analysis
        viewpoint_categories: Categories of viewpoints to analyze

    Returns:
        Updated research state with viewpoint analysis
    """
    start_time = time.time()
    logger.info(
        f"Performing cross-viewpoint analysis on {len(state.synthesized_info)} sub-questions"
    )

    # Prepare input for viewpoint analysis
    analysis_input = {
        "main_query": state.main_query,
        "sub_questions": state.sub_questions,
        "synthesized_information": {
            question: {
                "synthesized_answer": info.synthesized_answer,
                "key_sources": info.key_sources,
                "confidence_level": info.confidence_level,
                "information_gaps": info.information_gaps,
            }
            for question, info in state.synthesized_info.items()
        },
        "viewpoint_categories": viewpoint_categories,
    }

    # Perform viewpoint analysis
    try:
        logger.info(f"Calling {llm_model} for viewpoint analysis")
        # Get the prompt from the bundle
        system_prompt = prompts_bundle.get_prompt_content(
            "viewpoint_analysis_prompt"
        )

        # Use the run_llm_completion function from llm_utils
        content = run_llm_completion(
            prompt=json.dumps(analysis_input),
            system_prompt=system_prompt,
            model=llm_model,  # Model name will be prefixed in the function
            max_tokens=3000,  # Further increased for more comprehensive viewpoint analysis
            project=langfuse_project_name,
        )

        result = safe_json_loads(content)

        if not result:
            logger.warning("Failed to parse viewpoint analysis result")
            # Create a default viewpoint analysis
            viewpoint_analysis = ViewpointAnalysis(
                main_points_of_agreement=[
                    "Analysis failed to identify points of agreement."
                ],
                perspective_gaps="Analysis failed to identify perspective gaps.",
                integrative_insights="Analysis failed to provide integrative insights.",
            )
        else:
            # Create tension objects
            tensions = []
            for tension_data in result.get("areas_of_tension", []):
                tensions.append(
                    ViewpointTension(
                        topic=tension_data.get("topic", ""),
                        viewpoints=tension_data.get("viewpoints", {}),
                    )
                )

            # Create the viewpoint analysis object
            viewpoint_analysis = ViewpointAnalysis(
                main_points_of_agreement=result.get(
                    "main_points_of_agreement", []
                ),
                areas_of_tension=tensions,
                perspective_gaps=result.get("perspective_gaps", ""),
                integrative_insights=result.get("integrative_insights", ""),
            )

        logger.info("Completed viewpoint analysis")

        # Update the state with the viewpoint analysis
        state.update_viewpoint_analysis(viewpoint_analysis)

        # Calculate execution time
        execution_time = time.time() - start_time

        # Count viewpoint tensions by category
        tension_categories = {}
        for tension in viewpoint_analysis.areas_of_tension:
            for category in tension.viewpoints.keys():
                tension_categories[category] = (
                    tension_categories.get(category, 0) + 1
                )

        # Log metadata
        log_metadata(
            metadata={
                "viewpoint_analysis": {
                    "execution_time_seconds": execution_time,
                    "llm_model": llm_model,
                    "num_sub_questions_analyzed": len(state.synthesized_info),
                    "viewpoint_categories_requested": viewpoint_categories,
                    "num_agreement_points": len(
                        viewpoint_analysis.main_points_of_agreement
                    ),
                    "num_tension_areas": len(
                        viewpoint_analysis.areas_of_tension
                    ),
                    "tension_categories_distribution": tension_categories,
                    "has_perspective_gaps": bool(
                        viewpoint_analysis.perspective_gaps
                        and viewpoint_analysis.perspective_gaps != ""
                    ),
                    "has_integrative_insights": bool(
                        viewpoint_analysis.integrative_insights
                        and viewpoint_analysis.integrative_insights != ""
                    ),
                    "analysis_success": not viewpoint_analysis.main_points_of_agreement[
                        0
                    ].startswith("Analysis failed"),
                }
            }
        )

        # Log model metadata for cross-pipeline tracking
        log_metadata(
            metadata={
                "research_scope": {
                    "num_tension_areas": len(
                        viewpoint_analysis.areas_of_tension
                    ),
                }
            },
            infer_model=True,
        )

        # Log artifact metadata
        log_metadata(
            metadata={
                "state_with_viewpoint_analysis": {
                    "has_viewpoint_analysis": True,
                    "total_viewpoints_analyzed": sum(
                        tension_categories.values()
                    ),
                    "most_common_tension_category": max(
                        tension_categories, key=tension_categories.get
                    )
                    if tension_categories
                    else None,
                }
            },
            infer_artifact=True,
        )

        return state

    except Exception as e:
        logger.error(f"Error performing viewpoint analysis: {e}")

        # Create a fallback viewpoint analysis
        fallback_analysis = ViewpointAnalysis(
            main_points_of_agreement=[
                "Analysis failed due to technical error."
            ],
            perspective_gaps=f"Analysis failed: {str(e)}",
            integrative_insights="No insights available due to analysis failure.",
        )

        # Update the state with the fallback analysis
        state.update_viewpoint_analysis(fallback_analysis)

        # Log error metadata
        execution_time = time.time() - start_time
        log_metadata(
            metadata={
                "viewpoint_analysis": {
                    "execution_time_seconds": execution_time,
                    "llm_model": llm_model,
                    "num_sub_questions_analyzed": len(state.synthesized_info),
                    "viewpoint_categories_requested": viewpoint_categories,
                    "analysis_success": False,
                    "error_message": str(e),
                    "fallback_used": True,
                }
            }
        )

        return state
