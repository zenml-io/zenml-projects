import json
import logging
from typing import Annotated, List

from materializers.pydantic_materializer import ResearchStateMaterializer
from utils.helper_functions import (
    safe_json_loads,
)
from utils.llm_utils import run_llm_completion
from utils.prompts import VIEWPOINT_ANALYSIS_PROMPT
from utils.pydantic_models import (
    ResearchState,
    ViewpointAnalysis,
    ViewpointTension,
)
from zenml import step

logger = logging.getLogger(__name__)


@step(output_materializers=ResearchStateMaterializer)
def cross_viewpoint_analysis_step(
    state: ResearchState,
    llm_model: str = "sambanova/DeepSeek-R1-Distill-Llama-70B",
    viewpoint_categories: List[str] = [
        "scientific",
        "political",
        "economic",
        "social",
        "ethical",
        "historical",
    ],
    system_prompt: str = VIEWPOINT_ANALYSIS_PROMPT,
) -> Annotated[ResearchState, "updated_state"]:
    """Analyze synthesized information across different viewpoints.

    Args:
        state: The current research state
        llm_model: The model to use for viewpoint analysis
        viewpoint_categories: Categories of viewpoints to analyze
        system_prompt: System prompt for the LLM

    Returns:
        Updated research state with viewpoint analysis
    """
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
        # Use the run_llm_completion function from llm_utils
        content = run_llm_completion(
            prompt=json.dumps(analysis_input),
            system_prompt=system_prompt,
            model=llm_model,  # Model name will be prefixed in the function
            max_tokens=3000,  # Further increased for more comprehensive viewpoint analysis
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

        return state
