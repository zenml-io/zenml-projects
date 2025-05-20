import json
import logging
import os
from typing import Annotated, List

import openai
from materializers.research_state_materializer import ResearchStateMaterializer
from utils.data_models import (
    ResearchState,
    ViewpointAnalysis,
    ViewpointTension,
)
from utils.helper_functions import (
    clean_json_tags,
    remove_reasoning_from_output,
    safe_json_loads,
)
from zenml import step

logger = logging.getLogger(__name__)

# System prompt for cross-viewpoint analysis
VIEWPOINT_ANALYSIS_PROMPT = """
You are a Deep Research assistant specializing in analyzing multiple perspectives. You will be given a set of synthesized answers 
to sub-questions related to a main research query.

Your task is to analyze these answers across different viewpoints. Consider how different perspectives might interpret the same 
information differently. Identify where there are:
1. Clear agreements across perspectives
2. Notable disagreements or tensions between viewpoints
3. Blind spots where certain perspectives might be missing
4. Nuances that might be interpreted differently based on viewpoint

For this analysis, consider the following viewpoint categories: scientific, political, economic, social, ethical, and historical.
Not all categories may be relevant to every topic - use those that apply.

Format the output in json with the following json schema definition:

<OUTPUT JSON SCHEMA>
{
  "type": "object",
  "properties": {
    "main_points_of_agreement": {
      "type": "array",
      "items": {"type": "string"}
    },
    "areas_of_tension": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "topic": {"type": "string"},
          "viewpoints": {
            "type": "object",
            "additionalProperties": {"type": "string"}
          }
        }
      }
    },
    "perspective_gaps": {"type": "string"},
    "integrative_insights": {"type": "string"}
  }
}
</OUTPUT JSON SCHEMA>

Make sure that the output is a json object with an output json schema defined above.
Only return the json object, no explanation or additional text.
"""


@step(output_materializers=ResearchStateMaterializer)
def cross_viewpoint_analysis_step(
    state: ResearchState,
    sambanova_base_url: str = "https://api.sambanova.ai/v1",
    llm_model: str = "DeepSeek-R1-Distill-Llama-70B",
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
        sambanova_base_url: SambaNova API base URL
        llm_model: The model to use for viewpoint analysis
        viewpoint_categories: Categories of viewpoints to analyze
        system_prompt: System prompt for the LLM

    Returns:
        Updated research state with viewpoint analysis
    """
    logger.info(
        f"Performing cross-viewpoint analysis on {len(state.synthesized_info)} sub-questions"
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
        response = openai_client.chat.completions.create(
            model=llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(analysis_input)},
            ],
        )

        content = response.choices[0].message.content
        content = remove_reasoning_from_output(content)
        content = clean_json_tags(content)

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
