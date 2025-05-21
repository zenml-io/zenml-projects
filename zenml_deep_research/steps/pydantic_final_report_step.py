"""Final report generation step using Pydantic models and materializers.

This module provides a ZenML pipeline step for generating the final HTML research report
using Pydantic models and improved materializers.
"""

import json
import logging
from typing import Annotated, Tuple

from materializers.pydantic_materializer import ResearchStateMaterializer

# Import helper functions from the original module
from steps.final_report_step import (
    _generate_fallback_report,
    clean_html_output,
    generate_report_from_template,
)
from utils.helper_functions import (
    extract_html_from_content,
    remove_reasoning_from_output,
)
from utils.llm_utils import run_llm_completion
from utils.prompts import (
    REPORT_GENERATION_PROMPT,
)
from utils.pydantic_models import ResearchState
from zenml import step
from zenml.types import HTMLString

logger = logging.getLogger(__name__)


@step(
    output_materializers={
        "state": ResearchStateMaterializer,
    }
)
def pydantic_final_report_step(
    state: ResearchState,
    use_static_template: bool = True,
    llm_model: str = "gpt-3.5-turbo",
    system_prompt: str = REPORT_GENERATION_PROMPT,
) -> Tuple[
    Annotated[ResearchState, "state"], Annotated[HTMLString, "report_html"]
]:
    """Generate the final research report in HTML format using Pydantic models.

    This step uses the Pydantic models and materializers to generate a final
    HTML report and return both the updated state and the HTML report as
    separate artifacts.

    Args:
        state: The current research state (Pydantic model)
        use_static_template: Whether to use a static template instead of LLM generation
        llm_model: The model to use for report generation with provider prefix
        system_prompt: System prompt for the LLM

    Returns:
        A tuple containing the updated research state and the HTML report
    """
    logger.info("Generating final research report using Pydantic models")

    if use_static_template:
        # Use the static HTML template approach
        logger.info("Using static HTML template for report generation")
        html_content = generate_report_from_template(state)

        # Update the state with the final report HTML
        state.set_final_report(html_content)

        logger.info(
            "Final research report generated successfully with static template"
        )
        return state, HTMLString(html_content)

    # Otherwise use the LLM-generated approach
    # Convert Pydantic model to dict for LLM input
    report_input = {
        "main_query": state.main_query,
        "sub_questions": state.sub_questions,
        "synthesized_information": state.enhanced_info,
    }

    if state.viewpoint_analysis:
        report_input["viewpoint_analysis"] = state.viewpoint_analysis

    if state.reflection_metadata:
        report_input["reflection_metadata"] = state.reflection_metadata

    # Generate the report
    try:
        logger.info(f"Calling {llm_model} to generate final report")

        # Use the utility function to run LLM completion
        html_content = run_llm_completion(
            prompt=json.dumps(report_input),
            system_prompt=system_prompt,
            model=llm_model,
            clean_output=False,  # Don't clean in case of breaking HTML formatting
        )

        # Clean up any JSON wrapper or other artifacts
        html_content = remove_reasoning_from_output(html_content)

        # Process the HTML content to remove code block markers and fix common issues
        html_content = clean_html_output(html_content)

        # Basic validation of HTML content
        if not html_content.strip().startswith("<"):
            logger.warning(
                "Generated content does not appear to be valid HTML"
            )
            # Try to extract HTML if it might be wrapped in code blocks or JSON
            html_content = extract_html_from_content(html_content)

        # Update the state with the final report HTML
        state.set_final_report(html_content)

        logger.info("Final research report generated successfully")
        return state, HTMLString(html_content)

    except Exception as e:
        logger.error(f"Error generating final report: {e}")
        # Generate a minimal fallback report
        fallback_html = _generate_fallback_report(state)

        # Process the fallback HTML to ensure it's clean
        fallback_html = clean_html_output(fallback_html)

        # Update the state with the fallback report
        state.set_final_report(fallback_html)

        return state, HTMLString(fallback_html)
