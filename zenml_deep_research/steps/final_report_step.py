import json
import logging
from typing import Annotated

from utils.data_models import ResearchState
from utils.helper_functions import (
    extract_html_from_content,
    remove_reasoning_from_output,
)
from utils.llm_utils import get_sambanova_client, run_llm_completion
from utils.prompts import REPORT_GENERATION_PROMPT
from zenml import step
from zenml.types import HTMLString

logger = logging.getLogger(__name__)


@step
def final_report_generation_step(
    state: ResearchState,
    sambanova_base_url: str = "https://api.sambanova.ai/v1",
    llm_model: str = "DeepSeek-R1-Distill-Llama-70B",
    system_prompt: str = REPORT_GENERATION_PROMPT,
) -> Annotated[HTMLString, "final_report"]:
    """Generate the final research report in HTML format.

    Args:
        state: The current research state
        sambanova_base_url: SambaNova API base URL
        llm_model: The model to use for report generation
        system_prompt: System prompt for the LLM

    Returns:
        The final research report as an HTML string
    """
    logger.info("Generating final research report")

    # Initialize OpenAI client using the utility function
    openai_client = get_sambanova_client(base_url=sambanova_base_url)

    # Prepare input for report generation
    # Convert state objects to JSON-serializable dictionaries

    # Convert synthesized/enhanced info
    enhanced_info_dict = {}
    for question, info in state.enhanced_info.items():
        enhanced_info_dict[question] = {
            "synthesized_answer": info.synthesized_answer,
            "key_sources": info.key_sources,
            "confidence_level": info.confidence_level,
            "information_gaps": info.information_gaps,
            "improvements": info.improvements,
        }

    # Convert viewpoint analysis if available
    viewpoint_analysis_dict = None
    if state.viewpoint_analysis:
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

    # Convert reflection metadata if available
    reflection_metadata_dict = None
    if state.reflection_metadata:
        reflection_metadata_dict = {
            "critique_summary": state.reflection_metadata.critique_summary,
            "additional_questions_identified": state.reflection_metadata.additional_questions_identified,
            "searches_performed": state.reflection_metadata.searches_performed,
            "improvements_made": state.reflection_metadata.improvements_made,
            "error": state.reflection_metadata.error,
        }

    # Prepare the full report input
    report_input = {
        "main_query": state.main_query,
        "sub_questions": state.sub_questions,
        "synthesized_information": enhanced_info_dict,
    }

    if viewpoint_analysis_dict:
        report_input["viewpoint_analysis"] = viewpoint_analysis_dict

    if reflection_metadata_dict:
        report_input["reflection_metadata"] = reflection_metadata_dict

    # Generate the report
    try:
        logger.info(f"Calling {llm_model} to generate final report")

        # Use the utility function to run LLM completion
        html_content = run_llm_completion(
            prompt=json.dumps(report_input),
            system_prompt=system_prompt,
            client=openai_client,
            model=llm_model,
            clean_output=False,  # Don't clean in case of breaking HTML formatting
        )

        # Clean up any JSON wrapper or other artifacts
        html_content = remove_reasoning_from_output(html_content)

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
        return HTMLString(html_content)

    except Exception as e:
        logger.error(f"Error generating final report: {e}")
        # Generate a minimal fallback report
        fallback_html = _generate_fallback_report(state)

        # Update the state with the fallback report
        state.set_final_report(fallback_html)

        return HTMLString(fallback_html)


def _generate_fallback_report(state: ResearchState) -> str:
    """Generate a minimal fallback report when the main report generation fails.

    Args:
        state: The current research state

    Returns:
        A basic HTML report
    """
    # Create a simple HTML structure
    html = f"""
    <div class="research-report">
      <h1>Research Report: {state.main_query}</h1>
      
      <div class="notice error">
        <p>This is a fallback report generated due to an error in the report generation process.</p>
      </div>
      
      <div class="introduction">
        <h2>Introduction</h2>
        <p>This report addresses the research query: "{state.main_query}"</p>
      </div>
    """

    # Add each sub-question and its synthesized information
    for sub_question in state.sub_questions:
        info = state.enhanced_info.get(sub_question, None)

        if not info:
            # Try to get from synthesized info if not in enhanced info
            info = state.synthesized_info.get(sub_question, None)

        if info:
            answer = info.synthesized_answer
            confidence = info.confidence_level

            html += f"""
            <div class="section">
              <h2>{sub_question}</h2>
              <p class="confidence-level">Confidence: {confidence}</p>
              <div class="content">
                <p>{answer}</p>
              </div>
            """

            # Add improvements if available
            if hasattr(info, "improvements") and info.improvements:
                html += """
              <div class="improvements">
                <h3>Improvements:</h3>
                <ul>
                """

                for improvement in info.improvements:
                    html += f"<li>{improvement}</li>"

                html += """
                </ul>
              </div>
                """

            html += """
            </div>
            """
        else:
            html += f"""
            <div class="section">
              <h2>{sub_question}</h2>
              <p>No information available for this question.</p>
            </div>
            """

    # Add sources if available
    sources_set = set()
    for info in state.enhanced_info.values():
        if info.key_sources:
            sources_set.update(info.key_sources)

    if sources_set:
        html += """
        <div class="references">
          <h2>References</h2>
          <ul>
        """

        for source in sorted(sources_set):
            html += f"<li>{source}</li>"

        html += """
          </ul>
        </div>
        """

    # Close the main div
    html += """
    </div>
    """

    return html
