import html
import json
import logging
import re
from typing import Annotated

from utils.helper_functions import (
    extract_html_from_content,
    remove_reasoning_from_output,
)
from utils.llm_utils import run_llm_completion
from utils.prompts import (
    REPORT_GENERATION_PROMPT,
    STATIC_HTML_TEMPLATE,
    SUB_QUESTION_TEMPLATE,
    VIEWPOINT_ANALYSIS_TEMPLATE,
)

# Import Pydantic models instead of dataclasses
from utils.pydantic_models import ResearchState
from zenml import step
from zenml.types import HTMLString

# Import materializer

logger = logging.getLogger(__name__)


def clean_html_output(html_content: str) -> str:
    """Clean HTML output from LLM to ensure proper rendering.

    This function removes markdown code blocks, fixes common issues with LLM HTML output,
    and ensures we have proper HTML structure for rendering.

    Args:
        html_content: Raw HTML content from LLM

    Returns:
        Cleaned HTML content ready for rendering
    """
    # Remove markdown code block markers (```html and ```)
    html_content = re.sub(r"```html\s*", "", html_content)
    html_content = re.sub(r"```\s*$", "", html_content)
    html_content = re.sub(r"```", "", html_content)

    # Remove any CSS code block markers
    html_content = re.sub(r"```css\s*", "", html_content)

    # Ensure HTML content is properly wrapped in HTML tags if not already
    if not html_content.strip().startswith(
        "<!DOCTYPE"
    ) and not html_content.strip().startswith("<html"):
        if not html_content.strip().startswith('<div class="research-report"'):
            html_content = f'<div class="research-report">{html_content}</div>'

    html_content = re.sub(r"\[CSS STYLESHEET GOES HERE\]", "", html_content)
    html_content = re.sub(r"\[SUB-QUESTIONS LINKS\]", "", html_content)
    html_content = re.sub(r"\[ADDITIONAL SECTIONS LINKS\]", "", html_content)
    html_content = re.sub(r"\[FOR EACH SUB-QUESTION\]:", "", html_content)
    html_content = re.sub(r"\[FOR EACH TENSION\]:", "", html_content)

    # Replace content placeholders with appropriate defaults
    html_content = re.sub(
        r"\[CONCISE SUMMARY OF KEY FINDINGS\]",
        "Summary of findings from the research query.",
        html_content,
    )
    html_content = re.sub(
        r"\[INTRODUCTION TO THE RESEARCH QUERY\]",
        "Introduction to the research topic.",
        html_content,
    )
    html_content = re.sub(
        r"\[OVERVIEW OF THE APPROACH AND SUB-QUESTIONS\]",
        "Overview of the research approach.",
        html_content,
    )
    html_content = re.sub(
        r"\[CONCLUSION TEXT\]",
        "Conclusion of the research findings.",
        html_content,
    )

    return html_content


def format_text_with_code_blocks(text: str) -> str:
    """Format text with proper handling of code blocks and markdown formatting.

    Args:
        text: The raw text to format

    Returns:
        str: HTML-formatted text
    """
    if not text:
        return ""

    # First escape HTML
    escaped_text = html.escape(text)

    # Handle code blocks (wrap content in ``` or ```)
    pattern = r"```(?:\w*\n)?(.*?)```"

    def code_block_replace(match):
        code_content = match.group(1)
        # Strip extra newlines at beginning and end
        code_content = code_content.strip("\n")
        return f"<pre><code>{code_content}</code></pre>"

    # Replace code blocks
    formatted_text = re.sub(
        pattern, code_block_replace, escaped_text, flags=re.DOTALL
    )

    # Convert regular newlines to <br> tags (but not inside <pre> blocks)
    parts = []
    in_pre = False
    for line in formatted_text.split("\n"):
        if "<pre>" in line:
            in_pre = True
            parts.append(line)
        elif "</pre>" in line:
            in_pre = False
            parts.append(line)
        elif in_pre:
            # Inside a code block, preserve newlines
            parts.append(line)
        else:
            # Outside code blocks, convert newlines to <br>
            parts.append(line + "<br>")

    return "".join(parts)


def generate_report_from_template(state: ResearchState) -> str:
    """Generate a final HTML report from a static template.

    Instead of using an LLM to generate HTML, this function uses predefined HTML
    templates and populates them with data from the research state.

    Args:
        state: The current research state

    Returns:
        str: The HTML content of the report
    """
    logger.info(
        f"Generating templated HTML report for query: {state.main_query}"
    )

    # Generate table of contents for sub-questions
    sub_questions_toc = ""
    for i, question in enumerate(state.sub_questions, 1):
        safe_id = f"question-{i}"
        sub_questions_toc += (
            f'<li><a href="#{safe_id}">{html.escape(question)}</a></li>\n'
        )

    # Add viewpoint analysis to TOC if available
    additional_sections_toc = ""
    if state.viewpoint_analysis:
        additional_sections_toc += (
            '<li><a href="#viewpoint-analysis">Viewpoint Analysis</a></li>\n'
        )

    # Generate HTML for sub-questions
    sub_questions_html = ""
    all_sources = set()

    for i, question in enumerate(state.sub_questions, 1):
        info = state.enhanced_info.get(question, None)

        # Skip if no information is available
        if not info:
            continue

        # Process confidence level
        confidence = info.confidence_level.lower()
        confidence_upper = info.confidence_level.upper()

        # Process key sources
        key_sources_html = ""
        if info.key_sources:
            all_sources.update(info.key_sources)
            sources_list = "\n".join(
                [
                    f'<li><a href="{html.escape(source)}" target="_blank">{html.escape(source)}</a></li>'
                    if source.startswith(("http://", "https://"))
                    else f"<li>{html.escape(source)}</li>"
                    for source in info.key_sources
                ]
            )
            key_sources_html = f"""
            <div class="key-sources">
                <h3><span class="section-icon">📚</span> Key Sources</h3>
                <ul class="source-list">
                    {sources_list}
                </ul>
            </div>
            """

        # Process information gaps
        info_gaps_html = ""
        if info.information_gaps:
            info_gaps_html = f"""
            <div class="information-gaps">
                <h3><span class="section-icon">🧩</span> Information Gaps</h3>
                <p>{format_text_with_code_blocks(info.information_gaps)}</p>
            </div>
            """

        # Determine confidence icon based on level
        confidence_icon = "🔴"  # Default (low)
        if confidence_upper == "HIGH":
            confidence_icon = "🟢"
        elif confidence_upper == "MEDIUM":
            confidence_icon = "🟡"

        # Format the subquestion section using the template
        sub_question_html = SUB_QUESTION_TEMPLATE.format(
            index=i,
            question=html.escape(question),
            confidence=confidence,
            confidence_upper=confidence_upper,
            confidence_icon=confidence_icon,
            answer=format_text_with_code_blocks(info.synthesized_answer),
            info_gaps_html=info_gaps_html,
            key_sources_html=key_sources_html,
        )

        sub_questions_html += sub_question_html

    # Generate viewpoint analysis HTML if available
    viewpoint_analysis_html = ""
    if state.viewpoint_analysis:
        # Format points of agreement
        agreements_html = ""
        for point in state.viewpoint_analysis.main_points_of_agreement:
            agreements_html += f"<li>{html.escape(point)}</li>\n"

        # Format areas of tension
        tensions_html = ""
        for tension in state.viewpoint_analysis.areas_of_tension:
            viewpoints_html = ""
            for title, content in tension.viewpoints.items():
                # Create category-specific styling
                category_class = f"category-{title.lower()}"
                category_title = title.capitalize()

                viewpoints_html += f"""
                <div class="viewpoint-item">
                    <span class="viewpoint-category {category_class}">{category_title}</span>
                    <p>{html.escape(content)}</p>
                </div>
                """

            tensions_html += f"""
            <div class="viewpoint-tension">
                <h4>{html.escape(tension.topic)}</h4>
                <div class="viewpoint-content">
                    {viewpoints_html}
                </div>
            </div>
            """

        # Format the viewpoint analysis section using the template
        viewpoint_analysis_html = VIEWPOINT_ANALYSIS_TEMPLATE.format(
            agreements_html=agreements_html,
            tensions_html=tensions_html,
            perspective_gaps=format_text_with_code_blocks(
                state.viewpoint_analysis.perspective_gaps
            ),
            integrative_insights=format_text_with_code_blocks(
                state.viewpoint_analysis.integrative_insights
            ),
        )

    # Generate references HTML
    references_html = "<ul>"
    if all_sources:
        for source in sorted(all_sources):
            if source.startswith(("http://", "https://")):
                references_html += f'<li><a href="{html.escape(source)}" target="_blank">{html.escape(source)}</a></li>\n'
            else:
                references_html += f"<li>{html.escape(source)}</li>\n"
    else:
        references_html += (
            "<li>No external sources were referenced in this research.</li>"
        )
    references_html += "</ul>"

    # Generate executive summary based on reflection or create a default one
    executive_summary = ""
    if hasattr(state, "reflection") and state.reflection:
        executive_summary = format_text_with_code_blocks(state.reflection)
    else:
        executive_summary = f"This report examines {html.escape(state.main_query)} through a structured approach, breaking down the topic into {len(state.sub_questions)} focused sub-questions. The research synthesizes information from multiple sources to provide a comprehensive analysis of the topic."

    # Generate complete HTML report
    html_content = STATIC_HTML_TEMPLATE.format(
        main_query=html.escape(state.main_query),
        sub_questions_toc=sub_questions_toc,
        additional_sections_toc=additional_sections_toc,
        executive_summary=executive_summary,
        num_sub_questions=len(state.sub_questions),
        sub_questions_html=sub_questions_html,
        viewpoint_analysis_html=viewpoint_analysis_html,
        references_html=references_html,
    )

    return html_content


@step
def final_report_generation_step(
    state: ResearchState,
    use_static_template: bool = True,
    llm_model: str = "sambanova/DeepSeek-R1-Distill-Llama-70B",
    system_prompt: str = REPORT_GENERATION_PROMPT,
) -> Annotated[HTMLString, "final_report"]:
    """Generate the final research report in HTML format.

    Args:
        state: The current research state
        use_static_template: Whether to use a static template instead of LLM generation
        llm_model: The model to use for report generation with provider prefix
        system_prompt: System prompt for the LLM

    Returns:
        The final research report as an HTML string
    """
    logger.info("Generating final research report")

    if use_static_template:
        # Use the static HTML template approach
        logger.info("Using static HTML template for report generation")
        html_content = generate_report_from_template(state)

        # Update the state with the final report HTML
        state.set_final_report(html_content)

        logger.info(
            "Final research report generated successfully with static template"
        )
        return HTMLString(html_content)

    # Otherwise use the LLM-generated approach

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
        return HTMLString(html_content)

    except Exception as e:
        logger.error(f"Error generating final report: {e}")
        # Generate a minimal fallback report
        fallback_html = _generate_fallback_report(state)

        # Process the fallback HTML to ensure it's clean
        fallback_html = clean_html_output(fallback_html)

        # Update the state with the fallback report
        state.set_final_report(fallback_html)

        return HTMLString(fallback_html)


def _generate_fallback_report(state: ResearchState) -> str:
    """Generate a minimal fallback report when the main report generation fails.

    This function creates a simplified HTML report with a consistent structure when
    the main report generation process encounters an error. The HTML includes:
    - A header section with the main research query
    - An error notice
    - Introduction section
    - Individual sections for each sub-question with available answers
    - A references section if sources are available

    Args:
        state: The current research state containing query and answer information

    Returns:
        str: A basic HTML report with a standard research report structure
    """
    # Create a simple HTML structure with embedded CSS for styling
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        /* Global Styles */
        body {{
            font-family: Arial, Helvetica, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f9f9f9;
        }}
        
        .research-report {{
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
            padding: 30px;
        }}
        
        /* Typography */
        h1 {{
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
            margin-top: 0;
        }}
        
        h2 {{
            color: #2c3e50;
            border-bottom: 1px solid #eee;
            padding-bottom: 5px;
            margin-top: 30px;
        }}
        
        h3 {{
            color: #3498db;
            margin-top: 20px;
        }}
        
        p {{
            margin: 15px 0;
        }}
        
        /* Sections */
        .section {{
            margin: 30px 0;
            padding: 20px;
            background-color: #f8f9fa;
            border-left: 4px solid #3498db;
            border-radius: 4px;
        }}
        
        .content {{
            margin-top: 15px;
        }}
        
        /* Notice/Error Styles */
        .notice {{
            padding: 15px;
            margin: 20px 0;
            border-radius: 4px;
        }}
        
        .error {{
            background-color: #fee;
            border-left: 4px solid #e74c3c;
            color: #c0392b;
        }}
        
        /* Confidence Level Indicators */
        .confidence-level {{
            display: inline-block;
            padding: 5px 10px;
            border-radius: 4px;
            font-weight: bold;
            margin: 10px 0;
        }}
        
        .confidence-high {{
            background-color: #d4edda;
            color: #155724;
            border-left: 4px solid #28a745;
        }}
        
        .confidence-medium {{
            background-color: #fff3cd;
            color: #856404;
            border-left: 4px solid #ffc107;
        }}
        
        .confidence-low {{
            background-color: #f8d7da;
            color: #721c24;
            border-left: 4px solid #dc3545;
        }}
        
        /* Lists */
        ul {{
            padding-left: 20px;
        }}
        
        li {{
            margin: 8px 0;
        }}
        
        /* References Section */
        .references {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #eee;
        }}
        
        .references ul {{
            list-style-type: none;
            padding-left: 0;
        }}
        
        .references li {{
            padding: 8px 0;
            border-bottom: 1px dotted #ddd;
        }}
        
        /* Table of Contents */
        .toc {{
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 4px;
            margin: 20px 0;
        }}
        
        .toc ul {{
            list-style-type: none;
            padding-left: 10px;
        }}
        
        .toc li {{
            margin: 5px 0;
        }}
        
        .toc a {{
            color: #3498db;
            text-decoration: none;
        }}
        
        .toc a:hover {{
            text-decoration: underline;
        }}

        /* Executive Summary */
        .executive-summary {{
            background-color: #e8f4f8;
            padding: 20px;
            border-radius: 4px;
            margin: 20px 0;
            border-left: 4px solid #3498db;
        }}
    </style>
</head>
<body>
    <div class="research-report">
        <h1>Research Report: {state.main_query}</h1>
        
        <div class="notice error">
            <p><strong>Note:</strong> This is a fallback report generated due to an error in the report generation process.</p>
        </div>
        
        <!-- Table of Contents -->
        <div class="toc">
            <h2>Table of Contents</h2>
            <ul>
                <li><a href="#introduction">Introduction</a></li>
"""

    # Add TOC entries for each sub-question
    for i, sub_question in enumerate(state.sub_questions):
        safe_id = f"question-{i + 1}"
        html += f'                <li><a href="#{safe_id}">{sub_question}</a></li>\n'

    html += """                <li><a href="#references">References</a></li>
            </ul>
        </div>
        
        <!-- Executive Summary -->
        <div class="executive-summary">
            <h2>Executive Summary</h2>
            <p>This report presents findings related to the main research query. It explores multiple aspects of the topic through structured sub-questions and synthesizes information from various sources.</p>
        </div>
        
        <div class="introduction" id="introduction">
            <h2>Introduction</h2>
            <p>This report addresses the research query: "<strong>{state.main_query}</strong>"</p>
            <p>The analysis is structured around {len(state.sub_questions)} sub-questions that explore different dimensions of this topic.</p>
        </div>
"""

    # Add each sub-question and its synthesized information
    for i, sub_question in enumerate(state.sub_questions):
        safe_id = f"question-{i + 1}"
        info = state.enhanced_info.get(sub_question, None)

        if not info:
            # Try to get from synthesized info if not in enhanced info
            info = state.synthesized_info.get(sub_question, None)

        if info:
            answer = info.synthesized_answer
            confidence = info.confidence_level

            # Add appropriate confidence class
            confidence_class = ""
            if confidence == "high":
                confidence_class = "confidence-high"
            elif confidence == "medium":
                confidence_class = "confidence-medium"
            elif confidence == "low":
                confidence_class = "confidence-low"

            html += f"""
        <div class="section" id="{safe_id}">
            <h2>{i + 1}. {sub_question}</h2>
            <p class="confidence-level {confidence_class}">Confidence Level: {confidence.upper()}</p>
            <div class="content">
                <p>{answer}</p>
            </div>
            """

            # Add information gaps if available
            if hasattr(info, "information_gaps") and info.information_gaps:
                html += f"""
            <div class="information-gaps">
                <h3>Information Gaps</h3>
                <p>{info.information_gaps}</p>
            </div>
                """

            # Add improvements if available
            if hasattr(info, "improvements") and info.improvements:
                html += """
            <div class="improvements">
                <h3>Improvements Made</h3>
                <ul>
                """

                for improvement in info.improvements:
                    html += f"                <li>{improvement}</li>\n"

                html += """
                </ul>
            </div>
                """

            # Add key sources if available
            if hasattr(info, "key_sources") and info.key_sources:
                html += """
            <div class="key-sources">
                <h3>Key Sources</h3>
                <ul>
                """

                for source in info.key_sources:
                    html += f"                <li>{source}</li>\n"

                html += """
                </ul>
            </div>
                """

            html += """
        </div>
            """
        else:
            html += f"""
        <div class="section" id="{safe_id}">
            <h2>{i + 1}. {sub_question}</h2>
            <p>No information available for this question.</p>
        </div>
            """

    # Add conclusion section
    html += """
        <div class="section">
            <h2>Conclusion</h2>
            <p>This report has explored the research query through multiple sub-questions, providing synthesized information based on available sources. While limitations exist in some areas, the report provides a structured analysis of the topic.</p>
        </div>
    """

    # Add sources if available
    sources_set = set()
    for info in state.enhanced_info.values():
        if info.key_sources:
            sources_set.update(info.key_sources)

    if sources_set:
        html += """
        <div class="references" id="references">
            <h2>References</h2>
            <ul>
        """

        for source in sorted(sources_set):
            html += f"            <li>{source}</li>\n"

        html += """
            </ul>
        </div>
        """
    else:
        html += """
        <div class="references" id="references">
            <h2>References</h2>
            <p>No references available.</p>
        </div>
        """

    # Close the HTML structure
    html += """
    </div>
</body>
</html>
    """

    return html
