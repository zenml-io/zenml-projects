"""Final report generation step using Pydantic models and materializers.

This module provides a ZenML pipeline step for generating the final HTML research report
using Pydantic models and improved materializers.
"""

import html
import json
import logging
import re
import time
from typing import Annotated, Tuple

from materializers.pydantic_materializer import ResearchStateMaterializer
from utils.helper_functions import (
    extract_html_from_content,
    remove_reasoning_from_output,
)
from utils.llm_utils import run_llm_completion
from utils.prompts import (
    STATIC_HTML_TEMPLATE,
    SUB_QUESTION_TEMPLATE,
    VIEWPOINT_ANALYSIS_TEMPLATE,
)
from utils.pydantic_models import Prompt, ResearchState
from zenml import log_metadata, step
from zenml.types import HTMLString

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


def generate_executive_summary(
    state: ResearchState,
    executive_summary_prompt: Prompt,
    llm_model: str = "sambanova/DeepSeek-R1-Distill-Llama-70B",
    langfuse_project_name: str = "deep-research",
) -> str:
    """Generate an executive summary using LLM based on research findings.

    Args:
        state: The current research state
        executive_summary_prompt: Prompt for generating executive summary
        llm_model: The model to use for generation
        langfuse_project_name: Name of the Langfuse project for tracking

    Returns:
        HTML formatted executive summary
    """
    logger.info("Generating executive summary using LLM")

    # Prepare the context with all research findings
    context = f"Main Research Query: {state.main_query}\n\n"

    # Add synthesized findings for each sub-question
    for i, sub_question in enumerate(state.sub_questions, 1):
        info = state.enhanced_info.get(
            sub_question
        ) or state.synthesized_info.get(sub_question)
        if info:
            context += f"Sub-question {i}: {sub_question}\n"
            context += f"Answer Summary: {info.synthesized_answer[:500]}...\n"
            context += f"Confidence: {info.confidence_level}\n"
            context += f"Key Sources: {', '.join(info.key_sources[:3]) if info.key_sources else 'N/A'}\n\n"

    # Add viewpoint analysis insights if available
    if state.viewpoint_analysis:
        context += "Key Areas of Agreement:\n"
        for agreement in state.viewpoint_analysis.main_points_of_agreement[:3]:
            context += f"- {agreement}\n"
        context += "\nKey Tensions:\n"
        for tension in state.viewpoint_analysis.areas_of_tension[:2]:
            context += f"- {tension.topic}\n"

    # Use the executive summary prompt
    try:
        executive_summary_prompt_str = str(executive_summary_prompt)
        logger.info("Successfully retrieved executive_summary_prompt")
    except Exception as e:
        logger.error(f"Failed to get executive_summary_prompt: {e}")
        return generate_fallback_executive_summary(state)

    try:
        # Call LLM to generate executive summary
        result = run_llm_completion(
            prompt=context,
            system_prompt=executive_summary_prompt_str,
            model=llm_model,
            temperature=0.7,
            max_tokens=800,
            project=langfuse_project_name,
            tags=["executive_summary_generation"],
        )

        if result:
            content = remove_reasoning_from_output(result)
            # Clean up the HTML
            content = extract_html_from_content(content)
            logger.info("Successfully generated LLM-based executive summary")
            return content
        else:
            logger.warning("Failed to generate executive summary via LLM")
            return generate_fallback_executive_summary(state)

    except Exception as e:
        logger.error(f"Error generating executive summary: {e}")
        return generate_fallback_executive_summary(state)


def generate_introduction(
    state: ResearchState,
    introduction_prompt: Prompt,
    llm_model: str = "sambanova/DeepSeek-R1-Distill-Llama-70B",
    langfuse_project_name: str = "deep-research",
) -> str:
    """Generate an introduction using LLM based on research query and sub-questions.

    Args:
        state: The current research state
        introduction_prompt: Prompt for generating introduction
        llm_model: The model to use for generation
        langfuse_project_name: Name of the Langfuse project for tracking

    Returns:
        HTML formatted introduction
    """
    logger.info("Generating introduction using LLM")

    # Prepare the context
    context = f"Main Research Query: {state.main_query}\n\n"
    context += "Sub-questions being explored:\n"
    for i, sub_question in enumerate(state.sub_questions, 1):
        context += f"{i}. {sub_question}\n"

    # Get the introduction prompt
    try:
        introduction_prompt_str = str(introduction_prompt)
        logger.info("Successfully retrieved introduction_prompt")
    except Exception as e:
        logger.error(f"Failed to get introduction_prompt: {e}")
        return generate_fallback_introduction(state)

    try:
        # Call LLM to generate introduction
        result = run_llm_completion(
            prompt=context,
            system_prompt=introduction_prompt_str,
            model=llm_model,
            temperature=0.7,
            max_tokens=600,
            project=langfuse_project_name,
            tags=["introduction_generation"],
        )

        if result:
            content = remove_reasoning_from_output(result)
            # Clean up the HTML
            content = extract_html_from_content(content)
            logger.info("Successfully generated LLM-based introduction")
            return content
        else:
            logger.warning("Failed to generate introduction via LLM")
            return generate_fallback_introduction(state)

    except Exception as e:
        logger.error(f"Error generating introduction: {e}")
        return generate_fallback_introduction(state)


def generate_fallback_executive_summary(state: ResearchState) -> str:
    """Generate a fallback executive summary when LLM fails."""
    summary = f"<p>This report examines the question: <strong>{html.escape(state.main_query)}</strong></p>"
    summary += f"<p>The research explored {len(state.sub_questions)} key dimensions of this topic, "
    summary += "synthesizing findings from multiple sources to provide a comprehensive analysis.</p>"

    # Add confidence overview
    confidence_counts = {"high": 0, "medium": 0, "low": 0}
    for info in state.enhanced_info.values():
        level = info.confidence_level.lower()
        if level in confidence_counts:
            confidence_counts[level] += 1

    summary += f"<p>Overall confidence in findings: {confidence_counts['high']} high, "
    summary += f"{confidence_counts['medium']} medium, {confidence_counts['low']} low.</p>"

    return summary


def generate_fallback_introduction(state: ResearchState) -> str:
    """Generate a fallback introduction when LLM fails."""
    intro = f"<p>This report addresses the research query: <strong>{html.escape(state.main_query)}</strong></p>"
    intro += f"<p>The research was conducted by breaking down the main query into {len(state.sub_questions)} "
    intro += (
        "sub-questions to explore different aspects of the topic in depth. "
    )
    intro += "Each sub-question was researched independently, with findings synthesized from various sources.</p>"
    return intro


def generate_conclusion(
    state: ResearchState,
    conclusion_generation_prompt: Prompt,
    llm_model: str = "sambanova/DeepSeek-R1-Distill-Llama-70B",
    langfuse_project_name: str = "deep-research",
) -> str:
    """Generate a comprehensive conclusion using LLM based on all research findings.

    Args:
        state: The ResearchState containing all research findings
        conclusion_generation_prompt: Prompt for generating conclusion
        llm_model: The model to use for conclusion generation

    Returns:
        str: HTML-formatted conclusion content
    """
    logger.info("Generating comprehensive conclusion using LLM")

    # Prepare input data for conclusion generation
    conclusion_input = {
        "main_query": state.main_query,
        "sub_questions": state.sub_questions,
        "enhanced_info": {},
    }

    # Include enhanced information for each sub-question
    for question in state.sub_questions:
        if question in state.enhanced_info:
            info = state.enhanced_info[question]
            conclusion_input["enhanced_info"][question] = {
                "synthesized_answer": info.synthesized_answer,
                "confidence_level": info.confidence_level,
                "information_gaps": info.information_gaps,
                "key_sources": info.key_sources,
                "improvements": getattr(info, "improvements", []),
            }
        elif question in state.synthesized_info:
            # Fallback to synthesized info if enhanced info not available
            info = state.synthesized_info[question]
            conclusion_input["enhanced_info"][question] = {
                "synthesized_answer": info.synthesized_answer,
                "confidence_level": info.confidence_level,
                "information_gaps": info.information_gaps,
                "key_sources": info.key_sources,
                "improvements": [],
            }

    # Include viewpoint analysis if available
    if state.viewpoint_analysis:
        conclusion_input["viewpoint_analysis"] = {
            "main_points_of_agreement": state.viewpoint_analysis.main_points_of_agreement,
            "areas_of_tension": [
                {"topic": tension.topic, "viewpoints": tension.viewpoints}
                for tension in state.viewpoint_analysis.areas_of_tension
            ],
            "perspective_gaps": state.viewpoint_analysis.perspective_gaps,
            "integrative_insights": state.viewpoint_analysis.integrative_insights,
        }

    # Include reflection metadata if available
    if state.reflection_metadata:
        conclusion_input["reflection_metadata"] = {
            "critique_summary": state.reflection_metadata.critique_summary,
            "additional_questions_identified": state.reflection_metadata.additional_questions_identified,
            "improvements_made": state.reflection_metadata.improvements_made,
        }

    try:
        # Use the conclusion generation prompt
        conclusion_prompt_str = str(conclusion_generation_prompt)

        # Generate conclusion using LLM
        conclusion_html = run_llm_completion(
            prompt=json.dumps(conclusion_input, indent=2),
            system_prompt=conclusion_prompt_str,
            model=llm_model,
            clean_output=True,
            max_tokens=1500,  # Sufficient for comprehensive conclusion
            project=langfuse_project_name,
        )

        # Clean up any formatting issues
        conclusion_html = conclusion_html.strip()

        # Remove any h2 tags with "Conclusion" text that LLM might have added
        # Since we already have a Conclusion header in the template
        conclusion_html = re.sub(
            r"<h2[^>]*>\s*Conclusion\s*</h2>\s*",
            "",
            conclusion_html,
            flags=re.IGNORECASE,
        )
        conclusion_html = re.sub(
            r"<h3[^>]*>\s*Conclusion\s*</h3>\s*",
            "",
            conclusion_html,
            flags=re.IGNORECASE,
        )

        # Also remove plain text "Conclusion" at the start if it exists
        conclusion_html = re.sub(
            r"^Conclusion\s*\n*",
            "",
            conclusion_html.strip(),
            flags=re.IGNORECASE,
        )

        if not conclusion_html.startswith("<p>"):
            # Wrap in paragraph tags if not already formatted
            conclusion_html = f"<p>{conclusion_html}</p>"

        logger.info("Successfully generated LLM-based conclusion")
        return conclusion_html

    except Exception as e:
        logger.warning(f"Failed to generate LLM conclusion: {e}")
        # Return a basic fallback conclusion
        return f"""<p>This report has explored {html.escape(state.main_query)} through a structured research approach, examining {len(state.sub_questions)} focused sub-questions and synthesizing information from diverse sources. The findings provide a comprehensive understanding of the topic, highlighting key aspects, perspectives, and current knowledge.</p>
        <p>While some information gaps remain, as noted in the respective sections, this research provides a solid foundation for understanding the topic and its implications.</p>"""


def generate_report_from_template(
    state: ResearchState,
    conclusion_generation_prompt: Prompt,
    executive_summary_prompt: Prompt,
    introduction_prompt: Prompt,
    llm_model: str = "sambanova/DeepSeek-R1-Distill-Llama-70B",
    langfuse_project_name: str = "deep-research",
) -> str:
    """Generate a final HTML report from a static template.

    Instead of using an LLM to generate HTML, this function uses predefined HTML
    templates and populates them with data from the research state.

    Args:
        state: The current research state
        conclusion_generation_prompt: Prompt for generating conclusion
        executive_summary_prompt: Prompt for generating executive summary
        introduction_prompt: Prompt for generating introduction
        llm_model: The model to use for conclusion generation

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

    # Generate dynamic executive summary using LLM
    logger.info("Generating dynamic executive summary...")
    executive_summary = generate_executive_summary(
        state, executive_summary_prompt, llm_model, langfuse_project_name
    )
    logger.info(
        f"Executive summary generated: {len(executive_summary)} characters"
    )

    # Generate dynamic introduction using LLM
    logger.info("Generating dynamic introduction...")
    introduction_html = generate_introduction(
        state, introduction_prompt, llm_model, langfuse_project_name
    )
    logger.info(f"Introduction generated: {len(introduction_html)} characters")

    # Generate comprehensive conclusion using LLM
    conclusion_html = generate_conclusion(
        state, conclusion_generation_prompt, llm_model, langfuse_project_name
    )

    # Generate complete HTML report
    html_content = STATIC_HTML_TEMPLATE.format(
        main_query=html.escape(state.main_query),
        sub_questions_toc=sub_questions_toc,
        additional_sections_toc=additional_sections_toc,
        executive_summary=executive_summary,
        introduction_html=introduction_html,
        num_sub_questions=len(state.sub_questions),
        sub_questions_html=sub_questions_html,
        viewpoint_analysis_html=viewpoint_analysis_html,
        conclusion_html=conclusion_html,
        references_html=references_html,
    )

    return html_content


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


@step(
    output_materializers={
        "state": ResearchStateMaterializer,
    }
)
def pydantic_final_report_step(
    state: ResearchState,
    conclusion_generation_prompt: Prompt,
    executive_summary_prompt: Prompt,
    introduction_prompt: Prompt,
    use_static_template: bool = True,
    llm_model: str = "sambanova/DeepSeek-R1-Distill-Llama-70B",
    langfuse_project_name: str = "deep-research",
) -> Tuple[
    Annotated[ResearchState, "state"],
    Annotated[HTMLString, "report_html"],
]:
    """Generate the final research report in HTML format using Pydantic models.

    This step uses the Pydantic models and materializers to generate a final
    HTML report and return both the updated state and the HTML report as
    separate artifacts.

    Args:
        state: The current research state (Pydantic model)
        conclusion_generation_prompt: Prompt for generating conclusions
        executive_summary_prompt: Prompt for generating executive summary
        introduction_prompt: Prompt for generating introduction
        use_static_template: Whether to use a static template instead of LLM generation
        llm_model: The model to use for report generation with provider prefix

    Returns:
        A tuple containing the updated research state and the HTML report
    """
    start_time = time.time()
    logger.info("Generating final research report using Pydantic models")

    if use_static_template:
        # Use the static HTML template approach
        logger.info("Using static HTML template for report generation")
        html_content = generate_report_from_template(
            state,
            conclusion_generation_prompt,
            executive_summary_prompt,
            introduction_prompt,
            llm_model,
            langfuse_project_name,
        )

        # Update the state with the final report HTML
        state.set_final_report(html_content)

        # Collect metadata about the report
        execution_time = time.time() - start_time

        # Count sources
        all_sources = set()
        for info in state.enhanced_info.values():
            if info.key_sources:
                all_sources.update(info.key_sources)

        # Count confidence levels
        confidence_distribution = {"high": 0, "medium": 0, "low": 0}
        for info in state.enhanced_info.values():
            level = info.confidence_level.lower()
            if level in confidence_distribution:
                confidence_distribution[level] += 1

        # Log metadata
        log_metadata(
            metadata={
                "report_generation": {
                    "execution_time_seconds": execution_time,
                    "generation_method": "static_template",
                    "llm_model": llm_model,
                    "report_length_chars": len(html_content),
                    "num_sub_questions": len(state.sub_questions),
                    "num_sources": len(all_sources),
                    "has_viewpoint_analysis": bool(state.viewpoint_analysis),
                    "has_reflection": bool(state.reflection_metadata),
                    "confidence_distribution": confidence_distribution,
                    "fallback_report": False,
                }
            }
        )

        # Log model metadata for cross-pipeline tracking
        log_metadata(
            metadata={
                "research_quality": {
                    "confidence_distribution": confidence_distribution,
                }
            },
            infer_model=True,
        )

        # Log artifact metadata for the HTML report
        log_metadata(
            metadata={
                "html_report_characteristics": {
                    "size_bytes": len(html_content.encode("utf-8")),
                    "has_toc": "toc" in html_content.lower(),
                    "has_executive_summary": "executive summary"
                    in html_content.lower(),
                    "has_conclusion": "conclusion" in html_content.lower(),
                    "has_references": "references" in html_content.lower(),
                }
            },
            infer_artifact=True,
            artifact_name="report_html",
        )

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

        # Use a default report generation prompt
        report_prompt = "Generate a comprehensive HTML research report based on the provided research data. Include proper HTML structure with sections for executive summary, introduction, findings, and conclusion."

        # Use the utility function to run LLM completion
        html_content = run_llm_completion(
            prompt=json.dumps(report_input),
            system_prompt=report_prompt,
            model=llm_model,
            clean_output=False,  # Don't clean in case of breaking HTML formatting
            max_tokens=4000,  # Increased token limit for detailed report generation
            project=langfuse_project_name,
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

        # Collect metadata about the report
        execution_time = time.time() - start_time

        # Count sources
        all_sources = set()
        for info in state.enhanced_info.values():
            if info.key_sources:
                all_sources.update(info.key_sources)

        # Count confidence levels
        confidence_distribution = {"high": 0, "medium": 0, "low": 0}
        for info in state.enhanced_info.values():
            level = info.confidence_level.lower()
            if level in confidence_distribution:
                confidence_distribution[level] += 1

        # Log metadata
        log_metadata(
            metadata={
                "report_generation": {
                    "execution_time_seconds": execution_time,
                    "generation_method": "llm_generated",
                    "llm_model": llm_model,
                    "report_length_chars": len(html_content),
                    "num_sub_questions": len(state.sub_questions),
                    "num_sources": len(all_sources),
                    "has_viewpoint_analysis": bool(state.viewpoint_analysis),
                    "has_reflection": bool(state.reflection_metadata),
                    "confidence_distribution": confidence_distribution,
                    "fallback_report": False,
                }
            }
        )

        # Log model metadata for cross-pipeline tracking
        log_metadata(
            metadata={
                "research_quality": {
                    "confidence_distribution": confidence_distribution,
                }
            },
            infer_model=True,
        )

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

        # Collect metadata about the fallback report
        execution_time = time.time() - start_time

        # Count sources
        all_sources = set()
        for info in state.enhanced_info.values():
            if info.key_sources:
                all_sources.update(info.key_sources)

        # Count confidence levels
        confidence_distribution = {"high": 0, "medium": 0, "low": 0}
        for info in state.enhanced_info.values():
            level = info.confidence_level.lower()
            if level in confidence_distribution:
                confidence_distribution[level] += 1

        # Log metadata for fallback report
        log_metadata(
            metadata={
                "report_generation": {
                    "execution_time_seconds": execution_time,
                    "generation_method": "fallback",
                    "llm_model": llm_model,
                    "report_length_chars": len(fallback_html),
                    "num_sub_questions": len(state.sub_questions),
                    "num_sources": len(all_sources),
                    "has_viewpoint_analysis": bool(state.viewpoint_analysis),
                    "has_reflection": bool(state.reflection_metadata),
                    "confidence_distribution": confidence_distribution,
                    "fallback_report": True,
                    "error_message": str(e),
                }
            }
        )

        # Log model metadata for cross-pipeline tracking
        log_metadata(
            metadata={
                "research_quality": {
                    "confidence_distribution": confidence_distribution,
                }
            },
            infer_model=True,
        )

        return state, HTMLString(fallback_html)
