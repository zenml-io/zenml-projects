"""Final report generation step using artifact-based approach.

This module provides a ZenML pipeline step for generating the final HTML research report
using the new artifact-based approach.
"""

import html
import json
import logging
import re
import time
from typing import Annotated, Tuple

from materializers.final_report_materializer import FinalReportMaterializer
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
from utils.pydantic_models import (
    AnalysisData,
    FinalReport,
    Prompt,
    QueryContext,
    SearchData,
    SynthesisData,
)
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

    # Handle code blocks
    lines = text.split("\n")
    formatted_lines = []
    in_code_block = False
    code_language = ""
    code_lines = []

    for line in lines:
        # Check for code block start
        if line.strip().startswith("```"):
            if in_code_block:
                # End of code block
                code_content = "\n".join(code_lines)
                formatted_lines.append(
                    f'<pre><code class="language-{code_language}">{html.escape(code_content)}</code></pre>'
                )
                code_lines = []
                in_code_block = False
                code_language = ""
            else:
                # Start of code block
                in_code_block = True
                # Extract language if specified
                code_language = line.strip()[3:].strip() or "plaintext"
        elif in_code_block:
            code_lines.append(line)
        else:
            # Process inline code
            line = re.sub(r"`([^`]+)`", r"<code>\1</code>", html.escape(line))
            # Process bullet points
            if line.strip().startswith("•") or line.strip().startswith("-"):
                line = re.sub(r"^(\s*)[•-]\s*", r"\1", line)
                formatted_lines.append(f"<li>{line.strip()}</li>")
            elif line.strip():
                formatted_lines.append(f"<p>{line}</p>")

    # Handle case where code block wasn't closed
    if in_code_block and code_lines:
        code_content = "\n".join(code_lines)
        formatted_lines.append(
            f'<pre><code class="language-{code_language}">{html.escape(code_content)}</code></pre>'
        )

    # Wrap list items in ul tags
    result = []
    in_list = False
    for line in formatted_lines:
        if line.startswith("<li>"):
            if not in_list:
                result.append("<ul>")
                in_list = True
            result.append(line)
        else:
            if in_list:
                result.append("</ul>")
                in_list = False
            result.append(line)

    if in_list:
        result.append("</ul>")

    return "\n".join(result)


def generate_executive_summary(
    query_context: QueryContext,
    synthesis_data: SynthesisData,
    analysis_data: AnalysisData,
    executive_summary_prompt: Prompt,
    llm_model: str = "sambanova/DeepSeek-R1-Distill-Llama-70B",
    langfuse_project_name: str = "deep-research",
) -> str:
    """Generate an executive summary using LLM based on the complete research findings.

    Args:
        query_context: The query context with main query and sub-questions
        synthesis_data: The synthesis data with all synthesized information
        analysis_data: The analysis data with viewpoint analysis
        executive_summary_prompt: Prompt for generating executive summary
        llm_model: The model to use for generation
        langfuse_project_name: Name of the Langfuse project for tracking

    Returns:
        HTML formatted executive summary
    """
    logger.info("Generating executive summary using LLM")

    # Prepare the context
    summary_input = {
        "main_query": query_context.main_query,
        "sub_questions": query_context.sub_questions,
        "key_findings": {},
        "viewpoint_analysis": None,
    }

    # Include key findings from synthesis data
    # Prefer enhanced info if available
    info_source = (
        synthesis_data.enhanced_info
        if synthesis_data.enhanced_info
        else synthesis_data.synthesized_info
    )

    for question in query_context.sub_questions:
        if question in info_source:
            info = info_source[question]
            summary_input["key_findings"][question] = {
                "answer": info.synthesized_answer,
                "confidence": info.confidence_level,
                "gaps": info.information_gaps,
            }

    # Include viewpoint analysis if available
    if analysis_data.viewpoint_analysis:
        va = analysis_data.viewpoint_analysis
        summary_input["viewpoint_analysis"] = {
            "agreements": va.main_points_of_agreement,
            "tensions": len(va.areas_of_tension),
            "insights": va.integrative_insights,
        }

    try:
        # Call LLM to generate executive summary
        result = run_llm_completion(
            prompt=json.dumps(summary_input),
            system_prompt=str(executive_summary_prompt),
            model=llm_model,
            temperature=0.7,
            max_tokens=600,
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
            return generate_fallback_executive_summary(
                query_context, synthesis_data
            )

    except Exception as e:
        logger.error(f"Error generating executive summary: {e}")
        return generate_fallback_executive_summary(
            query_context, synthesis_data
        )


def generate_introduction(
    query_context: QueryContext,
    introduction_prompt: Prompt,
    llm_model: str = "sambanova/DeepSeek-R1-Distill-Llama-70B",
    langfuse_project_name: str = "deep-research",
) -> str:
    """Generate an introduction using LLM based on research query and sub-questions.

    Args:
        query_context: The query context with main query and sub-questions
        introduction_prompt: Prompt for generating introduction
        llm_model: The model to use for generation
        langfuse_project_name: Name of the Langfuse project for tracking

    Returns:
        HTML formatted introduction
    """
    logger.info("Generating introduction using LLM")

    # Prepare the context
    context = f"Main Research Query: {query_context.main_query}\n\n"
    context += "Sub-questions being explored:\n"
    for i, sub_question in enumerate(query_context.sub_questions, 1):
        context += f"{i}. {sub_question}\n"

    try:
        # Call LLM to generate introduction
        result = run_llm_completion(
            prompt=context,
            system_prompt=str(introduction_prompt),
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
            return generate_fallback_introduction(query_context)

    except Exception as e:
        logger.error(f"Error generating introduction: {e}")
        return generate_fallback_introduction(query_context)


def generate_fallback_executive_summary(
    query_context: QueryContext, synthesis_data: SynthesisData
) -> str:
    """Generate a fallback executive summary when LLM fails."""
    summary = f"<p>This report examines the question: <strong>{html.escape(query_context.main_query)}</strong></p>"
    summary += f"<p>The research explored {len(query_context.sub_questions)} key dimensions of this topic, "
    summary += "synthesizing findings from multiple sources to provide a comprehensive analysis.</p>"

    # Add confidence overview
    confidence_counts = {"high": 0, "medium": 0, "low": 0}
    info_source = (
        synthesis_data.enhanced_info
        if synthesis_data.enhanced_info
        else synthesis_data.synthesized_info
    )
    for info in info_source.values():
        level = info.confidence_level.lower()
        if level in confidence_counts:
            confidence_counts[level] += 1

    summary += f"<p>Overall confidence in findings: {confidence_counts['high']} high, "
    summary += f"{confidence_counts['medium']} medium, {confidence_counts['low']} low.</p>"

    return summary


def generate_fallback_introduction(query_context: QueryContext) -> str:
    """Generate a fallback introduction when LLM fails."""
    intro = f"<p>This report addresses the research query: <strong>{html.escape(query_context.main_query)}</strong></p>"
    intro += f"<p>The research was conducted by breaking down the main query into {len(query_context.sub_questions)} "
    intro += (
        "sub-questions to explore different aspects of the topic in depth. "
    )
    intro += "Each sub-question was researched independently, with findings synthesized from various sources.</p>"
    return intro


def generate_conclusion(
    query_context: QueryContext,
    synthesis_data: SynthesisData,
    analysis_data: AnalysisData,
    conclusion_generation_prompt: Prompt,
    llm_model: str = "sambanova/DeepSeek-R1-Distill-Llama-70B",
    langfuse_project_name: str = "deep-research",
) -> str:
    """Generate a comprehensive conclusion using LLM based on all research findings.

    Args:
        query_context: The query context with main query and sub-questions
        synthesis_data: The synthesis data with all synthesized information
        analysis_data: The analysis data with viewpoint analysis
        conclusion_generation_prompt: Prompt for generating conclusion
        llm_model: The model to use for conclusion generation
        langfuse_project_name: Name of the Langfuse project for tracking

    Returns:
        str: HTML-formatted conclusion content
    """
    logger.info("Generating comprehensive conclusion using LLM")

    # Prepare input data for conclusion generation
    conclusion_input = {
        "main_query": query_context.main_query,
        "sub_questions": query_context.sub_questions,
        "enhanced_info": {},
    }

    # Include enhanced information for each sub-question
    info_source = (
        synthesis_data.enhanced_info
        if synthesis_data.enhanced_info
        else synthesis_data.synthesized_info
    )

    for question in query_context.sub_questions:
        if question in info_source:
            info = info_source[question]
            conclusion_input["enhanced_info"][question] = {
                "synthesized_answer": info.synthesized_answer,
                "confidence_level": info.confidence_level,
                "information_gaps": info.information_gaps,
                "key_sources": info.key_sources,
                "improvements": getattr(info, "improvements", []),
            }

    # Include viewpoint analysis
    if analysis_data.viewpoint_analysis:
        va = analysis_data.viewpoint_analysis
        conclusion_input["viewpoint_analysis"] = {
            "main_points_of_agreement": va.main_points_of_agreement,
            "areas_of_tension": [
                {"topic": t.topic, "viewpoints": t.viewpoints}
                for t in va.areas_of_tension
            ],
            "integrative_insights": va.integrative_insights,
        }

    # Include reflection metadata if available
    if analysis_data.reflection_metadata:
        rm = analysis_data.reflection_metadata
        conclusion_input["reflection_insights"] = {
            "improvements_made": rm.improvements_made,
            "additional_questions_identified": rm.additional_questions_identified,
        }

    try:
        # Call LLM to generate conclusion
        result = run_llm_completion(
            prompt=json.dumps(conclusion_input),
            system_prompt=str(conclusion_generation_prompt),
            model=llm_model,
            temperature=0.7,
            max_tokens=800,
            project=langfuse_project_name,
            tags=["conclusion_generation"],
        )

        if result:
            content = remove_reasoning_from_output(result)
            # Clean up the HTML
            content = extract_html_from_content(content)
            logger.info("Successfully generated LLM-based conclusion")
            return content
        else:
            logger.warning("Failed to generate conclusion via LLM")
            return generate_fallback_conclusion(query_context, synthesis_data)

    except Exception as e:
        logger.error(f"Error generating conclusion: {e}")
        return generate_fallback_conclusion(query_context, synthesis_data)


def generate_fallback_conclusion(
    query_context: QueryContext, synthesis_data: SynthesisData
) -> str:
    """Generate a fallback conclusion when LLM fails.

    Args:
        query_context: The query context with main query and sub-questions
        synthesis_data: The synthesis data with all synthesized information

    Returns:
        str: Basic HTML-formatted conclusion
    """
    conclusion = f"<p>This research has explored the question: <strong>{html.escape(query_context.main_query)}</strong></p>"
    conclusion += f"<p>Through systematic investigation of {len(query_context.sub_questions)} sub-questions, "
    conclusion += (
        "we have gathered insights from multiple sources and perspectives.</p>"
    )

    # Add a summary of confidence levels
    info_source = (
        synthesis_data.enhanced_info
        if synthesis_data.enhanced_info
        else synthesis_data.synthesized_info
    )
    high_confidence = sum(
        1
        for info in info_source.values()
        if info.confidence_level.lower() == "high"
    )

    if high_confidence > 0:
        conclusion += f"<p>The research yielded {high_confidence} high-confidence findings out of "
        conclusion += f"{len(info_source)} total areas investigated.</p>"

    conclusion += "<p>Further research may be beneficial to address remaining information gaps "
    conclusion += "and explore emerging questions identified during this investigation.</p>"

    return conclusion


def generate_report_from_template(
    query_context: QueryContext,
    search_data: SearchData,
    synthesis_data: SynthesisData,
    analysis_data: AnalysisData,
    conclusion_generation_prompt: Prompt,
    executive_summary_prompt: Prompt,
    introduction_prompt: Prompt,
    llm_model: str = "sambanova/DeepSeek-R1-Distill-Llama-70B",
    langfuse_project_name: str = "deep-research",
) -> str:
    """Generate a final HTML report from a static template.

    Instead of using an LLM to generate HTML, this function uses predefined HTML
    templates and populates them with data from the research artifacts.

    Args:
        query_context: The query context with main query and sub-questions
        search_data: The search data (for source information)
        synthesis_data: The synthesis data with all synthesized information
        analysis_data: The analysis data with viewpoint analysis
        conclusion_generation_prompt: Prompt for generating conclusion
        executive_summary_prompt: Prompt for generating executive summary
        introduction_prompt: Prompt for generating introduction
        llm_model: The model to use for conclusion generation
        langfuse_project_name: Name of the Langfuse project for tracking

    Returns:
        str: The HTML content of the report
    """
    logger.info(
        f"Generating templated HTML report for query: {query_context.main_query}"
    )

    # Generate table of contents for sub-questions
    sub_questions_toc = ""
    for i, question in enumerate(query_context.sub_questions, 1):
        safe_id = f"question-{i}"
        sub_questions_toc += (
            f'<li><a href="#{safe_id}">{html.escape(question)}</a></li>\n'
        )

    # Add viewpoint analysis to TOC if available
    additional_sections_toc = ""
    if analysis_data.viewpoint_analysis:
        additional_sections_toc += (
            '<li><a href="#viewpoint-analysis">Viewpoint Analysis</a></li>\n'
        )

    # Generate HTML for sub-questions
    sub_questions_html = ""
    all_sources = set()

    # Determine which info source to use (merge original with enhanced)
    # Start with the original synthesized info
    info_source = synthesis_data.synthesized_info.copy()

    # Override with enhanced info where available
    if synthesis_data.enhanced_info:
        info_source.update(synthesis_data.enhanced_info)

    # Debug logging
    logger.info(
        f"Synthesis data has enhanced_info: {bool(synthesis_data.enhanced_info)}"
    )
    logger.info(
        f"Synthesis data has synthesized_info: {bool(synthesis_data.synthesized_info)}"
    )
    logger.info(f"Info source has {len(info_source)} entries")
    logger.info(f"Processing {len(query_context.sub_questions)} sub-questions")

    # Log the keys in info_source for debugging
    if info_source:
        logger.info(
            f"Keys in info_source: {list(info_source.keys())[:3]}..."
        )  # First 3 keys
    logger.info(
        f"Sub-questions from query_context: {query_context.sub_questions[:3]}..."
    )  # First 3

    for i, question in enumerate(query_context.sub_questions, 1):
        info = info_source.get(question, None)

        # Skip if no information is available
        if not info:
            logger.warning(
                f"No synthesis info found for question {i}: {question}"
            )
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
            key_sources_html=key_sources_html,
            info_gaps_html=info_gaps_html,
        )

        sub_questions_html += sub_question_html

    # Generate viewpoint analysis HTML if available
    viewpoint_analysis_html = ""
    if analysis_data.viewpoint_analysis:
        va = analysis_data.viewpoint_analysis
        # Format tensions
        tensions_html = ""
        for tension in va.areas_of_tension:
            viewpoints_list = "\n".join(
                [
                    f"<li><strong>{html.escape(viewpoint)}:</strong> {html.escape(description)}</li>"
                    for viewpoint, description in tension.viewpoints.items()
                ]
            )
            tensions_html += f"""
            <div class="tension-area">
                <h4>{html.escape(tension.topic)}</h4>
                <ul class="viewpoints-list">
                    {viewpoints_list}
                </ul>
            </div>
            """

        # Format agreements (just the list items)
        agreements_html = ""
        if va.main_points_of_agreement:
            agreements_html = "\n".join(
                [
                    f"<li>{html.escape(point)}</li>"
                    for point in va.main_points_of_agreement
                ]
            )

        # Get perspective gaps if available
        perspective_gaps = ""
        if hasattr(va, "perspective_gaps") and va.perspective_gaps:
            perspective_gaps = va.perspective_gaps
        else:
            perspective_gaps = "No significant perspective gaps identified."

        # Get integrative insights
        integrative_insights = ""
        if va.integrative_insights:
            integrative_insights = format_text_with_code_blocks(
                va.integrative_insights
            )

        viewpoint_analysis_html = VIEWPOINT_ANALYSIS_TEMPLATE.format(
            agreements_html=agreements_html,
            tensions_html=tensions_html,
            perspective_gaps=perspective_gaps,
            integrative_insights=integrative_insights,
        )

    # Generate references section
    references_html = '<ul class="reference-list">'
    if all_sources:
        for source in sorted(all_sources):
            if source.startswith(("http://", "https://")):
                references_html += f'<li><a href="{html.escape(source)}" target="_blank">{html.escape(source)}</a></li>'
            else:
                references_html += f"<li>{html.escape(source)}</li>"
    else:
        references_html += (
            "<li>No external sources were referenced in this research.</li>"
        )
    references_html += "</ul>"

    # Generate dynamic executive summary using LLM
    logger.info("Generating dynamic executive summary...")
    executive_summary = generate_executive_summary(
        query_context,
        synthesis_data,
        analysis_data,
        executive_summary_prompt,
        llm_model,
        langfuse_project_name,
    )
    logger.info(
        f"Executive summary generated: {len(executive_summary)} characters"
    )

    # Generate dynamic introduction using LLM
    logger.info("Generating dynamic introduction...")
    introduction_html = generate_introduction(
        query_context, introduction_prompt, llm_model, langfuse_project_name
    )
    logger.info(f"Introduction generated: {len(introduction_html)} characters")

    # Generate comprehensive conclusion using LLM
    conclusion_html = generate_conclusion(
        query_context,
        synthesis_data,
        analysis_data,
        conclusion_generation_prompt,
        llm_model,
        langfuse_project_name,
    )

    # Generate complete HTML report
    html_content = STATIC_HTML_TEMPLATE.format(
        main_query=html.escape(query_context.main_query),
        sub_questions_toc=sub_questions_toc,
        additional_sections_toc=additional_sections_toc,
        executive_summary=executive_summary,
        introduction_html=introduction_html,
        num_sub_questions=len(query_context.sub_questions),
        sub_questions_html=sub_questions_html,
        viewpoint_analysis_html=viewpoint_analysis_html,
        conclusion_html=conclusion_html,
        references_html=references_html,
    )

    return html_content


def _generate_fallback_report(
    query_context: QueryContext,
    synthesis_data: SynthesisData,
    analysis_data: AnalysisData,
) -> str:
    """Generate a minimal fallback report when the main report generation fails.

    This function creates a simplified HTML report with a consistent structure when
    the main report generation process encounters an error.

    Args:
        query_context: The query context with main query and sub-questions
        synthesis_data: The synthesis data with all synthesized information
        analysis_data: The analysis data with viewpoint analysis

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
            color: #34495e;
            margin-top: 20px;
        }}
        
        p {{
            margin: 10px 0;
        }}
        
        /* Sections */
        .section {{
            margin-bottom: 30px;
            padding: 20px;
            background-color: #f8f9fa;
            border-radius: 8px;
        }}
        
        .error-notice {{
            background-color: #fee;
            border: 1px solid #fcc;
            color: #c33;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
        }}
        
        /* Confidence badges */
        .confidence {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: bold;
            margin-left: 10px;
        }}
        
        .confidence.high {{
            background-color: #d4edda;
            color: #155724;
        }}
        
        .confidence.medium {{
            background-color: #fff3cd;
            color: #856404;
        }}
        
        .confidence.low {{
            background-color: #f8d7da;
            color: #721c24;
        }}
        
        /* Lists */
        ul {{
            margin: 10px 0;
            padding-left: 25px;
        }}
        
        li {{
            margin: 5px 0;
        }}
        
        /* References */
        .references {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 2px solid #eee;
        }}
        
        .reference-list {{
            font-size: 14px;
        }}
        
        .reference-list a {{
            color: #3498db;
            text-decoration: none;
            word-break: break-word;
        }}
        
        .reference-list a:hover {{
            text-decoration: underline;
        }}
    </style>
    <title>Research Report - {html.escape(query_context.main_query)}</title>
</head>
<body>
    <div class="research-report">
        <h1>Research Report: {html.escape(query_context.main_query)}</h1>
        
        <div class="error-notice">
            <strong>Note:</strong> This is a simplified version of the report generated due to processing limitations.
        </div>
        
        <div class="section">
            <h2>Introduction</h2>
            <p>This report investigates the research query: <strong>{html.escape(query_context.main_query)}</strong></p>
            <p>The investigation was structured around {len(query_context.sub_questions)} key sub-questions to provide comprehensive coverage of the topic.</p>
        </div>
        
        <div class="section">
            <h2>Research Findings</h2>
"""

    # Add findings for each sub-question
    info_source = (
        synthesis_data.enhanced_info
        if synthesis_data.enhanced_info
        else synthesis_data.synthesized_info
    )

    for i, question in enumerate(query_context.sub_questions, 1):
        if question in info_source:
            info = info_source[question]
            confidence_class = info.confidence_level.lower()

            html += f"""
            <div class="sub-question">
                <h3>{i}. {html.escape(question)}</h3>
                <span class="confidence {confidence_class}">Confidence: {info.confidence_level.upper()}</span>
                <p>{html.escape(info.synthesized_answer)}</p>
            """

            if info.information_gaps:
                html += f"<p><strong>Information Gaps:</strong> {html.escape(info.information_gaps)}</p>"

            html += "</div>"

    html += """
        </div>
        
        <div class="section">
            <h2>Conclusion</h2>
            <p>This research has provided insights into the various aspects of the main query through systematic investigation.</p>
            <p>The findings represent a synthesis of available information, with varying levels of confidence across different areas.</p>
        </div>
        
        <div class="references">
            <h2>References</h2>
            <p>Sources were gathered from various search providers and synthesized to create this report.</p>
        </div>
    </div>
</body>
</html>"""

    return html


@step(
    output_materializers={
        "final_report": FinalReportMaterializer,
    }
)
def pydantic_final_report_step(
    query_context: QueryContext,
    search_data: SearchData,
    synthesis_data: SynthesisData,
    analysis_data: AnalysisData,
    conclusion_generation_prompt: Prompt,
    executive_summary_prompt: Prompt,
    introduction_prompt: Prompt,
    use_static_template: bool = True,
    llm_model: str = "sambanova/DeepSeek-R1-Distill-Llama-70B",
    langfuse_project_name: str = "deep-research",
) -> Tuple[
    Annotated[FinalReport, "final_report"],
    Annotated[HTMLString, "report_html"],
]:
    """Generate the final research report in HTML format using artifact-based approach.

    This step uses the individual artifacts to generate a final HTML report.

    Args:
        query_context: The query context with main query and sub-questions
        search_data: The search data (for source information)
        synthesis_data: The synthesis data with all synthesized information
        analysis_data: The analysis data with viewpoint analysis and reflection metadata
        conclusion_generation_prompt: Prompt for generating conclusions
        executive_summary_prompt: Prompt for generating executive summary
        introduction_prompt: Prompt for generating introduction
        use_static_template: Whether to use a static template instead of LLM generation
        llm_model: The model to use for report generation with provider prefix
        langfuse_project_name: Name of the Langfuse project for tracking

    Returns:
        A tuple containing the FinalReport artifact and the HTML report string
    """
    start_time = time.time()
    logger.info(
        "Generating final research report using artifact-based approach"
    )

    if use_static_template:
        # Use the static HTML template approach
        logger.info("Using static HTML template for report generation")
        html_content = generate_report_from_template(
            query_context,
            search_data,
            synthesis_data,
            analysis_data,
            conclusion_generation_prompt,
            executive_summary_prompt,
            introduction_prompt,
            llm_model,
            langfuse_project_name,
        )

        # Create the FinalReport artifact
        final_report = FinalReport(
            report_html=html_content,
            main_query=query_context.main_query,
        )

        # Calculate execution time
        execution_time = time.time() - start_time

        # Calculate report metrics
        info_source = (
            synthesis_data.enhanced_info
            if synthesis_data.enhanced_info
            else synthesis_data.synthesized_info
        )
        confidence_distribution = {"high": 0, "medium": 0, "low": 0}
        for info in info_source.values():
            level = info.confidence_level.lower()
            if level in confidence_distribution:
                confidence_distribution[level] += 1

        # Count various elements in the report
        num_sources = len(
            set(
                source
                for info in info_source.values()
                for source in info.key_sources
            )
        )
        has_viewpoint_analysis = analysis_data.viewpoint_analysis is not None
        has_reflection_insights = (
            analysis_data.reflection_metadata is not None
            and analysis_data.reflection_metadata.improvements_made > 0
        )

        # Log step metadata
        log_metadata(
            metadata={
                "final_report_generation": {
                    "execution_time_seconds": execution_time,
                    "use_static_template": use_static_template,
                    "llm_model": llm_model,
                    "main_query_length": len(query_context.main_query),
                    "num_sub_questions": len(query_context.sub_questions),
                    "num_synthesized_answers": len(info_source),
                    "has_enhanced_info": bool(synthesis_data.enhanced_info),
                    "confidence_distribution": confidence_distribution,
                    "num_unique_sources": num_sources,
                    "has_viewpoint_analysis": has_viewpoint_analysis,
                    "has_reflection_insights": has_reflection_insights,
                    "report_length_chars": len(html_content),
                    "report_generation_success": True,
                }
            }
        )

        # Log artifact metadata
        log_metadata(
            metadata={
                "final_report_characteristics": {
                    "report_length": len(html_content),
                    "main_query": query_context.main_query,
                    "num_sections": len(query_context.sub_questions)
                    + (1 if has_viewpoint_analysis else 0),
                    "has_executive_summary": True,
                    "has_introduction": True,
                    "has_conclusion": True,
                }
            },
            artifact_name="final_report",
            infer_artifact=True,
        )

        # Add tags to the artifact
        # add_tags(tags=["report", "final", "html"], artifact_name="final_report", infer_artifact=True)

        logger.info(
            f"Successfully generated final report ({len(html_content)} characters)"
        )
        return final_report, HTMLString(html_content)

    else:
        # Handle non-static template case (future implementation)
        logger.warning(
            "Non-static template generation not yet implemented, falling back to static template"
        )
        return pydantic_final_report_step(
            query_context=query_context,
            search_data=search_data,
            synthesis_data=synthesis_data,
            analysis_data=analysis_data,
            conclusion_generation_prompt=conclusion_generation_prompt,
            executive_summary_prompt=executive_summary_prompt,
            introduction_prompt=introduction_prompt,
            use_static_template=True,
            llm_model=llm_model,
            langfuse_project_name=langfuse_project_name,
        )
