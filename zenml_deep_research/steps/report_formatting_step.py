import json
import logging
import os
import openai
import re
from datetime import datetime
from typing import Dict, Any, List
from zenml import step
from zenml.types import HTMLString
from typing import Annotated
from utils.data_models import State
from utils.helper_functions import (
    remove_reasoning_from_output,
    clean_markdown_tags,
)

logger = logging.getLogger(__name__)

# System prompt for report formatting
REPORT_FORMATTING_PROMPT = """
You are a Deep Research assistant. You have already performed the research and constructed final versions of all paragraphs in the report.
You will get the data in the following json format:

<INPUT JSON SCHEMA>
{
  "type": "array",
  "items": {
    "type": "object",
    "properties": {
      "title": {"type": "string"},
      "paragraph_latest_state": {"type": "string"}
    }
  }
}
</INPUT JSON SCHEMA>

Your job is to format the Report nicely and return it in MarkDown.
If Conclusion paragraph is not present, add it to the end of the report from the latest state of the other paragraphs.
Use titles of the paragraphs to create a title for the report.
"""


def convert_markdown_to_html(markdown_text):
    """Convert markdown to HTML with proper formatting.

    Args:
        markdown_text: The markdown text to convert

    Returns:
        HTML formatted text
    """
    # Process headings (# Heading) - handle h1 to h6
    for i in range(6, 0, -1):
        pattern = r"^{} (.+)$".format("#" * i)
        markdown_text = re.sub(
            pattern,
            r"<h{0}>\1</h{0}>".format(i),
            markdown_text,
            flags=re.MULTILINE,
        )

    # Process bold (**text**)
    markdown_text = re.sub(
        r"\*\*(.+?)\*\*", r"<strong>\1</strong>", markdown_text
    )

    # Process italic (*text*)
    markdown_text = re.sub(r"\*(.+?)\*", r"<em>\1</em>", markdown_text)

    # Process links [text](url)
    markdown_text = re.sub(
        r"\[(.+?)\]\((.+?)\)", r'<a href="\2">\1</a>', markdown_text
    )

    # Process unordered lists
    # This is a simplistic approach - a full implementation would be more complex
    list_pattern = r"^\* (.+)$"
    if re.search(list_pattern, markdown_text, re.MULTILINE):
        # Find all list items
        list_items = re.findall(list_pattern, markdown_text, re.MULTILINE)
        list_html = (
            "<ul>\n"
            + "".join([f"<li>{item}</li>\n" for item in list_items])
            + "</ul>"
        )
        markdown_text = re.sub(
            list_pattern, "", markdown_text, flags=re.MULTILINE
        )
        markdown_text = markdown_text + list_html

    # Process ordered lists
    # Simplistic approach
    ordered_list_pattern = r"^(\d+)\. (.+)$"
    if re.search(ordered_list_pattern, markdown_text, re.MULTILINE):
        # Find all list items
        list_items = re.findall(
            ordered_list_pattern, markdown_text, re.MULTILINE
        )
        list_html = (
            "<ol>\n"
            + "".join([f"<li>{item[1]}</li>\n" for item in list_items])
            + "</ol>"
        )
        markdown_text = re.sub(
            ordered_list_pattern, "", markdown_text, flags=re.MULTILINE
        )
        markdown_text = markdown_text + list_html

    # Process paragraphs - split by double newlines and wrap in <p> tags
    # We need to be careful not to wrap already processed elements
    paragraphs = []
    for line in markdown_text.split("\n\n"):
        if not line.strip():
            continue
        if not (
            line.startswith("<h")
            or line.startswith("<ul")
            or line.startswith("<ol")
            or line.startswith("<p")
        ):
            line = f"<p>{line}</p>"
        paragraphs.append(line)

    markdown_text = "\n\n".join(paragraphs)

    # Remove extra newlines
    markdown_text = re.sub(r"\n{3,}", "\n\n", markdown_text)

    return markdown_text


@step
def report_formatting_step(
    final_state: State,
    sambanova_base_url: str = "https://api.sambanova.ai/v1",
    llm_model: str = "DeepSeek-R1-Distill-Llama-70B",
    system_prompt: str = REPORT_FORMATTING_PROMPT,
) -> Annotated[HTMLString, "final_report"]:
    """Format the final report from the researched paragraphs.

    Args:
        final_state: State with researched paragraphs
        sambanova_base_url: SambaNova API base URL
        llm_model: The reasoning model to use
        system_prompt: System prompt for the LLM

    Returns:
        Formatted report as HTML
    """
    logger.info("Formatting final report")

    # Get API key directly from environment variables
    sambanova_api_key = os.environ.get("SAMBANOVA_API_KEY", "")
    if not sambanova_api_key:
        logger.error("SAMBANOVA_API_KEY environment variable not set")
        raise ValueError("SAMBANOVA_API_KEY environment variable not set")

    # Initialize OpenAI client
    openai_client = openai.OpenAI(
        api_key=sambanova_api_key, base_url=sambanova_base_url
    )

    # Extract report data
    report_data = []
    for paragraph in final_state.paragraphs:
        report_data.append(
            {
                "title": paragraph.title,
                "paragraph_latest_state": paragraph.research.latest_summary,
            }
        )

    # Get current date for report footer
    current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_title = (
        final_state.report_title or f"Research Report: {final_state.query}"
    )

    try:
        # Call OpenAI API to format the report
        logger.info(f"Calling {llm_model} to format final report")
        response = openai_client.chat.completions.create(
            model=llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(report_data)},
            ],
        )

        # Process the response
        content = response.choices[0].message.content
        content = remove_reasoning_from_output(content)
        content = clean_markdown_tags(content)

        # Convert markdown to HTML with proper formatting
        formatted_content = convert_markdown_to_html(content)

        # Create a rich HTML document with modern styling
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{report_title}</title>
            <style>
                :root {{
                    --primary-color: #2c3e50;
                    --secondary-color: #3498db;
                    --bg-color: #f8f9fa;
                    --card-bg: #ffffff;
                    --text-color: #333333;
                    --heading-color: #1a365d;
                    --border-color: #e0e0e0;
                    --accent-color: #9b59b6;
                    --success-color: #2ecc71;
                    --link-color: #0366d6;
                }}
                
                body {{
                    font-family: 'Segoe UI', Roboto, -apple-system, BlinkMacSystemFont, 'Helvetica Neue', Arial, sans-serif;
                    line-height: 1.8;
                    color: var(--text-color);
                    background-color: var(--bg-color);
                    margin: 0;
                    padding: 0;
                }}
                
                .container {{
                    max-width: 900px;
                    margin: 0 auto;
                    padding: 0 20px;
                }}
                
                .report-card {{
                    background-color: var(--card-bg);
                    border-radius: 8px;
                    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
                    margin: 30px 0;
                    overflow: hidden;
                }}
                
                .report-header {{
                    background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
                    color: white;
                    padding: 30px 40px;
                    position: relative;
                }}
                
                .report-title {{
                    font-size: 2.4em;
                    font-weight: 700;
                    margin-bottom: 15px;
                    line-height: 1.3;
                }}
                
                .report-subtitle {{
                    font-size: 1.2em;
                    opacity: 0.9;
                }}
                
                .query-box {{
                    background-color: rgba(255, 255, 255, 0.1);
                    border-left: 4px solid white;
                    padding: 15px 20px;
                    margin-top: 20px;
                    border-radius: 4px;
                }}
                
                .report-content {{
                    padding: 40px;
                }}
                
                .report-content h1 {{
                    color: var(--heading-color);
                    font-size: 2.2em;
                    margin-top: 1.5em;
                    margin-bottom: 0.8em;
                    border-bottom: 2px solid var(--border-color);
                    padding-bottom: 0.3em;
                }}
                
                .report-content h2 {{
                    color: var(--heading-color);
                    font-size: 1.8em;
                    margin-top: 1.8em;
                    margin-bottom: 0.8em;
                    border-bottom: 1px solid var(--border-color);
                    padding-bottom: 0.3em;
                }}
                
                .report-content h3 {{
                    color: var(--heading-color);
                    font-size: 1.5em;
                    margin-top: 1.5em;
                    margin-bottom: 0.8em;
                }}
                
                .report-content h4 {{
                    color: var(--heading-color);
                    font-size: 1.3em;
                    margin-top: 1.5em;
                    margin-bottom: 0.8em;
                }}
                
                .report-content p {{
                    margin-bottom: 1.2em;
                    line-height: 1.8;
                    text-align: justify;
                }}
                
                .report-content ul, .report-content ol {{
                    margin-bottom: 1.5em;
                    padding-left: 2em;
                }}
                
                .report-content li {{
                    margin-bottom: 0.5em;
                }}
                
                .report-content blockquote {{
                    border-left: 4px solid var(--accent-color);
                    padding: 0.5em 1.2em;
                    margin-left: 0;
                    margin-right: 0;
                    margin-bottom: 1.5em;
                    background-color: rgba(155, 89, 182, 0.05);
                    color: #555;
                    font-style: italic;
                }}
                
                .report-content code {{
                    background-color: #f5f5f5;
                    padding: 0.2em 0.4em;
                    border-radius: 3px;
                    font-family: Consolas, Monaco, 'Andale Mono', monospace;
                    font-size: 0.9em;
                }}
                
                .report-content pre {{
                    background-color: #f5f5f5;
                    padding: 1.2em;
                    border-radius: 5px;
                    overflow-x: auto;
                    margin-bottom: 1.5em;
                }}
                
                .report-content a {{
                    color: var(--link-color);
                    text-decoration: none;
                    border-bottom: 1px dotted var(--link-color);
                }}
                
                .report-content a:hover {{
                    border-bottom: 1px solid var(--link-color);
                }}
                
                .report-content img {{
                    max-width: 100%;
                    height: auto;
                    border-radius: 5px;
                    display: block;
                    margin: 25px auto;
                    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
                }}
                
                .report-footer {{
                    border-top: 1px solid var(--border-color);
                    margin-top: 40px;
                    padding-top: 20px;
                    font-size: 0.9em;
                    color: #777;
                    text-align: center;
                }}
                
                .report-meta {{
                    display: flex;
                    justify-content: space-between;
                    margin-top: 15px;
                    color: #777;
                    font-size: 0.9em;
                }}
                
                .report-date {{
                    display: flex;
                    align-items: center;
                }}
                
                .report-date svg {{
                    margin-right: 5px;
                }}
                
                @media (max-width: 768px) {{
                    .report-header {{
                        padding: 25px;
                    }}
                    
                    .report-title {{
                        font-size: 1.8em;
                    }}
                    
                    .report-content {{
                        padding: 25px;
                    }}
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="report-card">
                    <div class="report-header">
                        <h1 class="report-title">{report_title}</h1>
                        <div class="query-box">
                            <p><strong>Research Query:</strong> {final_state.query}</p>
                        </div>
                    </div>
                    
                    <div class="report-content">
                        {formatted_content}
                        
                        <div class="report-footer">
                            <p>Generated on {current_date} • ZenML Deep Research</p>
                        </div>
                    </div>
                </div>
            </div>
        </body>
        </html>
        """

        logger.info("Report formatting completed")

        return HTMLString(html_content)

    except Exception as e:
        logger.error(f"Error formatting report: {e}")

        # Create a fallback HTML report if formatting fails
        fallback_html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Research Report</title>
            <style>
                body {{
                    font-family: 'Segoe UI', Roboto, -apple-system, BlinkMacSystemFont, 'Helvetica Neue', Arial, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    max-width: 900px;
                    margin: 0 auto;
                    padding: 20px;
                    background-color: #f8f9fa;
                }}
                
                .report-card {{
                    background-color: white;
                    border-radius: 8px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                    padding: 30px;
                    margin-bottom: 30px;
                }}
                
                h1 {{
                    color: #2c3e50;
                    border-bottom: 2px solid #eee;
                    padding-bottom: 10px;
                }}
                
                h2 {{
                    color: #2c3e50;
                    margin-top: 25px;
                    border-bottom: 1px solid #eee;
                    padding-bottom: 5px;
                }}
                
                p {{
                    margin-bottom: 16px;
                }}
                
                .query-box {{
                    background-color: #f1f8ff;
                    border-left: 4px solid #0366d6;
                    padding: 15px;
                    margin: 20px 0;
                }}
                
                .footer {{
                    margin-top: 30px;
                    padding-top: 15px;
                    border-top: 1px solid #eee;
                    font-size: 0.9em;
                    color: #777;
                    text-align: center;
                }}
            </style>
        </head>
        <body>
            <div class="report-card">
                <h1>Research Report: {final_state.query}</h1>
                
                <div class="query-box">
                    <p><strong>Research Query:</strong> {final_state.query}</p>
                </div>
                
                {"".join([f"<h2>{p.title}</h2><p>{p.research.latest_summary}</p>" for p in final_state.paragraphs])}
                
                <div class="footer">
                    <p>Generated on {current_date} • ZenML Deep Research</p>
                </div>
            </div>
        </body>
        </html>
        """

        return HTMLString(fallback_html)
