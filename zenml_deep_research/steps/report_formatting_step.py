import json
import logging
import os
import openai
from typing import Dict, Any, List
from zenml import step
from zenml.types import HTMLString

from utils.data_models import State
from utils.helper_functions import remove_reasoning_from_output, clean_markdown_tags

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

@step
def report_formatting_step(
    final_state: State,
    sambanova_base_url: str = "https://api.sambanova.ai/v1",
    llm_model: str = "DeepSeek-R1-Distill-Llama-70B",
    system_prompt: str = REPORT_FORMATTING_PROMPT
) -> HTMLString:
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
        api_key=sambanova_api_key,
        base_url=sambanova_base_url
    )
    
    # Extract report data
    report_data = []
    for paragraph in final_state.paragraphs:
        report_data.append({
            "title": paragraph.title,
            "paragraph_latest_state": paragraph.research.latest_summary
        })
    
    try:
        # Call OpenAI API to format the report
        logger.info(f"Calling {llm_model} to format final report")
        response = openai_client.chat.completions.create(
            model=llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(report_data)}
            ]
        )
        
        # Process the response
        content = response.choices[0].message.content
        content = remove_reasoning_from_output(content)
        content = clean_markdown_tags(content)
        
        # Convert markdown to HTML
        # Note: This expects markdown content, which is what the LLM should return
        # In a full implementation, you might want to use a proper markdown to HTML converter
        
        # Here's a simple implementation that replaces common markdown elements with HTML
        html_content = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; max-width: 800px; margin: 0 auto; padding: 20px; }}
                h1 {{ color: #333; }}
                h2 {{ color: #444; border-bottom: 1px solid #eee; padding-bottom: 5px; }}
                h3 {{ color: #555; }}
                p {{ margin-bottom: 16px; }}
                .report-content {{ margin-top: 30px; }}
            </style>
        </head>
        <body>
            <div class="report-content">
                {content}
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
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; max-width: 800px; margin: 0 auto; padding: 20px; }}
                h1 {{ color: #333; }}
                h2 {{ color: #444; border-bottom: 1px solid #eee; padding-bottom: 5px; }}
                h3 {{ color: #555; }}
                p {{ margin-bottom: 16px; }}
                .report-content {{ margin-top: 30px; }}
            </style>
        </head>
        <body>
            <div class="report-content">
                <h1>Research Report: {final_state.query}</h1>
                
                {''.join([f"<h2>{p.title}</h2><p>{p.research.latest_summary}</p>" for p in final_state.paragraphs])}
            </div>
        </body>
        </html>
        """
        
        return HTMLString(fallback_html) 