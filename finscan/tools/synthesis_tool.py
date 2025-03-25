import json
from openai import OpenAI
from smolagents import tool

from typing import Dict, Any
from config.prompts import SYNTHESIS_SYSTEM_PROMPT, SYNTHESIS_USER_PROMPT
from dotenv import load_dotenv
load_dotenv()

client = OpenAI()

@tool
def synthesis(metric_result: Dict[str, Any], context_result: Dict[str, Any], competitor_result: Dict[str, Any], contradictory_result: Dict[str, Any], gap_analysis_result: Dict[str, Any]) -> str:
    """
    Synthesizes financial analysis into a cohesive report with properly attributed sources.
    
    Args:
        metric_result: Financial metric analysis.
        context_result: Market trends and analyst opinions.
        competitor_result: Competitor comparison.
        contradictory_result: conflicts present in metric and context results.
        gap_analysis_result: gaps identified that needs to be reported.
        
    Returns:
        Dictionary containing a structured financial summary.
    """

    # Call GPT model
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
            {"role": "user", "content": SYNTHESIS_USER_PROMPT.format(metric_result=metric_result, context_result=context_result, competitor_result=competitor_result, contradictory_result=contradictory_result, gap_analysis_result=gap_analysis_result)}
        ]
    )

    return {"synthesis_report": completion.choices[0].message.content.strip()}