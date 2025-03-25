import json
from openai import OpenAI
from smolagents import tool
from typing import Dict, Any
from config.prompts import GAP_SYSTEM_PROMPT, GAP_USER_PROMPT
from dotenv import load_dotenv
load_dotenv()

client = OpenAI()

@tool
def gap_analysis(metric_result: Dict[str, Any], context_result: Dict[str, Any], competitor_result: Dict[str, Any], company_data: Dict[str, Any]) -> str:
    """
    Identifies missing critical information in the provided financial analysis.
    
    Args:
        metric_result: Financial metric analysis.
        context_result: Market trends and analyst opinions.
        competitor_result: Competitor comparison.
        company_data: Original structured financial document data.
        
    Returns:
        Dictionary containing missing data points and suggested research topics.
    """

    # Call GPT model
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": GAP_SYSTEM_PROMPT},
            {"role": "user", "content": GAP_USER_PROMPT.format(metric_result=metric_result, context_result=context_result, competitor_result=competitor_result, company_data=company_data)}
        ]
    )

    return completion.choices[0].message
