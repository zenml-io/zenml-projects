import json
from openai import OpenAI
from typing import Dict, Any
from smolagents import tool
from config.prompts import CONSISTENCY_USER_PROMPT, CONSISTENCY_SYSTEM_PROMPT
from dotenv import load_dotenv
load_dotenv()

client = OpenAI()

@tool
def consistency_checker(metric_result:Dict[str, Any], context_result:Dict[str, Any], company_data:Dict[str, Any]) -> str:
    """
    Cross-validates financial metrics, market context, and competitor analysis against 
    company-reported data to detect inconsistencies.

    Args:
        metric_result: Financial metrics from the Metrics Agent.
        context_result: Market trends from the Context Agent.
        company_data: Official company-reported financial data.

    Returns:
        str: A report highlighting inconsistencies."
    """

    # Call the GPT model
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": CONSISTENCY_SYSTEM_PROMPT},
            {"role": "user", "content": CONSISTENCY_USER_PROMPT.format(metric_result=metric_result, context_result=context_result, company_data=company_data)}
        ]
    )

    return completion.choices[0].message