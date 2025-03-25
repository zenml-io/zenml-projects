from openai import OpenAI
from typing import Dict, Any
from smolagents import tool
from config.prompts import RISK_ASSESSMENT_SYSTEM_PROMPT, RISK_ASSESSMENT_USER_PROMPT
from dotenv import load_dotenv
load_dotenv()

client = OpenAI()

@tool
def risk_assessment(company_data: Dict[str, Any]) -> str:
    """
    Analyzes risk disclosures and enriches with external risk factors.

    Args:
        company_data: Official company-reported financial data including risk disclosures.

    Returns:
        str: A report detailing risk factors enriched with external insights.
    """
    # Call the GPT model with risk assessment prompts
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": RISK_ASSESSMENT_SYSTEM_PROMPT },
            {"role": "user", "content": RISK_ASSESSMENT_USER_PROMPT.format(company_data=company_data)}
        ]
    )

    return completion.choices[0].message