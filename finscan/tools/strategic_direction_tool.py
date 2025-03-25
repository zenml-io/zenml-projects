
from openai import OpenAI
from typing import Dict, Any
from smolagents import tool
from config.prompts import STRATEGIC_DIRECTION_SYSTEM_PROMPT,STRATEGIC_DIRECTION_USER_PROMPT
from dotenv import load_dotenv
load_dotenv()

client = OpenAI()
@tool
def strategic_direction(company_data: Dict[str, Any]) -> str:
    """
    Evaluates stated company strategy against market position and execution history.

    Args:
        company_data: Official company-reported financial data including strategic disclosures.

    Returns:
        str: A report evaluating the company's strategic direction in the context of market performance and historical execution.
    """
    # Call the GPT model with strategic direction prompts
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": STRATEGIC_DIRECTION_SYSTEM_PROMPT},
            {"role": "user", "content": STRATEGIC_DIRECTION_USER_PROMPT.format(company_data=company_data)}
        ]
    )

    return completion.choices[0].message