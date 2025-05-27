import logging
import os

import openai
import requests
from openai import BadRequestError, RateLimitError
from zenml import get_step_context, pipeline, step

logger = logging.getLogger(__name__)

LITELLM_API_KEY = os.getenv("LITELLM_BUDGET_TEST_API_KEY")
LITELLM_BASE_URL = os.getenv("LITELLM_BUDGET_TEST_BASE_URL")

client = openai.OpenAI(
    api_key=LITELLM_API_KEY,
    base_url=LITELLM_BASE_URL,
)


@step
def set_budget(amount: float) -> None:
    """Set the budget for the research project."""
    context = get_step_context()
    run_name = context.pipeline_run.name
    breakpoint()

    # create a user with the name set to run_name
    response = requests.post(
        f"{LITELLM_BASE_URL}/user/new",
        headers={
            "Authorization": f"Bearer {LITELLM_API_KEY}",
            "Content-Type": "application/json",
        },
        json={"user_id": run_name},
    )
    logger.info(response.json())

    logger.info(f"Setting budget to ${amount:.2f} for run {run_name}")

    # Update the user with budget settings
    budget_response = requests.post(
        f"{LITELLM_BASE_URL}/user/update",
        headers={
            "Authorization": f"Bearer {LITELLM_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "user_id": run_name,
            "max_budget": amount,
            "budget_duration": "1d",  # Optional: resets daily - can be "1s", "1m", "1h", "1d", "1mo"
        },
    )
    logger.info(f"Budget update response: {budget_response.json()}")
    return


@step
def llm_functionality() -> str:
    """Test the LLM functionality by generating a short joke."""
    context = get_step_context()
    run_name = context.pipeline_run.name

    prompt = "Tell me a short joke about programming."

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            user=run_name,
            max_tokens=50,
        )
        joke = response.choices[0].message.content
        logger.info(joke)
        return joke

    except BadRequestError as e:
        if "budget" not in str(e).lower() and "exceeded" not in str(e).lower():
            raise  # Re-raise if it's a different 400 error

        logger.error(f"Budget exceeded for user {run_name}: {e}")
        raise ValueError(f"Budget exceeded for user {run_name}: {str(e)}")
    except RateLimitError as e:
        # HTTP 429 - Could be rate limits or provider budget exceeded
        if "budget" in str(e).lower():
            logger.error(f"Provider budget exceeded: {e}")
            raise ValueError(f"Provider budget exceeded: {str(e)}")
        else:
            logger.error(f"Rate limit hit: {e}")
            raise ValueError(f"Rate limit exceeded: {str(e)}")


@pipeline(enable_cache=False)
def budget_test_pipeline() -> str:
    """Test pipeline to set budget and check LLM functionality."""
    set_budget(amount=0.01)
    llm_functionality(after="set_budget")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    budget_test_pipeline()
