import logging
import os
from typing import Annotated

import openai
from materializers.research_state_materializer import ResearchStateMaterializer
from utils.data_models import ResearchState
from utils.helper_functions import (
    clean_json_tags,
    remove_reasoning_from_output,
    safe_json_loads,
)
from zenml import step

logger = logging.getLogger(__name__)

# System prompt for the query decomposition
QUERY_DECOMPOSITION_PROMPT = """
You are a Deep Research assistant. Given a complex research query, your task is to break it down into specific sub-questions that 
would help create a comprehensive understanding of the topic.

A good set of sub-questions should:
1. Cover different aspects or dimensions of the main query
2. Include both factual and analytical questions
3. Build towards a complete understanding of the topic
4. Be specific enough to guide targeted research

Format the output in json with the following json schema definition:

<OUTPUT JSON SCHEMA>
{
  "type": "array",
  "items": {
    "type": "object",
    "properties": {
      "sub_question": {"type": "string"},
      "reasoning": {"type": "string"}
    }
  }
}
</OUTPUT JSON SCHEMA>

Make sure that the output is a json object with an output json schema defined above.
Only return the json object, no explanation or additional text.
"""


@step(output_materializers=ResearchStateMaterializer)
def initial_query_decomposition_step(
    state: ResearchState,
    sambanova_base_url: str = "https://api.sambanova.ai/v1",
    llm_model: str = "DeepSeek-R1-Distill-Llama-70B",
    system_prompt: str = QUERY_DECOMPOSITION_PROMPT,
    max_sub_questions: int = 8,
) -> Annotated[ResearchState, "updated_state"]:
    """Break down a complex research query into specific sub-questions.

    Args:
        state: The current research state
        sambanova_base_url: SambaNova API base URL
        llm_model: The reasoning model to use
        system_prompt: System prompt for the LLM
        max_sub_questions: Maximum number of sub-questions to generate

    Returns:
        Updated research state with sub-questions
    """
    logger.info(f"Decomposing research query: {state.main_query}")

    # Get API key directly from environment variables
    sambanova_api_key = os.environ.get("SAMBANOVA_API_KEY", "")
    if not sambanova_api_key:
        logger.error("SAMBANOVA_API_KEY environment variable not set")
        raise ValueError("SAMBANOVA_API_KEY environment variable not set")

    # Initialize OpenAI client
    openai_client = openai.OpenAI(
        api_key=sambanova_api_key, base_url=sambanova_base_url
    )

    try:
        # Call OpenAI API to decompose the query
        updated_system_prompt = (
            system_prompt
            + f"\nPlease generate at most {max_sub_questions} sub-questions."
        )
        logger.info(
            f"Calling {llm_model} to decompose query into max {max_sub_questions} sub-questions"
        )
        response = openai_client.chat.completions.create(
            model=llm_model,
            messages=[
                {"role": "system", "content": updated_system_prompt},
                {"role": "user", "content": state.main_query},
            ],
        )

        # Process the response
        content = response.choices[0].message.content
        content = remove_reasoning_from_output(content)
        content = clean_json_tags(content)

        # Parse the JSON
        decomposed_questions = safe_json_loads(content)

        if not decomposed_questions:
            logger.warning(
                "Failed to parse query decomposition, using fallback questions"
            )
            decomposed_questions = [
                {
                    "sub_question": f"What is {state.main_query}?",
                    "reasoning": "Basic understanding of the topic",
                },
                {
                    "sub_question": f"What are the key aspects of {state.main_query}?",
                    "reasoning": "Exploring important dimensions",
                },
                {
                    "sub_question": f"What are the implications of {state.main_query}?",
                    "reasoning": "Understanding broader impact",
                },
            ]

        # Extract just the sub-questions
        sub_questions = [
            item.get("sub_question")
            for item in decomposed_questions
            if "sub_question" in item
        ]

        # Limit to max_sub_questions
        sub_questions = sub_questions[:max_sub_questions]

        logger.info(f"Generated {len(sub_questions)} sub-questions")

        # Update the state with the new sub-questions
        state.update_sub_questions(sub_questions)

        return state

    except Exception as e:
        logger.error(f"Error decomposing query: {e}")
        # Return fallback questions in the state
        fallback_questions = [
            f"What is {state.main_query}?",
            f"What are the key aspects of {state.main_query}?",
            f"What are the implications of {state.main_query}?",
        ]
        fallback_questions = fallback_questions[:max_sub_questions]
        state.update_sub_questions(fallback_questions)
        return state
