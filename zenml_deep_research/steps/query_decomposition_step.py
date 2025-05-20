import logging
from typing import Annotated

from materializers.research_state_materializer import ResearchStateMaterializer
from utils.data_models import ResearchState
from utils.llm_utils import get_sambanova_client, get_structured_llm_output
from utils.prompts import QUERY_DECOMPOSITION_PROMPT
from zenml import step

logger = logging.getLogger(__name__)


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

    # Initialize OpenAI client using the utility function
    openai_client = get_sambanova_client(base_url=sambanova_base_url)

    try:
        # Call OpenAI API to decompose the query
        updated_system_prompt = (
            system_prompt
            + f"\nPlease generate at most {max_sub_questions} sub-questions."
        )
        logger.info(
            f"Calling {llm_model} to decompose query into max {max_sub_questions} sub-questions"
        )

        # Define fallback questions
        fallback_questions = [
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

        # Use utility function to get structured output
        decomposed_questions = get_structured_llm_output(
            prompt=state.main_query,
            system_prompt=updated_system_prompt,
            client=openai_client,
            model=llm_model,
            fallback_response=fallback_questions,
        )

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
