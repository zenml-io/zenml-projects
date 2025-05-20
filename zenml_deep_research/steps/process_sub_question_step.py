import copy
import logging
from typing import Annotated

from materializers.research_state_materializer import ResearchStateMaterializer

# Import functions from other steps still used in this file
from steps.information_synthesis_step import _synthesize_information
from utils.data_models import ResearchState, SynthesizedInfo
from utils.llm_utils import get_sambanova_client
from utils.search_utils import (
    generate_search_query,
    search_and_extract_results,
)
from zenml import step

logger = logging.getLogger(__name__)


@step(output_materializers=ResearchStateMaterializer)
def process_sub_question_step(
    state: ResearchState,
    question_index: int,
    sambanova_base_url: str = "https://api.sambanova.ai/v1",
    llm_model_search: str = "Meta-Llama-3.3-70B-Instruct",
    llm_model_synthesis: str = "DeepSeek-R1-Distill-Llama-70B",
    num_results_per_search: int = 3,
    cap_search_length: int = 20000,
) -> Annotated[ResearchState, "output"]:
    """Process a single sub-question if it exists at the given index.

    This step combines the gathering and synthesis steps for a single sub-question.
    It's designed to be run in parallel for each sub-question.

    Args:
        state: The original research state with all sub-questions
        question_index: The index of the sub-question to process
        sambanova_base_url: The SambaNova API base URL
        llm_model_search: Model to use for search query generation
        llm_model_synthesis: Model to use for synthesis
        num_results_per_search: Number of results to fetch per search
        cap_search_length: Maximum length of content to process from search results

    Returns:
        A new ResearchState containing only the processed sub-question's results
    """
    # Create a copy of the state to avoid modifying the original
    sub_state = copy.deepcopy(state)

    # Clear all existing data except the main query
    sub_state.search_results = {}
    sub_state.synthesized_info = {}
    sub_state.enhanced_info = {}
    sub_state.viewpoint_analysis = None
    sub_state.reflection_metadata = None
    sub_state.final_report_html = ""

    # Check if this index exists in sub-questions
    if question_index >= len(state.sub_questions):
        logger.info(
            f"No sub-question at index {question_index}, skipping processing"
        )
        # Return an empty state since there's no question to process
        sub_state.sub_questions = []
        return sub_state

    # Get the target sub-question
    sub_question = state.sub_questions[question_index]
    logger.info(f"Processing sub-question {question_index}: {sub_question}")

    # Store only this sub-question in the sub-state
    sub_state.sub_questions = [sub_question]

    # Initialize OpenAI client using the utility function
    openai_client = get_sambanova_client(base_url=sambanova_base_url)

    # === INFORMATION GATHERING ===

    # Generate search query
    search_query_data = generate_search_query(
        sub_question=sub_question,
        openai_client=openai_client,
        model=llm_model_search,
    )
    search_query = search_query_data.get(
        "search_query", f"research about {sub_question}"
    )

    # Perform search
    logger.info(f"Performing search with query: {search_query}")
    results_list = search_and_extract_results(
        query=search_query,
        max_results=num_results_per_search,
        cap_content_length=cap_search_length,
    )

    search_results = {sub_question: results_list}
    sub_state.update_search_results(search_results)

    # === INFORMATION SYNTHESIS ===

    # Extract raw contents and URLs
    raw_contents = []
    sources = []
    for result in results_list:
        raw_contents.append(result.content)
        sources.append(result.url)

    # Prepare input for synthesis
    synthesis_input = {
        "sub_question": sub_question,
        "search_results": raw_contents,
        "sources": sources,
    }

    # Synthesize information
    synthesis_result = _synthesize_information(
        synthesis_input=synthesis_input,
        openai_client=openai_client,
        model=llm_model_synthesis,
        system_prompt=None,  # Use default from the function
    )

    # Create SynthesizedInfo object
    synthesized_info = {
        sub_question: SynthesizedInfo(
            synthesized_answer=synthesis_result.get(
                "synthesized_answer", f"Synthesis for '{sub_question}' failed."
            ),
            key_sources=synthesis_result.get("key_sources", sources[:1]),
            confidence_level=synthesis_result.get("confidence_level", "low"),
            information_gaps=synthesis_result.get(
                "information_gaps",
                "Synthesis process encountered technical difficulties.",
            ),
        )
    }

    # Update the state with synthesized information
    sub_state.update_synthesized_info(synthesized_info)

    return sub_state
