import copy
import logging
import warnings
from typing import Annotated

# Suppress Pydantic serialization warnings from ZenML artifact metadata
# These occur when ZenML stores timestamp metadata as floats but models expect ints
warnings.filterwarnings(
    "ignore", message=".*PydanticSerializationUnexpectedValue.*"
)

from materializers.pydantic_materializer import ResearchStateMaterializer
from utils.llm_utils import synthesize_information
from utils.prompt_models import PromptsBundle
from utils.pydantic_models import ResearchState, SynthesizedInfo
from utils.search_utils import (
    generate_search_query,
    search_and_extract_results,
)
from zenml import step

logger = logging.getLogger(__name__)


@step(output_materializers=ResearchStateMaterializer)
def process_sub_question_step(
    state: ResearchState,
    prompts_bundle: PromptsBundle,
    question_index: int,
    llm_model_search: str = "sambanova/DeepSeek-R1-Distill-Llama-70B",
    llm_model_synthesis: str = "sambanova/DeepSeek-R1-Distill-Llama-70B",
    num_results_per_search: int = 3,
    cap_search_length: int = 20000,
    search_provider: str = "tavily",
    search_mode: str = "auto",
    langfuse_project_name: str = "deep-research",
) -> Annotated[ResearchState, "output"]:
    """Process a single sub-question if it exists at the given index.

    This step combines the gathering and synthesis steps for a single sub-question.
    It's designed to be run in parallel for each sub-question.

    Args:
        state: The original research state with all sub-questions
        prompts_bundle: Bundle containing all prompts for the pipeline
        question_index: The index of the sub-question to process
        llm_model_search: Model to use for search query generation
        llm_model_synthesis: Model to use for synthesis
        num_results_per_search: Number of results to fetch per search
        cap_search_length: Maximum length of content to process from search results
        search_provider: Search provider to use (tavily, exa, or both)
        search_mode: Search mode for Exa provider (neural, keyword, or auto)

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
    logger.info(
        f"Processing sub-question {question_index + 1}: {sub_question}"
    )

    # Store only this sub-question in the sub-state
    sub_state.sub_questions = [sub_question]

    # === INFORMATION GATHERING ===

    # Generate search query with prompt from bundle
    search_query_prompt = prompts_bundle.get_prompt_content(
        "search_query_prompt"
    )
    search_query_data = generate_search_query(
        sub_question=sub_question,
        model=llm_model_search,
        system_prompt=search_query_prompt,
        project=langfuse_project_name,
    )
    search_query = search_query_data.get(
        "search_query", f"research about {sub_question}"
    )

    # Perform search
    logger.info(f"Performing search with query: {search_query}")
    if search_provider:
        logger.info(f"Using search provider: {search_provider}")
    results_list = search_and_extract_results(
        query=search_query,
        max_results=num_results_per_search,
        cap_content_length=cap_search_length,
        provider=search_provider,
        search_mode=search_mode,
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

    # Synthesize information with prompt from bundle
    synthesis_prompt = prompts_bundle.get_prompt_content("synthesis_prompt")
    synthesis_result = synthesize_information(
        synthesis_input=synthesis_input,
        model=llm_model_synthesis,
        system_prompt=synthesis_prompt,
        project=langfuse_project_name,
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
            improvements=synthesis_result.get("improvements", []),
        )
    }

    # Update the state with synthesized information
    sub_state.update_synthesized_info(synthesized_info)

    return sub_state
