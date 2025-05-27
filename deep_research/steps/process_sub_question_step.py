import copy
import logging
import time
import warnings
from typing import Annotated

# Suppress Pydantic serialization warnings from ZenML artifact metadata
# These occur when ZenML stores timestamp metadata as floats but models expect ints
warnings.filterwarnings(
    "ignore", message=".*PydanticSerializationUnexpectedValue.*"
)

from materializers.pydantic_materializer import ResearchStateMaterializer
from utils.llm_utils import synthesize_information
from utils.pydantic_models import Prompt, ResearchState, SynthesizedInfo
from utils.search_utils import (
    generate_search_query,
    search_and_extract_results,
)
from zenml import add_tags, log_metadata, step

logger = logging.getLogger(__name__)


@step(output_materializers=ResearchStateMaterializer)
def process_sub_question_step(
    state: ResearchState,
    search_query_prompt: Prompt,
    synthesis_prompt: Prompt,
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
        search_query_prompt: Prompt for generating search queries
        synthesis_prompt: Prompt for synthesizing search results
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
    start_time = time.time()

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
        # Log metadata for skipped processing
        log_metadata(
            metadata={
                "sub_question_processing": {
                    "question_index": question_index,
                    "status": "skipped",
                    "reason": "index_out_of_range",
                    "total_sub_questions": len(state.sub_questions),
                }
            }
        )
        # Return an empty state since there's no question to process
        sub_state.sub_questions = []
        # Add tags to the artifact
        add_tags(tags=["state", "sub-question"], artifact="output")
        return sub_state

    # Get the target sub-question
    sub_question = state.sub_questions[question_index]
    logger.info(
        f"Processing sub-question {question_index + 1}: {sub_question}"
    )

    # Store only this sub-question in the sub-state
    sub_state.sub_questions = [sub_question]

    # === INFORMATION GATHERING ===
    search_phase_start = time.time()

    # Generate search query with prompt
    search_query_data = generate_search_query(
        sub_question=sub_question,
        model=llm_model_search,
        system_prompt=str(search_query_prompt),
        project=langfuse_project_name,
    )
    search_query = search_query_data.get(
        "search_query", f"research about {sub_question}"
    )

    # Perform search
    logger.info(f"Performing search with query: {search_query}")
    if search_provider:
        logger.info(f"Using search provider: {search_provider}")
    results_list, search_cost = search_and_extract_results(
        query=search_query,
        max_results=num_results_per_search,
        cap_content_length=cap_search_length,
        provider=search_provider,
        search_mode=search_mode,
    )

    # Track search costs if using Exa
    if (
        search_provider
        and search_provider.lower() in ["exa", "both"]
        and search_cost > 0
    ):
        # Update total costs
        sub_state.search_costs["exa"] = (
            sub_state.search_costs.get("exa", 0.0) + search_cost
        )

        # Add detailed cost entry
        sub_state.search_cost_details.append(
            {
                "provider": "exa",
                "query": search_query,
                "cost": search_cost,
                "timestamp": time.time(),
                "step": "process_sub_question",
                "sub_question": sub_question,
                "question_index": question_index,
            }
        )
        logger.info(
            f"Exa search cost for sub-question {question_index}: ${search_cost:.4f}"
        )

    search_results = {sub_question: results_list}
    sub_state.update_search_results(search_results)

    search_phase_time = time.time() - search_phase_start

    # === INFORMATION SYNTHESIS ===
    synthesis_phase_start = time.time()

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

    # Synthesize information with prompt
    synthesis_result = synthesize_information(
        synthesis_input=synthesis_input,
        model=llm_model_synthesis,
        system_prompt=str(synthesis_prompt),
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

    synthesis_phase_time = time.time() - synthesis_phase_start
    total_execution_time = time.time() - start_time

    # Calculate total content length processed
    total_content_length = sum(len(content) for content in raw_contents)

    # Get unique domains from sources
    unique_domains = set()
    for url in sources:
        try:
            from urllib.parse import urlparse

            domain = urlparse(url).netloc
            unique_domains.add(domain)
        except:
            pass

    # Log comprehensive metadata
    log_metadata(
        metadata={
            "sub_question_processing": {
                "question_index": question_index,
                "status": "completed",
                "sub_question": sub_question,
                "execution_time_seconds": total_execution_time,
                "search_phase_time_seconds": search_phase_time,
                "synthesis_phase_time_seconds": synthesis_phase_time,
                "search_query": search_query,
                "search_provider": search_provider,
                "search_mode": search_mode,
                "num_results_requested": num_results_per_search,
                "num_results_retrieved": len(results_list),
                "total_content_length": total_content_length,
                "cap_search_length": cap_search_length,
                "unique_domains": list(unique_domains),
                "llm_model_search": llm_model_search,
                "llm_model_synthesis": llm_model_synthesis,
                "confidence_level": synthesis_result.get(
                    "confidence_level", "low"
                ),
                "information_gaps": synthesis_result.get(
                    "information_gaps", ""
                ),
                "key_sources_count": len(
                    synthesis_result.get("key_sources", [])
                ),
                "search_cost": search_cost,
                "search_cost_provider": "exa"
                if search_provider
                and search_provider.lower() in ["exa", "both"]
                else None,
            }
        }
    )

    # Log model metadata for cross-pipeline tracking
    log_metadata(
        metadata={
            "search_metrics": {
                "confidence_level": synthesis_result.get(
                    "confidence_level", "low"
                ),
                "search_provider": search_provider,
            }
        },
        infer_model=True,
    )

    # Log artifact metadata for the output state
    log_metadata(
        metadata={
            "sub_state_characteristics": {
                "has_search_results": bool(sub_state.search_results),
                "has_synthesized_info": bool(sub_state.synthesized_info),
                "sub_question_processed": sub_question,
                "confidence_level": synthesis_result.get(
                    "confidence_level", "low"
                ),
            }
        },
        infer_artifact=True,
    )

    # Add tags to the artifact
    add_tags(tags=["state", "sub-question"], artifact="output")

    return sub_state
