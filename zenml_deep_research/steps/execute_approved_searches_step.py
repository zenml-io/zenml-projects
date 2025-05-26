import json
import logging
from typing import Annotated

from materializers.pydantic_materializer import ResearchStateMaterializer
from utils.llm_utils import (
    find_most_relevant_string,
    get_structured_llm_output,
    is_text_relevant,
)
from utils.prompt_models import PromptsBundle
from utils.pydantic_models import (
    ApprovalDecision,
    ReflectionMetadata,
    ReflectionOutput,
    ResearchState,
    SynthesizedInfo,
)
from utils.search_utils import search_and_extract_results
from zenml import step

logger = logging.getLogger(__name__)


def create_enhanced_info_copy(synthesized_info):
    """Create a deep copy of synthesized info for enhancement."""
    return {
        k: SynthesizedInfo(
            synthesized_answer=v.synthesized_answer,
            key_sources=v.key_sources.copy(),
            confidence_level=v.confidence_level,
            information_gaps=v.information_gaps,
            improvements=v.improvements.copy()
            if hasattr(v, "improvements")
            else [],
        )
        for k, v in synthesized_info.items()
    }


@step(output_materializers=ResearchStateMaterializer)
def execute_approved_searches_step(
    reflection_output: ReflectionOutput,
    approval_decision: ApprovalDecision,
    prompts_bundle: PromptsBundle,
    num_results_per_search: int = 3,
    cap_search_length: int = 20000,
    llm_model: str = "sambanova/DeepSeek-R1-Distill-Llama-70B",
    search_provider: str = "tavily",
    search_mode: str = "auto",
    langfuse_project_name: str = "deep-research",
) -> Annotated[ResearchState, "updated_state"]:
    """
    Execute approved searches and enhance the research state.

    This step receives the approval decision and only executes
    searches that were approved by the human reviewer (or auto-approved).

    Args:
        reflection_output: Output from the reflection generation step
        approval_decision: Human approval decision
        num_results_per_search: Number of results to fetch per search
        cap_search_length: Maximum length of content to process from search results
        llm_model: The model to use for synthesis enhancement
        prompts_bundle: Bundle containing all prompts for the pipeline
        search_provider: Search provider to use
        search_mode: Search mode for the provider

    Returns:
        Updated research state with enhanced information and reflection metadata
    """
    logger.info(
        f"Processing approval decision: {approval_decision.approval_method}"
    )

    state = reflection_output.state
    enhanced_info = create_enhanced_info_copy(state.synthesized_info)

    # Track improvements count
    improvements_count = 0

    # Check if we should execute searches
    if (
        not approval_decision.approved
        or not approval_decision.selected_queries
    ):
        logger.info("No additional searches approved")

        # Add any additional questions as new synthesized entries (from reflection)
        for new_question in reflection_output.additional_questions:
            if (
                new_question not in state.sub_questions
                and new_question not in enhanced_info
            ):
                enhanced_info[new_question] = SynthesizedInfo(
                    synthesized_answer=f"This question was identified during reflection but has not yet been researched: {new_question}",
                    key_sources=[],
                    confidence_level="low",
                    information_gaps="This question requires additional research.",
                )

        # Create metadata indicating no additional research
        reflection_metadata = ReflectionMetadata(
            critique_summary=[
                c.get("issue", "") for c in reflection_output.critique_summary
            ],
            additional_questions_identified=reflection_output.additional_questions,
            searches_performed=[],
            improvements_made=improvements_count,
        )

        # Add approval decision info to metadata
        if hasattr(reflection_metadata, "__dict__"):
            reflection_metadata.__dict__["user_decision"] = (
                approval_decision.approval_method
            )
            reflection_metadata.__dict__["reviewer_notes"] = (
                approval_decision.reviewer_notes
            )

        state.update_after_reflection(enhanced_info, reflection_metadata)
        return state

    # Execute approved searches
    logger.info(
        f"Executing {len(approval_decision.selected_queries)} approved searches"
    )

    try:
        for query in approval_decision.selected_queries:
            logger.info(f"Performing approved search: {query}")

            # Execute search using the utility function
            search_results = search_and_extract_results(
                query=query,
                max_results=num_results_per_search,
                cap_content_length=cap_search_length,
                provider=search_provider,
                search_mode=search_mode,
            )

            # Extract raw contents
            raw_contents = [result.content for result in search_results]

            # Find the most relevant sub-question for this query
            most_relevant_question = find_most_relevant_string(
                query,
                state.sub_questions,
                llm_model,
                project=langfuse_project_name,
            )

            if (
                most_relevant_question
                and most_relevant_question in enhanced_info
            ):
                # Enhance the synthesis with new information
                enhancement_input = {
                    "original_synthesis": enhanced_info[
                        most_relevant_question
                    ].synthesized_answer,
                    "new_information": raw_contents,
                    "critique": [
                        item
                        for item in reflection_output.critique_summary
                        if is_text_relevant(
                            item.get("issue", ""), most_relevant_question
                        )
                    ],
                }

                # Get the prompt from the bundle
                additional_synthesis_prompt = (
                    prompts_bundle.get_prompt_content(
                        "additional_synthesis_prompt"
                    )
                )

                # Use the utility function for enhancement
                enhanced_synthesis = get_structured_llm_output(
                    prompt=json.dumps(enhancement_input),
                    system_prompt=additional_synthesis_prompt,
                    model=llm_model,
                    fallback_response={
                        "enhanced_synthesis": enhanced_info[
                            most_relevant_question
                        ].synthesized_answer,
                        "improvements_made": ["Failed to enhance synthesis"],
                        "remaining_limitations": "Enhancement process failed.",
                    },
                    project=langfuse_project_name,
                )

                if (
                    enhanced_synthesis
                    and "enhanced_synthesis" in enhanced_synthesis
                ):
                    # Update the synthesized answer
                    enhanced_info[
                        most_relevant_question
                    ].synthesized_answer = enhanced_synthesis[
                        "enhanced_synthesis"
                    ]

                    # Add improvements
                    improvements = enhanced_synthesis.get(
                        "improvements_made", []
                    )
                    enhanced_info[most_relevant_question].improvements.extend(
                        improvements
                    )
                    improvements_count += len(improvements)

        # Add any additional questions as new synthesized entries
        for new_question in reflection_output.additional_questions:
            if (
                new_question not in state.sub_questions
                and new_question not in enhanced_info
            ):
                enhanced_info[new_question] = SynthesizedInfo(
                    synthesized_answer=f"This question was identified during reflection but has not yet been researched: {new_question}",
                    key_sources=[],
                    confidence_level="low",
                    information_gaps="This question requires additional research.",
                )

        # Create final metadata with approval info
        reflection_metadata = ReflectionMetadata(
            critique_summary=[
                c.get("issue", "") for c in reflection_output.critique_summary
            ],
            additional_questions_identified=reflection_output.additional_questions,
            searches_performed=approval_decision.selected_queries,
            improvements_made=improvements_count,
        )

        # Add approval decision info to metadata
        if hasattr(reflection_metadata, "__dict__"):
            reflection_metadata.__dict__["user_decision"] = (
                approval_decision.approval_method
            )
            reflection_metadata.__dict__["reviewer_notes"] = (
                approval_decision.reviewer_notes
            )

        logger.info(
            f"Completed approved searches with {improvements_count} improvements"
        )

        state.update_after_reflection(enhanced_info, reflection_metadata)
        return state

    except Exception as e:
        logger.error(f"Error during approved search execution: {e}")

        # Create error metadata
        error_metadata = ReflectionMetadata(
            error=f"Approved search execution failed: {str(e)}",
            critique_summary=[
                c.get("issue", "") for c in reflection_output.critique_summary
            ],
            additional_questions_identified=reflection_output.additional_questions,
            searches_performed=[],
            improvements_made=0,
        )

        # Update the state with the original synthesized info as enhanced info
        state.update_after_reflection(state.synthesized_info, error_metadata)

        return state
