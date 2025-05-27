import json
import logging
import time
from typing import Annotated

from materializers.pydantic_materializer import ResearchStateMaterializer
from utils.llm_utils import (
    find_most_relevant_string,
    get_structured_llm_output,
    is_text_relevant,
)
from utils.prompts import ADDITIONAL_SYNTHESIS_PROMPT, REFLECTION_PROMPT
from utils.pydantic_models import (
    ReflectionMetadata,
    ResearchState,
    SynthesizedInfo,
)
from utils.search_utils import search_and_extract_results
from zenml import log_metadata, step

logger = logging.getLogger(__name__)


@step(output_materializers=ResearchStateMaterializer)
def iterative_reflection_step(
    state: ResearchState,
    max_additional_searches: int = 2,
    num_results_per_search: int = 3,
    cap_search_length: int = 20000,
    llm_model: str = "sambanova/DeepSeek-R1-Distill-Llama-70B",
    reflection_prompt: str = REFLECTION_PROMPT,
    additional_synthesis_prompt: str = ADDITIONAL_SYNTHESIS_PROMPT,
) -> Annotated[ResearchState, "updated_state"]:
    """Perform iterative reflection on the research, identifying gaps and improving it.

    Args:
        state: The current research state
        max_additional_searches: Maximum number of additional searches to perform
        num_results_per_search: Number of results to fetch per search
        cap_search_length: Maximum length of content to process from search results
        llm_model: The model to use for reflection
        reflection_prompt: System prompt for the reflection
        additional_synthesis_prompt: System prompt for incorporating additional information

    Returns:
        Updated research state with enhanced information and reflection metadata
    """
    start_time = time.time()
    logger.info("Starting iterative reflection on research")

    # Prepare input for reflection
    synthesized_info_dict = {
        question: {
            "synthesized_answer": info.synthesized_answer,
            "key_sources": info.key_sources,
            "confidence_level": info.confidence_level,
            "information_gaps": info.information_gaps,
        }
        for question, info in state.synthesized_info.items()
    }

    viewpoint_analysis_dict = None
    if state.viewpoint_analysis:
        # Convert the viewpoint analysis to a dict for the LLM
        tension_list = []
        for tension in state.viewpoint_analysis.areas_of_tension:
            tension_list.append(
                {"topic": tension.topic, "viewpoints": tension.viewpoints}
            )

        viewpoint_analysis_dict = {
            "main_points_of_agreement": state.viewpoint_analysis.main_points_of_agreement,
            "areas_of_tension": tension_list,
            "perspective_gaps": state.viewpoint_analysis.perspective_gaps,
            "integrative_insights": state.viewpoint_analysis.integrative_insights,
        }

    reflection_input = {
        "main_query": state.main_query,
        "sub_questions": state.sub_questions,
        "synthesized_information": synthesized_info_dict,
    }

    if viewpoint_analysis_dict:
        reflection_input["viewpoint_analysis"] = viewpoint_analysis_dict

    # Get reflection critique
    try:
        logger.info(f"Generating self-critique via {llm_model}")

        # Define fallback for reflection result
        fallback_reflection = {
            "critique": [],
            "additional_questions": [],
            "recommended_search_queries": [],
        }

        # Use utility function to get structured output
        reflection_result = get_structured_llm_output(
            prompt=json.dumps(reflection_input),
            system_prompt=reflection_prompt,
            model=llm_model,
            fallback_response=fallback_reflection,
        )

        # Make a deep copy of the synthesized info to create enhanced_info
        enhanced_info = {
            k: SynthesizedInfo(
                synthesized_answer=v.synthesized_answer,
                key_sources=v.key_sources.copy(),
                confidence_level=v.confidence_level,
                information_gaps=v.information_gaps,
                improvements=v.improvements.copy()
                if hasattr(v, "improvements")
                else [],
            )
            for k, v in state.synthesized_info.items()
        }

        # Perform additional searches based on recommendations
        search_queries = reflection_result.get(
            "recommended_search_queries", []
        )
        if max_additional_searches > 0 and search_queries:
            # Limit to max_additional_searches
            search_queries = search_queries[:max_additional_searches]

            for query in search_queries:
                logger.info(f"Performing additional search: {query}")
                # Execute the search using the utility function
                search_results = search_and_extract_results(
                    query=query,
                    max_results=num_results_per_search,
                    cap_content_length=cap_search_length,
                )

                # Extract raw contents
                raw_contents = [result.content for result in search_results]

                # Find the most relevant sub-question for this query
                most_relevant_question = find_most_relevant_string(
                    query, state.sub_questions, llm_model
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
                            for item in reflection_result.get("critique", [])
                            if is_text_relevant(
                                item.get("issue", ""), most_relevant_question
                            )
                        ],
                    }

                    # Use the utility function for enhancement
                    enhanced_synthesis = get_structured_llm_output(
                        prompt=json.dumps(enhancement_input),
                        system_prompt=additional_synthesis_prompt,
                        model=llm_model,
                        fallback_response={
                            "enhanced_synthesis": enhanced_info[
                                most_relevant_question
                            ].synthesized_answer,
                            "improvements_made": [
                                "Failed to enhance synthesis"
                            ],
                            "remaining_limitations": "Enhancement process failed.",
                        },
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
                        enhanced_info[
                            most_relevant_question
                        ].improvements.extend(improvements)

        # Add any additional questions as new synthesized entries
        for new_question in reflection_result.get("additional_questions", []):
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

        # Prepare metadata about the reflection process
        reflection_metadata = ReflectionMetadata(
            critique_summary=[
                item.get("issue", "")
                for item in reflection_result.get("critique", [])
            ],
            additional_questions_identified=reflection_result.get(
                "additional_questions", []
            ),
            searches_performed=search_queries,
            improvements_made=sum(
                [len(info.improvements) for info in enhanced_info.values()]
            ),
        )

        logger.info(
            f"Completed iterative reflection with {reflection_metadata.improvements_made} improvements"
        )

        # Update the state with enhanced info and metadata
        state.update_after_reflection(enhanced_info, reflection_metadata)

        # Calculate execution time
        execution_time = time.time() - start_time

        # Count questions that were enhanced
        questions_enhanced = 0
        for question, enhanced in enhanced_info.items():
            if question in state.synthesized_info:
                original = state.synthesized_info[question]
                if enhanced.synthesized_answer != original.synthesized_answer:
                    questions_enhanced += 1

        # Calculate confidence level changes
        confidence_improvements = {"improved": 0, "unchanged": 0, "new": 0}
        for question, enhanced in enhanced_info.items():
            if question in state.synthesized_info:
                original = state.synthesized_info[question]
                original_level = original.confidence_level.lower()
                enhanced_level = enhanced.confidence_level.lower()

                level_map = {"low": 0, "medium": 1, "high": 2}
                if enhanced_level in level_map and original_level in level_map:
                    if level_map[enhanced_level] > level_map[original_level]:
                        confidence_improvements["improved"] += 1
                    else:
                        confidence_improvements["unchanged"] += 1
            else:
                confidence_improvements["new"] += 1

        # Log metadata
        log_metadata(
            metadata={
                "iterative_reflection": {
                    "execution_time_seconds": execution_time,
                    "llm_model": llm_model,
                    "max_additional_searches": max_additional_searches,
                    "searches_performed": len(search_queries),
                    "num_critique_points": len(
                        reflection_result.get("critique", [])
                    ),
                    "num_additional_questions": len(
                        reflection_result.get("additional_questions", [])
                    ),
                    "questions_enhanced": questions_enhanced,
                    "total_improvements": reflection_metadata.improvements_made,
                    "confidence_improvements": confidence_improvements,
                    "has_viewpoint_analysis": bool(viewpoint_analysis_dict),
                }
            }
        )

        # Log model metadata for cross-pipeline tracking
        log_metadata(
            metadata={
                "improvement_metrics": {
                    "confidence_improvements": confidence_improvements,
                    "total_improvements": reflection_metadata.improvements_made,
                }
            },
            infer_model=True,
        )

        # Log artifact metadata
        log_metadata(
            metadata={
                "enhanced_state_characteristics": {
                    "total_questions": len(enhanced_info),
                    "questions_with_improvements": sum(
                        1
                        for info in enhanced_info.values()
                        if info.improvements
                    ),
                    "high_confidence_count": sum(
                        1
                        for info in enhanced_info.values()
                        if info.confidence_level.lower() == "high"
                    ),
                    "medium_confidence_count": sum(
                        1
                        for info in enhanced_info.values()
                        if info.confidence_level.lower() == "medium"
                    ),
                    "low_confidence_count": sum(
                        1
                        for info in enhanced_info.values()
                        if info.confidence_level.lower() == "low"
                    ),
                }
            },
            infer_artifact=True,
        )

        return state

    except Exception as e:
        logger.error(f"Error during iterative reflection: {e}")

        # Create error metadata
        error_metadata = ReflectionMetadata(
            error=f"Reflection failed: {str(e)}"
        )

        # Update the state with the original synthesized info as enhanced info
        # and the error metadata
        state.update_after_reflection(state.synthesized_info, error_metadata)

        # Log error metadata
        execution_time = time.time() - start_time
        log_metadata(
            metadata={
                "iterative_reflection": {
                    "execution_time_seconds": execution_time,
                    "llm_model": llm_model,
                    "max_additional_searches": max_additional_searches,
                    "searches_performed": 0,
                    "status": "failed",
                    "error_message": str(e),
                }
            }
        )

        return state
