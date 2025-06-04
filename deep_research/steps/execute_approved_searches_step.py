import json
import logging
import time
from typing import Annotated, List, Tuple

from materializers.analysis_data_materializer import AnalysisDataMaterializer
from materializers.search_data_materializer import SearchDataMaterializer
from materializers.synthesis_data_materializer import SynthesisDataMaterializer
from utils.llm_utils import (
    find_most_relevant_string,
    get_structured_llm_output,
    is_text_relevant,
)
from utils.pydantic_models import (
    AnalysisData,
    ApprovalDecision,
    Prompt,
    QueryContext,
    SearchCostDetail,
    SearchData,
    SynthesisData,
    SynthesizedInfo,
)
from utils.search_utils import search_and_extract_results
from zenml import log_metadata, step

logger = logging.getLogger(__name__)


@step(
    output_materializers={
        "enhanced_search_data": SearchDataMaterializer,
        "enhanced_synthesis_data": SynthesisDataMaterializer,
        "updated_analysis_data": AnalysisDataMaterializer,
    }
)
def execute_approved_searches_step(
    query_context: QueryContext,
    search_data: SearchData,
    synthesis_data: SynthesisData,
    analysis_data: AnalysisData,
    recommended_queries: List[str],
    approval_decision: ApprovalDecision,
    additional_synthesis_prompt: Prompt,
    num_results_per_search: int = 3,
    cap_search_length: int = 20000,
    llm_model: str = "openrouter/google/gemini-2.0-flash-lite-001",
    search_provider: str = "tavily",
    search_mode: str = "auto",
    langfuse_project_name: str = "deep-research",
) -> Tuple[
    Annotated[SearchData, "enhanced_search_data"],
    Annotated[SynthesisData, "enhanced_synthesis_data"],
    Annotated[AnalysisData, "updated_analysis_data"],
]:
    """Execute approved searches and enhance the research artifacts.

    This step receives the approval decision and only executes
    searches that were approved by the human reviewer (or auto-approved).

    Args:
        query_context: The query context with main query and sub-questions
        search_data: The existing search data
        synthesis_data: The existing synthesis data
        analysis_data: The analysis data with viewpoint and reflection metadata
        recommended_queries: The recommended queries from reflection
        approval_decision: Human approval decision
        additional_synthesis_prompt: Prompt for synthesis enhancement
        num_results_per_search: Number of results to fetch per search
        cap_search_length: Maximum length of content to process from search results
        llm_model: The model to use for synthesis enhancement
        search_provider: Search provider to use
        search_mode: Search mode for the provider
        langfuse_project_name: Project name for tracing

    Returns:
        Tuple of enhanced SearchData, SynthesisData, and updated AnalysisData
    """
    start_time = time.time()
    logger.info(
        f"Processing approval decision: {approval_decision.approval_method}"
    )

    # Create copies of the data to enhance
    enhanced_search_data = SearchData(
        search_results=search_data.search_results.copy(),
        search_costs=search_data.search_costs.copy(),
        search_cost_details=search_data.search_cost_details.copy(),
        total_searches=search_data.total_searches,
    )

    enhanced_synthesis_data = SynthesisData(
        synthesized_info=synthesis_data.synthesized_info.copy(),
        enhanced_info={},  # Will be populated with enhanced versions
    )

    # Track improvements count
    improvements_count = 0

    # Check if we should execute searches
    if (
        not approval_decision.approved
        or not approval_decision.selected_queries
    ):
        logger.info("No additional searches approved")

        # Update reflection metadata with no searches
        if analysis_data.reflection_metadata:
            analysis_data.reflection_metadata.searches_performed = []
            analysis_data.reflection_metadata.improvements_made = 0.0

        # Log metadata for no approved searches
        execution_time = time.time() - start_time
        log_metadata(
            metadata={
                "execute_approved_searches": {
                    "execution_time_seconds": execution_time,
                    "approval_method": approval_decision.approval_method,
                    "approval_status": "not_approved"
                    if not approval_decision.approved
                    else "no_queries",
                    "num_queries_approved": 0,
                    "num_searches_executed": 0,
                    "num_recommended": len(recommended_queries),
                    "improvements_made": improvements_count,
                    "search_provider": search_provider,
                    "llm_model": llm_model,
                }
            }
        )

        # Add tags to the artifacts
        # add_tags(
        #     tags=["search", "not-enhanced"], artifact_name="enhanced_search_data", infer_artifact=True
        # )
        # add_tags(
        #     tags=["synthesis", "not-enhanced"],
        #     artifact_name="enhanced_synthesis_data",
        #     infer_artifact=True,
        # )
        # add_tags(
        #     tags=["analysis", "no-searches"], artifact_name="updated_analysis_data", infer_artifact=True
        # )

        return enhanced_search_data, enhanced_synthesis_data, analysis_data

    # Execute approved searches
    logger.info(
        f"Executing {len(approval_decision.selected_queries)} approved searches"
    )

    try:
        search_enhancements = []  # Track search results for metadata

        for query in approval_decision.selected_queries:
            logger.info(f"Performing approved search: {query}")

            # Execute search using the utility function
            search_results, search_cost = search_and_extract_results(
                query=query,
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
                enhanced_search_data.search_costs["exa"] = (
                    enhanced_search_data.search_costs.get("exa", 0.0)
                    + search_cost
                )

                # Add detailed cost entry
                enhanced_search_data.search_cost_details.append(
                    SearchCostDetail(
                        provider="exa",
                        query=query,
                        cost=search_cost,
                        timestamp=time.time(),
                        step="execute_approved_searches",
                        sub_question=None,  # These are reflection queries
                    )
                )
                logger.info(
                    f"Exa search cost for approved query: ${search_cost:.4f}"
                )

            # Update total searches
            enhanced_search_data.total_searches += 1

            # Extract raw contents
            raw_contents = [result.content for result in search_results]

            # Find the most relevant sub-question for this query
            most_relevant_question = find_most_relevant_string(
                query,
                query_context.sub_questions,
                llm_model,
                project=langfuse_project_name,
            )

            if (
                most_relevant_question
                and most_relevant_question in synthesis_data.synthesized_info
            ):
                # Store the search results under the relevant question
                if (
                    most_relevant_question
                    in enhanced_search_data.search_results
                ):
                    enhanced_search_data.search_results[
                        most_relevant_question
                    ].extend(search_results)
                else:
                    enhanced_search_data.search_results[
                        most_relevant_question
                    ] = search_results

                # Enhance the synthesis with new information
                original_synthesis = synthesis_data.synthesized_info[
                    most_relevant_question
                ]

                enhancement_input = {
                    "original_synthesis": original_synthesis.synthesized_answer,
                    "new_information": raw_contents,
                    "critique": [
                        item
                        for item in analysis_data.reflection_metadata.critique_summary
                        if is_text_relevant(item, most_relevant_question)
                    ]
                    if analysis_data.reflection_metadata
                    else [],
                }

                # Use the utility function for enhancement
                enhanced_synthesis = get_structured_llm_output(
                    prompt=json.dumps(enhancement_input),
                    system_prompt=str(additional_synthesis_prompt),
                    model=llm_model,
                    fallback_response={
                        "enhanced_synthesis": original_synthesis.synthesized_answer,
                        "improvements_made": ["Failed to enhance synthesis"],
                        "remaining_limitations": "Enhancement process failed.",
                    },
                    project=langfuse_project_name,
                )

                if (
                    enhanced_synthesis
                    and "enhanced_synthesis" in enhanced_synthesis
                ):
                    # Create enhanced synthesis info
                    enhanced_info = SynthesizedInfo(
                        synthesized_answer=enhanced_synthesis[
                            "enhanced_synthesis"
                        ],
                        key_sources=original_synthesis.key_sources
                        + [r.url for r in search_results[:2]],
                        confidence_level="high"
                        if original_synthesis.confidence_level == "medium"
                        else original_synthesis.confidence_level,
                        information_gaps=enhanced_synthesis.get(
                            "remaining_limitations", ""
                        ),
                        improvements=original_synthesis.improvements
                        + enhanced_synthesis.get("improvements_made", []),
                    )

                    # Store in enhanced_info
                    enhanced_synthesis_data.enhanced_info[
                        most_relevant_question
                    ] = enhanced_info

                    improvements = enhanced_synthesis.get(
                        "improvements_made", []
                    )
                    improvements_count += len(improvements)

                    # Track enhancement for metadata
                    search_enhancements.append(
                        {
                            "query": query,
                            "relevant_question": most_relevant_question,
                            "num_results": len(search_results),
                            "improvements": len(improvements),
                            "enhanced": True,
                            "search_cost": search_cost
                            if search_provider
                            and search_provider.lower() in ["exa", "both"]
                            else 0.0,
                        }
                    )

        # Update reflection metadata with search info
        if analysis_data.reflection_metadata:
            analysis_data.reflection_metadata.searches_performed = (
                approval_decision.selected_queries
            )
            analysis_data.reflection_metadata.improvements_made = float(
                improvements_count
            )

        logger.info(
            f"Completed approved searches with {improvements_count} improvements"
        )

        # Calculate metrics for metadata
        execution_time = time.time() - start_time
        total_results = sum(
            e.get("num_results", 0) for e in search_enhancements
        )
        questions_enhanced = len(
            set(
                e.get("relevant_question")
                for e in search_enhancements
                if e.get("enhanced")
            )
        )

        # Log successful execution metadata
        log_metadata(
            metadata={
                "execute_approved_searches": {
                    "execution_time_seconds": execution_time,
                    "approval_method": approval_decision.approval_method,
                    "approval_status": "approved",
                    "num_queries_recommended": len(recommended_queries),
                    "num_queries_approved": len(
                        approval_decision.selected_queries
                    ),
                    "num_searches_executed": len(
                        approval_decision.selected_queries
                    ),
                    "total_search_results": total_results,
                    "questions_enhanced": questions_enhanced,
                    "improvements_made": improvements_count,
                    "search_provider": search_provider,
                    "search_mode": search_mode,
                    "llm_model": llm_model,
                    "success": True,
                    "total_search_cost": enhanced_search_data.search_costs.get(
                        "exa", 0.0
                    ),
                }
            }
        )

        # Log artifact metadata
        log_metadata(
            metadata={
                "search_data_characteristics": {
                    "new_searches": len(approval_decision.selected_queries),
                    "total_searches": enhanced_search_data.total_searches,
                    "additional_cost": enhanced_search_data.search_costs.get(
                        "exa", 0.0
                    )
                    - search_data.search_costs.get("exa", 0.0),
                }
            },
            artifact_name="enhanced_search_data",
            infer_artifact=True,
        )

        log_metadata(
            metadata={
                "synthesis_data_characteristics": {
                    "questions_enhanced": questions_enhanced,
                    "total_enhancements": len(
                        enhanced_synthesis_data.enhanced_info
                    ),
                    "improvements_made": improvements_count,
                }
            },
            artifact_name="enhanced_synthesis_data",
            infer_artifact=True,
        )

        log_metadata(
            metadata={
                "analysis_data_characteristics": {
                    "searches_performed": len(
                        approval_decision.selected_queries
                    ),
                    "approval_method": approval_decision.approval_method,
                }
            },
            artifact_name="updated_analysis_data",
            infer_artifact=True,
        )

        # Add tags to the artifacts
        # add_tags(tags=["search", "enhanced"], artifact_name="enhanced_search_data", infer_artifact=True)
        # add_tags(
        #     tags=["synthesis", "enhanced"], artifact_name="enhanced_synthesis_data", infer_artifact=True
        # )
        # add_tags(
        #     tags=["analysis", "with-searches"],
        #     artifact_name="updated_analysis_data",
        #     infer_artifact=True,
        # )

        return enhanced_search_data, enhanced_synthesis_data, analysis_data

    except Exception as e:
        logger.error(f"Error during approved search execution: {e}")

        # Update reflection metadata with error
        if analysis_data.reflection_metadata:
            analysis_data.reflection_metadata.error = (
                f"Approved search execution failed: {str(e)}"
            )
            analysis_data.reflection_metadata.searches_performed = []
            analysis_data.reflection_metadata.improvements_made = 0.0

        # Log error metadata
        execution_time = time.time() - start_time
        log_metadata(
            metadata={
                "execute_approved_searches": {
                    "execution_time_seconds": execution_time,
                    "approval_method": approval_decision.approval_method,
                    "approval_status": "approved",
                    "num_queries_approved": len(
                        approval_decision.selected_queries
                    ),
                    "num_searches_executed": 0,
                    "improvements_made": 0,
                    "search_provider": search_provider,
                    "llm_model": llm_model,
                    "success": False,
                    "error_message": str(e),
                }
            }
        )

        # Add tags to the artifacts
        # add_tags(tags=["search", "error"], artifact_name="enhanced_search_data", infer_artifact=True)
        # add_tags(
        #     tags=["synthesis", "error"], artifact_name="enhanced_synthesis_data", infer_artifact=True
        # )
        # add_tags(tags=["analysis", "error"], artifact_name="updated_analysis_data", infer_artifact=True)

        return enhanced_search_data, enhanced_synthesis_data, analysis_data
