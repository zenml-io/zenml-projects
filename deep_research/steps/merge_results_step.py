import copy
import logging
import time
from typing import Annotated

from materializers.pydantic_materializer import ResearchStateMaterializer
from utils.pydantic_models import ResearchState
from zenml import add_tags, get_step_context, log_metadata, step
from zenml.client import Client

logger = logging.getLogger(__name__)


@step(output_materializers=ResearchStateMaterializer)
def merge_sub_question_results_step(
    original_state: ResearchState,
    step_prefix: str = "process_question_",
    output_name: str = "output",
) -> Annotated[ResearchState, "merged_state"]:
    """Merge results from individual sub-question processing steps.

    This step collects the results from the parallel sub-question processing steps
    and combines them into a single, comprehensive state object.

    Args:
        original_state: The original research state with all sub-questions
        step_prefix: The prefix used in step IDs for the parallel processing steps
        output_name: The name of the output artifact from the processing steps

    Returns:
        Annotated[ResearchState, "merged_state"]: A merged ResearchState with combined
            results from all sub-questions

    Note:
        This step is typically configured with the 'after' parameter in the pipeline
        definition to ensure it runs after all parallel sub-question processing steps
        have completed.
    """
    start_time = time.time()

    # Start with the original state that has all sub-questions
    merged_state = copy.deepcopy(original_state)

    # Initialize empty dictionaries for the results
    merged_state.search_results = {}
    merged_state.synthesized_info = {}

    # Initialize search cost tracking
    merged_state.search_costs = {}
    merged_state.search_cost_details = []

    # Get pipeline run information to access outputs
    try:
        ctx = get_step_context()
        if not ctx or not ctx.pipeline_run:
            logger.error("Could not get pipeline run context")
            return merged_state

        run_name = ctx.pipeline_run.name
        client = Client()
        run = client.get_pipeline_run(run_name)

        logger.info(
            f"Merging results from parallel sub-question processing steps in run: {run_name}"
        )

        # Track which sub-questions were successfully processed
        processed_questions = set()
        parallel_steps_processed = 0

        # Process each step in the run
        for step_name, step_info in run.steps.items():
            # Only process steps with the specified prefix
            if step_name.startswith(step_prefix):
                try:
                    # Extract the sub-question index from the step name
                    if "_" in step_name:
                        index = int(step_name.split("_")[-1])
                        logger.info(
                            f"Processing results from step: {step_name} (index: {index})"
                        )

                        # Get the output artifact
                        if output_name in step_info.outputs:
                            output_artifacts = step_info.outputs[output_name]
                            if output_artifacts:
                                output_artifact = output_artifacts[0]
                                sub_state = output_artifact.load()

                                # Check if the sub-state has valid data
                                if (
                                    hasattr(sub_state, "sub_questions")
                                    and sub_state.sub_questions
                                ):
                                    sub_question = sub_state.sub_questions[0]
                                    logger.info(
                                        f"Found results for sub-question: {sub_question}"
                                    )
                                    parallel_steps_processed += 1
                                    processed_questions.add(sub_question)

                                    # Merge search results
                                    if (
                                        hasattr(sub_state, "search_results")
                                        and sub_question
                                        in sub_state.search_results
                                    ):
                                        merged_state.search_results[
                                            sub_question
                                        ] = sub_state.search_results[
                                            sub_question
                                        ]
                                        logger.info(
                                            f"Added search results for: {sub_question}"
                                        )

                                    # Merge synthesized info
                                    if (
                                        hasattr(sub_state, "synthesized_info")
                                        and sub_question
                                        in sub_state.synthesized_info
                                    ):
                                        merged_state.synthesized_info[
                                            sub_question
                                        ] = sub_state.synthesized_info[
                                            sub_question
                                        ]
                                        logger.info(
                                            f"Added synthesized info for: {sub_question}"
                                        )

                                    # Merge search costs
                                    if hasattr(sub_state, "search_costs"):
                                        for (
                                            provider,
                                            cost,
                                        ) in sub_state.search_costs.items():
                                            merged_state.search_costs[
                                                provider
                                            ] = (
                                                merged_state.search_costs.get(
                                                    provider, 0.0
                                                )
                                                + cost
                                            )

                                    # Merge search cost details
                                    if hasattr(
                                        sub_state, "search_cost_details"
                                    ):
                                        merged_state.search_cost_details.extend(
                                            sub_state.search_cost_details
                                        )
                except (ValueError, IndexError, KeyError, AttributeError) as e:
                    logger.warning(f"Error processing step {step_name}: {e}")
                    continue

        # Log summary
        logger.info(
            f"Merged results from {parallel_steps_processed} parallel steps"
        )
        logger.info(
            f"Successfully processed {len(processed_questions)} sub-questions"
        )

        # Log search cost summary
        if merged_state.search_costs:
            total_cost = sum(merged_state.search_costs.values())
            logger.info(
                f"Total search costs merged: ${total_cost:.4f} across {len(merged_state.search_cost_details)} queries"
            )
            for provider, cost in merged_state.search_costs.items():
                logger.info(f"  {provider}: ${cost:.4f}")

        # Check for any missing sub-questions
        for sub_q in merged_state.sub_questions:
            if sub_q not in processed_questions:
                logger.warning(f"Missing results for sub-question: {sub_q}")

    except Exception as e:
        logger.error(f"Error during merge step: {e}")

    # Final check for empty results
    if not merged_state.search_results or not merged_state.synthesized_info:
        logger.warning(
            "No results were found or merged from parallel processing steps!"
        )

    # Calculate execution time
    execution_time = time.time() - start_time

    # Calculate metrics
    missing_questions = [
        q for q in merged_state.sub_questions if q not in processed_questions
    ]

    # Count total search results across all questions
    total_search_results = sum(
        len(results) for results in merged_state.search_results.values()
    )

    # Get confidence distribution for merged results
    confidence_distribution = {"high": 0, "medium": 0, "low": 0}
    for info in merged_state.synthesized_info.values():
        level = info.confidence_level.lower()
        if level in confidence_distribution:
            confidence_distribution[level] += 1

    # Calculate completeness ratio
    completeness_ratio = (
        len(processed_questions) / len(merged_state.sub_questions)
        if merged_state.sub_questions
        else 0
    )

    # Log metadata
    log_metadata(
        metadata={
            "merge_results": {
                "execution_time_seconds": execution_time,
                "total_sub_questions": len(merged_state.sub_questions),
                "parallel_steps_processed": parallel_steps_processed,
                "questions_successfully_merged": len(processed_questions),
                "missing_questions_count": len(missing_questions),
                "missing_questions": missing_questions[:5]
                if missing_questions
                else [],  # Limit to 5 for metadata
                "total_search_results": total_search_results,
                "confidence_distribution": confidence_distribution,
                "merge_success": bool(
                    merged_state.search_results
                    and merged_state.synthesized_info
                ),
                "total_search_costs": merged_state.search_costs,
                "total_search_queries": len(merged_state.search_cost_details),
                "total_exa_cost": merged_state.search_costs.get("exa", 0.0),
            }
        }
    )

    # Log model metadata for cross-pipeline tracking
    log_metadata(
        metadata={
            "research_quality": {
                "completeness_ratio": completeness_ratio,
            }
        },
        infer_model=True,
    )

    # Log artifact metadata
    log_metadata(
        metadata={
            "merged_state_characteristics": {
                "has_search_results": bool(merged_state.search_results),
                "has_synthesized_info": bool(merged_state.synthesized_info),
                "search_results_count": len(merged_state.search_results),
                "synthesized_info_count": len(merged_state.synthesized_info),
                "completeness_ratio": completeness_ratio,
            }
        },
        infer_artifact=True,
    )

    # Add tags to the artifact
    add_tags(tags=["state", "merged"], artifact="merged_state")

    return merged_state
