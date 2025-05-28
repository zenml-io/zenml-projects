import logging
import time
from typing import Annotated, Tuple

from materializers.search_data_materializer import SearchDataMaterializer
from materializers.synthesis_data_materializer import SynthesisDataMaterializer
from utils.pydantic_models import SearchData, SynthesisData
from zenml import get_step_context, log_metadata, step
from zenml.client import Client

logger = logging.getLogger(__name__)


@step(
    output_materializers={
        "merged_search_data": SearchDataMaterializer,
        "merged_synthesis_data": SynthesisDataMaterializer,
    }
)
def merge_sub_question_results_step(
    step_prefix: str = "process_question_",
) -> Tuple[
    Annotated[SearchData, "merged_search_data"],
    Annotated[SynthesisData, "merged_synthesis_data"],
]:
    """Merge results from individual sub-question processing steps.

    This step collects the results from the parallel sub-question processing steps
    and combines them into single SearchData and SynthesisData artifacts.

    Args:
        step_prefix: The prefix used in step IDs for the parallel processing steps

    Returns:
        Tuple of merged SearchData and SynthesisData artifacts

    Note:
        This step is typically configured with the 'after' parameter in the pipeline
        definition to ensure it runs after all parallel sub-question processing steps
        have completed.
    """
    start_time = time.time()

    # Initialize merged artifacts
    merged_search_data = SearchData()
    merged_synthesis_data = SynthesisData()

    # Get pipeline run information to access outputs
    try:
        ctx = get_step_context()
        if not ctx or not ctx.pipeline_run:
            logger.error("Could not get pipeline run context")
            return merged_search_data, merged_synthesis_data

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

                        # Get the search_data artifact
                        if "search_data" in step_info.outputs:
                            search_artifacts = step_info.outputs["search_data"]
                            if search_artifacts:
                                search_artifact = search_artifacts[0]
                                sub_search_data = search_artifact.load()

                                # Merge search data
                                merged_search_data.merge(sub_search_data)

                                # Track processed questions
                                for sub_q in sub_search_data.search_results:
                                    processed_questions.add(sub_q)
                                    logger.info(
                                        f"Merged search results for: {sub_q}"
                                    )

                        # Get the synthesis_data artifact
                        if "synthesis_data" in step_info.outputs:
                            synthesis_artifacts = step_info.outputs[
                                "synthesis_data"
                            ]
                            if synthesis_artifacts:
                                synthesis_artifact = synthesis_artifacts[0]
                                sub_synthesis_data = synthesis_artifact.load()

                                # Merge synthesis data
                                merged_synthesis_data.merge(sub_synthesis_data)

                                # Track processed questions
                                for (
                                    sub_q
                                ) in sub_synthesis_data.synthesized_info:
                                    logger.info(
                                        f"Merged synthesis info for: {sub_q}"
                                    )

                        parallel_steps_processed += 1

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
        if merged_search_data.search_costs:
            total_cost = sum(merged_search_data.search_costs.values())
            logger.info(
                f"Total search costs merged: ${total_cost:.4f} across {len(merged_search_data.search_cost_details)} queries"
            )
            for provider, cost in merged_search_data.search_costs.items():
                logger.info(f"  {provider}: ${cost:.4f}")

    except Exception as e:
        logger.error(f"Error during merge step: {e}")

    # Final check for empty results
    if (
        not merged_search_data.search_results
        or not merged_synthesis_data.synthesized_info
    ):
        logger.warning(
            "No results were found or merged from parallel processing steps!"
        )

    # Calculate execution time
    execution_time = time.time() - start_time

    # Count total search results across all questions
    total_search_results = sum(
        len(results) for results in merged_search_data.search_results.values()
    )

    # Get confidence distribution for merged results
    confidence_distribution = {"high": 0, "medium": 0, "low": 0}
    for info in merged_synthesis_data.synthesized_info.values():
        level = info.confidence_level.lower()
        if level in confidence_distribution:
            confidence_distribution[level] += 1

    # Log metadata
    log_metadata(
        metadata={
            "merge_results": {
                "execution_time_seconds": execution_time,
                "parallel_steps_processed": parallel_steps_processed,
                "questions_successfully_merged": len(processed_questions),
                "total_search_results": total_search_results,
                "confidence_distribution": confidence_distribution,
                "merge_success": bool(
                    merged_search_data.search_results
                    and merged_synthesis_data.synthesized_info
                ),
                "total_search_costs": merged_search_data.search_costs,
                "total_search_queries": len(
                    merged_search_data.search_cost_details
                ),
                "total_exa_cost": merged_search_data.search_costs.get(
                    "exa", 0.0
                ),
            }
        }
    )

    # Log artifact metadata
    log_metadata(
        metadata={
            "search_data_characteristics": {
                "total_searches": merged_search_data.total_searches,
                "search_results_count": len(merged_search_data.search_results),
                "total_cost": sum(merged_search_data.search_costs.values()),
            }
        },
        artifact_name="merged_search_data",
        infer_artifact=True,
    )

    log_metadata(
        metadata={
            "synthesis_data_characteristics": {
                "synthesized_info_count": len(
                    merged_synthesis_data.synthesized_info
                ),
                "enhanced_info_count": len(
                    merged_synthesis_data.enhanced_info
                ),
                "confidence_distribution": confidence_distribution,
            }
        },
        artifact_name="merged_synthesis_data",
        infer_artifact=True,
    )

    # Add tags to the artifacts
    # add_tags(
    #     tags=["search", "merged"],
    #     artifact_name="merged_search_data",
    #     infer_artifact=True,
    # )
    # add_tags(
    #     tags=["synthesis", "merged"],
    #     artifact_name="merged_synthesis_data",
    #     infer_artifact=True,
    # )

    return merged_search_data, merged_synthesis_data
