import copy
import logging
from typing import Annotated

from materializers.pydantic_materializer import ResearchStateMaterializer
from utils.pydantic_models import ResearchState
from zenml import get_step_context, step
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
    # Start with the original state that has all sub-questions
    merged_state = copy.deepcopy(original_state)

    # Initialize empty dictionaries for the results
    merged_state.search_results = {}
    merged_state.synthesized_info = {}

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

    return merged_state
