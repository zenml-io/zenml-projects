from steps.cross_viewpoint_step import cross_viewpoint_analysis_step
from steps.pydantic_final_report_step import pydantic_final_report_step
from steps.iterative_reflection_step import iterative_reflection_step
from steps.merge_results_step import merge_sub_question_results_step
from steps.process_sub_question_step import process_sub_question_step
from steps.query_decomposition_step import initial_query_decomposition_step
from utils.pydantic_models import ResearchState
from zenml import pipeline
from zenml.types import HTMLString


@pipeline(name="parallelized_deep_research_pipeline")
def parallelized_deep_research_pipeline(
    query: str = "What is ZenML?", max_sub_questions: int = 10
) -> HTMLString:
    """Parallelized ZenML pipeline for deep research on a given query.

    This pipeline uses the fan-out/fan-in pattern for parallel processing of sub-questions,
    potentially improving execution time when using distributed orchestrators.

    Args:
        query: The research query/topic
        max_sub_questions: Maximum number of sub-questions to process in parallel

    Returns:
        Formatted research report as HTML
    """
    # Initialize the research state with the main query
    state = ResearchState(main_query=query)

    # Step 1: Decompose the query into sub-questions, limiting to max_sub_questions
    decomposed_state = initial_query_decomposition_step(
        state=state, max_sub_questions=max_sub_questions
    )

    # Fan out: Process each sub-question in parallel
    # Collect artifacts to establish dependencies for the merge step
    after = []
    for i in range(max_sub_questions):
        # Process the i-th sub-question (if it exists)
        sub_state = process_sub_question_step(
            state=decomposed_state,
            question_index=i,
            id=f"process_question_{i}",
        )
        after.append(sub_state)

    # Fan in: Merge results from all parallel processing
    # The 'after' parameter ensures this step runs after all processing steps
    # It doesn't directly use the processed_states input
    merged_state = merge_sub_question_results_step(
        original_state=decomposed_state,
        step_prefix="process_question_",
        output_name="output",
        after=after,  # This creates the dependency
    )

    # Continue with subsequent steps
    analyzed_state = cross_viewpoint_analysis_step(state=merged_state)
    reflected_state = iterative_reflection_step(state=analyzed_state)
    
    # Use our new Pydantic-based final report step
    # This returns a tuple (state, html_report)
    _, final_report = pydantic_final_report_step(state=reflected_state)

    return final_report
