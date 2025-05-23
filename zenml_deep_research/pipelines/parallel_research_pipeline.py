from steps.approval_step import get_research_approval_step
from steps.cross_viewpoint_step import cross_viewpoint_analysis_step
from steps.execute_approved_searches_step import execute_approved_searches_step
from steps.generate_reflection_step import generate_reflection_step
from steps.initialize_prompts_step import initialize_prompts_step
from steps.merge_results_step import merge_sub_question_results_step
from steps.process_sub_question_step import process_sub_question_step
from steps.pydantic_final_report_step import pydantic_final_report_step
from steps.query_decomposition_step import initial_query_decomposition_step
from utils.pydantic_models import ResearchState
from zenml import pipeline
from zenml.types import HTMLString


@pipeline(enable_cache=False)
def parallelized_deep_research_pipeline(
    query: str = "What is ZenML?",
    max_sub_questions: int = 10,
    require_approval: bool = False,
    approval_timeout: int = 3600,
    max_additional_searches: int = 2,
    search_provider: str = "tavily",
    search_mode: str = "auto",
    num_results_per_search: int = 3,
) -> HTMLString:
    """Parallelized ZenML pipeline for deep research on a given query.

    This pipeline uses the fan-out/fan-in pattern for parallel processing of sub-questions,
    potentially improving execution time when using distributed orchestrators.

    Args:
        query: The research query/topic
        max_sub_questions: Maximum number of sub-questions to process in parallel
        require_approval: Whether to require human approval for additional searches
        approval_timeout: Timeout in seconds for human approval
        max_additional_searches: Maximum number of additional searches to perform
        search_provider: Search provider to use (tavily, exa, or both)
        search_mode: Search mode for Exa provider (neural, keyword, or auto)
        num_results_per_search: Number of search results to return per query

    Returns:
        Formatted research report as HTML
    """
    # Initialize prompts bundle for tracking
    prompts_bundle = initialize_prompts_step(pipeline_version="1.0.0")

    # Initialize the research state with the main query
    state = ResearchState(main_query=query)

    # Step 1: Decompose the query into sub-questions, limiting to max_sub_questions
    decomposed_state = initial_query_decomposition_step(
        state=state,
        prompts_bundle=prompts_bundle,
        max_sub_questions=max_sub_questions,
    )

    # Fan out: Process each sub-question in parallel
    # Collect artifacts to establish dependencies for the merge step
    after = []
    for i in range(max_sub_questions):
        # Process the i-th sub-question (if it exists)
        sub_state = process_sub_question_step(
            state=decomposed_state,
            prompts_bundle=prompts_bundle,
            question_index=i,
            search_provider=search_provider,
            search_mode=search_mode,
            num_results_per_search=num_results_per_search,
            id=f"process_question_{i + 1}",
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
    analyzed_state = cross_viewpoint_analysis_step(
        state=merged_state, prompts_bundle=prompts_bundle
    )

    # New 3-step reflection flow with optional human approval
    # Step 1: Generate reflection and recommendations (no searches yet)
    reflection_output = generate_reflection_step(
        state=analyzed_state, prompts_bundle=prompts_bundle
    )

    # Step 2: Get approval for recommended searches
    approval_decision = get_research_approval_step(
        reflection_output=reflection_output,
        require_approval=require_approval,
        timeout=approval_timeout,
        max_queries=max_additional_searches,
    )

    # Step 3: Execute approved searches (if any)
    reflected_state = execute_approved_searches_step(
        reflection_output=reflection_output,
        approval_decision=approval_decision,
        prompts_bundle=prompts_bundle,
        search_provider=search_provider,
        search_mode=search_mode,
        num_results_per_search=num_results_per_search,
    )

    # Use our new Pydantic-based final report step
    # This returns a tuple (state, html_report)
    _, final_report = pydantic_final_report_step(
        state=reflected_state, prompts_bundle=prompts_bundle
    )

    return final_report
