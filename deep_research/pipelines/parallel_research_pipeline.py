from steps.approval_step import get_research_approval_step
from steps.collect_tracing_metadata_step import collect_tracing_metadata_step
from steps.cross_viewpoint_step import cross_viewpoint_analysis_step
from steps.execute_approved_searches_step import execute_approved_searches_step
from steps.generate_reflection_step import generate_reflection_step
from steps.initialize_prompts_step import initialize_prompts_step
from steps.merge_results_step import merge_sub_question_results_step
from steps.process_sub_question_step import process_sub_question_step
from steps.pydantic_final_report_step import pydantic_final_report_step
from steps.query_decomposition_step import initial_query_decomposition_step
from zenml import pipeline


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
    langfuse_project_name: str = "deep-research",
) -> None:
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
        langfuse_project_name: Langfuse project name for LLM tracking

    Returns:
        Formatted research report as HTML
    """
    # Initialize individual prompts for tracking
    (
        search_query_prompt,
        query_decomposition_prompt,
        synthesis_prompt,
        viewpoint_analysis_prompt,
        reflection_prompt,
        additional_synthesis_prompt,
        conclusion_generation_prompt,
        executive_summary_prompt,
        introduction_prompt,
    ) = initialize_prompts_step(pipeline_version="1.0.0")

    # Step 1: Decompose the query into sub-questions, limiting to max_sub_questions
    query_context = initial_query_decomposition_step(
        main_query=query,
        query_decomposition_prompt=query_decomposition_prompt,
        max_sub_questions=max_sub_questions,
        langfuse_project_name=langfuse_project_name,
    )

    # Fan out: Process each sub-question in parallel
    # Collect step names to establish dependencies for the merge step
    parallel_step_names = []
    for i in range(max_sub_questions):
        # Process the i-th sub-question (if it exists)
        step_name = f"process_question_{i + 1}"
        search_data, synthesis_data = process_sub_question_step(
            query_context=query_context,
            search_query_prompt=search_query_prompt,
            synthesis_prompt=synthesis_prompt,
            question_index=i,
            search_provider=search_provider,
            search_mode=search_mode,
            num_results_per_search=num_results_per_search,
            langfuse_project_name=langfuse_project_name,
            id=step_name,
            after="initial_query_decomposition_step",
        )
        parallel_step_names.append(step_name)

    # Fan in: Merge results from all parallel processing
    # The 'after' parameter ensures this step runs after all processing steps
    merged_search_data, merged_synthesis_data = (
        merge_sub_question_results_step(
            step_prefix="process_question_",
            after=parallel_step_names,  # Wait for all parallel steps to complete
        )
    )

    # Continue with subsequent steps
    analysis_data = cross_viewpoint_analysis_step(
        query_context=query_context,
        synthesis_data=merged_synthesis_data,
        viewpoint_analysis_prompt=viewpoint_analysis_prompt,
        langfuse_project_name=langfuse_project_name,
        after="merge_sub_question_results_step",
    )

    # New 3-step reflection flow with optional human approval
    # Step 1: Generate reflection and recommendations (no searches yet)
    analysis_with_reflection, recommended_queries = generate_reflection_step(
        query_context=query_context,
        synthesis_data=merged_synthesis_data,
        analysis_data=analysis_data,
        reflection_prompt=reflection_prompt,
        langfuse_project_name=langfuse_project_name,
        after="cross_viewpoint_analysis_step",
    )

    # Step 2: Get approval for recommended searches
    approval_decision = get_research_approval_step(
        query_context=query_context,
        synthesis_data=merged_synthesis_data,
        analysis_data=analysis_with_reflection,
        recommended_queries=recommended_queries,
        require_approval=require_approval,
        timeout=approval_timeout,
        max_queries=max_additional_searches,
        after="generate_reflection_step",
    )

    # Step 3: Execute approved searches (if any)
    enhanced_search_data, enhanced_synthesis_data, enhanced_analysis_data = (
        execute_approved_searches_step(
            query_context=query_context,
            search_data=merged_search_data,
            synthesis_data=merged_synthesis_data,
            analysis_data=analysis_with_reflection,
            recommended_queries=recommended_queries,
            approval_decision=approval_decision,
            additional_synthesis_prompt=additional_synthesis_prompt,
            search_provider=search_provider,
            search_mode=search_mode,
            num_results_per_search=num_results_per_search,
            langfuse_project_name=langfuse_project_name,
            after="get_research_approval_step",
        )
    )

    # Use our new Pydantic-based final report step
    pydantic_final_report_step(
        query_context=query_context,
        search_data=enhanced_search_data,
        synthesis_data=enhanced_synthesis_data,
        analysis_data=enhanced_analysis_data,
        conclusion_generation_prompt=conclusion_generation_prompt,
        executive_summary_prompt=executive_summary_prompt,
        introduction_prompt=introduction_prompt,
        langfuse_project_name=langfuse_project_name,
        after="execute_approved_searches_step",
    )

    # Collect tracing metadata for the entire pipeline run
    collect_tracing_metadata_step(
        query_context=query_context,
        search_data=enhanced_search_data,
        langfuse_project_name=langfuse_project_name,
        after="pydantic_final_report_step",
    )
