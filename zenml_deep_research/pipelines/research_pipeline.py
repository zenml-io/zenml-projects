from zenml import pipeline
from zenml.types import HTMLString

from utils.data_models import ResearchState
from steps.query_decomposition_step import initial_query_decomposition_step
from steps.information_gathering_step import (
    parallel_information_gathering_step,
)
from steps.information_synthesis_step import (
    information_validation_synthesis_step,
)
from steps.cross_viewpoint_step import cross_viewpoint_analysis_step
from steps.iterative_reflection_step import iterative_reflection_step
from steps.final_report_step import final_report_generation_step


@pipeline(name="enhanced_deep_research_pipeline")
def enhanced_deep_research_pipeline(
    query: str = "What is ZenML?",
) -> HTMLString:
    """Enhanced ZenML pipeline for deep research on a given query with more granular steps.

    Args:
        query: The research query/topic

    Returns:
        Formatted research report as HTML
    """
    # Initialize the research state with the main query
    state = ResearchState(main_query=query)

    # Step 1: Decompose the query into sub-questions
    state = initial_query_decomposition_step(state=state)

    # Step 2: Gather information for each sub-question
    state = parallel_information_gathering_step(state=state)

    # Step 3: Validate and synthesize the gathered information
    state = information_validation_synthesis_step(state=state)

    # Step 4: Analyze different perspectives
    state = cross_viewpoint_analysis_step(state=state)

    # Step 5: Perform iterative reflection and enhancement
    state = iterative_reflection_step(state=state)

    # Step 6: Generate the final report
    final_report = final_report_generation_step(state=state)

    return final_report
