from zenml import pipeline
from zenml.types import HTMLString

from steps.report_structure_step import report_structure_step
from steps.paragraph_research_step import paragraph_research_step
from steps.report_formatting_step import report_formatting_step

@pipeline(name="deep_research_pipeline")
def deep_research_pipeline(
    query: str
) -> HTMLString:
    """ZenML pipeline for deep research on a given query.
    
    Args:
        query: The research query/topic
        
    Returns:
        Formatted research report as HTML
    """
    # Step 1: Generate report structure
    initial_state = report_structure_step(query=query)
    
    # Step 2: Research each paragraph
    researched_state = paragraph_research_step(current_state=initial_state)
    
    # Step 3: Format the final report
    final_report = report_formatting_step(final_state=researched_state)
    
    return final_report 