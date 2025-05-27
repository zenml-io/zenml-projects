"""Step to initialize and track prompts as individual artifacts.

This step creates individual Prompt artifacts at the beginning of the pipeline,
making all prompts trackable and versioned in ZenML.
"""

import logging
from typing import Annotated, Tuple

from materializers.prompt_materializer import PromptMaterializer
from utils import prompts
from utils.pydantic_models import Prompt
from zenml import add_tags, step

logger = logging.getLogger(__name__)


@step(output_materializers=PromptMaterializer)
def initialize_prompts_step(
    pipeline_version: str = "1.1.0",
) -> Tuple[
    Annotated[Prompt, "search_query_prompt"],
    Annotated[Prompt, "query_decomposition_prompt"],
    Annotated[Prompt, "synthesis_prompt"],
    Annotated[Prompt, "viewpoint_analysis_prompt"],
    Annotated[Prompt, "reflection_prompt"],
    Annotated[Prompt, "additional_synthesis_prompt"],
    Annotated[Prompt, "conclusion_generation_prompt"],
    Annotated[Prompt, "executive_summary_prompt"],
    Annotated[Prompt, "introduction_prompt"],
]:
    """Initialize individual prompts for the pipeline.

    This step loads all prompts from the prompts.py module and creates
    individual Prompt artifacts that can be tracked and visualized in ZenML.

    Args:
        pipeline_version: Version of the pipeline using these prompts

    Returns:
        Tuple of individual Prompt artifacts used in the pipeline
    """
    logger.info(
        f"Initializing prompts for pipeline version {pipeline_version}"
    )

    # Create individual prompt instances
    search_query_prompt = Prompt(
        content=prompts.DEFAULT_SEARCH_QUERY_PROMPT,
        name="search_query_prompt",
        description="Generates effective search queries from sub-questions",
        version="1.0.0",
        tags=["search", "query", "information-gathering"],
    )

    query_decomposition_prompt = Prompt(
        content=prompts.QUERY_DECOMPOSITION_PROMPT,
        name="query_decomposition_prompt",
        description="Breaks down complex research queries into specific sub-questions",
        version="1.0.0",
        tags=["analysis", "decomposition", "planning"],
    )

    synthesis_prompt = Prompt(
        content=prompts.SYNTHESIS_PROMPT,
        name="synthesis_prompt",
        description="Synthesizes search results into comprehensive answers for sub-questions",
        version="1.1.0",
        tags=["synthesis", "integration", "analysis"],
    )

    viewpoint_analysis_prompt = Prompt(
        content=prompts.VIEWPOINT_ANALYSIS_PROMPT,
        name="viewpoint_analysis_prompt",
        description="Analyzes synthesized answers across different perspectives and viewpoints",
        version="1.1.0",
        tags=["analysis", "viewpoint", "perspective"],
    )

    reflection_prompt = Prompt(
        content=prompts.REFLECTION_PROMPT,
        name="reflection_prompt",
        description="Evaluates research and identifies gaps, biases, and areas for improvement",
        version="1.0.0",
        tags=["reflection", "critique", "improvement"],
    )

    additional_synthesis_prompt = Prompt(
        content=prompts.ADDITIONAL_SYNTHESIS_PROMPT,
        name="additional_synthesis_prompt",
        description="Enhances original synthesis with new information and addresses critique points",
        version="1.1.0",
        tags=["synthesis", "enhancement", "integration"],
    )

    conclusion_generation_prompt = Prompt(
        content=prompts.CONCLUSION_GENERATION_PROMPT,
        name="conclusion_generation_prompt",
        description="Synthesizes all research findings into a comprehensive conclusion",
        version="1.0.0",
        tags=["report", "conclusion", "synthesis"],
    )

    executive_summary_prompt = Prompt(
        content=prompts.EXECUTIVE_SUMMARY_GENERATION_PROMPT,
        name="executive_summary_prompt",
        description="Creates a compelling, insight-driven executive summary",
        version="1.1.0",
        tags=["report", "summary", "insights"],
    )

    introduction_prompt = Prompt(
        content=prompts.INTRODUCTION_GENERATION_PROMPT,
        name="introduction_prompt",
        description="Creates a contextual, engaging introduction",
        version="1.1.0",
        tags=["report", "introduction", "context"],
    )

    logger.info(f"Loaded 9 individual prompts")

    # add tags to all prompts
    add_tags(tags=["prompt", "search"], artifact="search_query_prompt")
    add_tags(
        tags=["prompt", "generation"], artifact="query_decomposition_prompt"
    )
    add_tags(tags=["prompt", "generation"], artifact="synthesis_prompt")
    add_tags(
        tags=["prompt", "generation"], artifact="viewpoint_analysis_prompt"
    )
    add_tags(tags=["prompt", "generation"], artifact="reflection_prompt")
    add_tags(
        tags=["prompt", "generation"], artifact="additional_synthesis_prompt"
    )
    add_tags(
        tags=["prompt", "generation"], artifact="conclusion_generation_prompt"
    )
    add_tags(
        tags=["prompt", "generation"], artifact="executive_summary_prompt"
    )
    add_tags(tags=["prompt", "generation"], artifact="introduction_prompt")

    return (
        search_query_prompt,
        query_decomposition_prompt,
        synthesis_prompt,
        viewpoint_analysis_prompt,
        reflection_prompt,
        additional_synthesis_prompt,
        conclusion_generation_prompt,
        executive_summary_prompt,
        introduction_prompt,
    )
