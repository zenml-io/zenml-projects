"""Utility functions for loading prompts into the PromptsBundle model.

This module provides functions to create PromptsBundle instances from
the existing prompt definitions in prompts.py.
"""

from utils import prompts
from utils.prompt_models import PromptsBundle, PromptTemplate


def load_prompts_bundle(pipeline_version: str = "1.1.0") -> PromptsBundle:
    """Load all prompts from prompts.py into a PromptsBundle.

    Args:
        pipeline_version: Version of the pipeline using these prompts

    Returns:
        PromptsBundle containing all prompts
    """
    # Create PromptTemplate instances for each prompt
    search_query_prompt = PromptTemplate(
        name="search_query_prompt",
        content=prompts.DEFAULT_SEARCH_QUERY_PROMPT,
        description="Generates effective search queries from sub-questions",
        version="1.0.0",
        tags=["search", "query", "information-gathering"],
    )

    query_decomposition_prompt = PromptTemplate(
        name="query_decomposition_prompt",
        content=prompts.QUERY_DECOMPOSITION_PROMPT,
        description="Breaks down complex research queries into specific sub-questions",
        version="1.0.0",
        tags=["analysis", "decomposition", "planning"],
    )

    synthesis_prompt = PromptTemplate(
        name="synthesis_prompt",
        content=prompts.SYNTHESIS_PROMPT,
        description="Synthesizes search results into comprehensive answers for sub-questions",
        version="1.0.0",
        tags=["synthesis", "integration", "analysis"],
    )

    viewpoint_analysis_prompt = PromptTemplate(
        name="viewpoint_analysis_prompt",
        content=prompts.VIEWPOINT_ANALYSIS_PROMPT,
        description="Analyzes synthesized answers across different perspectives and viewpoints",
        version="1.0.0",
        tags=["analysis", "viewpoint", "perspective"],
    )

    reflection_prompt = PromptTemplate(
        name="reflection_prompt",
        content=prompts.REFLECTION_PROMPT,
        description="Evaluates research and identifies gaps, biases, and areas for improvement",
        version="1.0.0",
        tags=["reflection", "critique", "improvement"],
    )

    additional_synthesis_prompt = PromptTemplate(
        name="additional_synthesis_prompt",
        content=prompts.ADDITIONAL_SYNTHESIS_PROMPT,
        description="Enhances original synthesis with new information and addresses critique points",
        version="1.0.0",
        tags=["synthesis", "enhancement", "integration"],
    )

    conclusion_generation_prompt = PromptTemplate(
        name="conclusion_generation_prompt",
        content=prompts.CONCLUSION_GENERATION_PROMPT,
        description="Synthesizes all research findings into a comprehensive conclusion",
        version="1.0.0",
        tags=["report", "conclusion", "synthesis"],
    )

    executive_summary_prompt = PromptTemplate(
        name="executive_summary_prompt",
        content=prompts.EXECUTIVE_SUMMARY_GENERATION_PROMPT,
        description="Creates a compelling, insight-driven executive summary",
        version="1.1.0",
        tags=["report", "summary", "insights"],
    )

    introduction_prompt = PromptTemplate(
        name="introduction_prompt",
        content=prompts.INTRODUCTION_GENERATION_PROMPT,
        description="Creates a contextual, engaging introduction",
        version="1.1.0",
        tags=["report", "introduction", "context"],
    )

    # Create and return the bundle
    return PromptsBundle(
        search_query_prompt=search_query_prompt,
        query_decomposition_prompt=query_decomposition_prompt,
        synthesis_prompt=synthesis_prompt,
        viewpoint_analysis_prompt=viewpoint_analysis_prompt,
        reflection_prompt=reflection_prompt,
        additional_synthesis_prompt=additional_synthesis_prompt,
        conclusion_generation_prompt=conclusion_generation_prompt,
        executive_summary_prompt=executive_summary_prompt,
        introduction_prompt=introduction_prompt,
        pipeline_version=pipeline_version,
    )


def get_prompt_for_step(bundle: PromptsBundle, step_name: str) -> str:
    """Get the appropriate prompt content for a specific step.

    Args:
        bundle: The PromptsBundle containing all prompts
        step_name: Name of the step requesting the prompt

    Returns:
        The prompt content string

    Raises:
        ValueError: If no prompt mapping exists for the step
    """
    # Map step names to prompt attributes
    step_to_prompt_mapping = {
        "query_decomposition": "query_decomposition_prompt",
        "search_query_generation": "search_query_prompt",
        "synthesis": "synthesis_prompt",
        "viewpoint_analysis": "viewpoint_analysis_prompt",
        "reflection": "reflection_prompt",
        "additional_synthesis": "additional_synthesis_prompt",
        "conclusion_generation": "conclusion_generation_prompt",
    }

    prompt_attr = step_to_prompt_mapping.get(step_name)
    if not prompt_attr:
        raise ValueError(f"No prompt mapping found for step: {step_name}")

    return bundle.get_prompt_content(prompt_attr)
