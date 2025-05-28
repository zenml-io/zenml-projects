"""
Pipelines package for the ZenML Deep Research project.

This package contains the ZenML pipeline definitions for running deep research
workflows. Each pipeline orchestrates a sequence of steps for comprehensive
research on a given query topic.
"""

from .parallel_research_pipeline import parallelized_deep_research_pipeline

__all__ = ["parallelized_deep_research_pipeline"]
