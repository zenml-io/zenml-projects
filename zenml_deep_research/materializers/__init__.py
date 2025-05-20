"""
Materializers package for the ZenML Deep Research project.

This package contains custom ZenML materializers that handle serialization and
deserialization of complex data types used in the research pipeline, particularly
the ResearchState object that tracks the state of the research process.
"""

from .research_state_materializer import ResearchStateMaterializer

__all__ = ["ResearchStateMaterializer"]
