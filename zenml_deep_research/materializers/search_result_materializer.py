"""SearchResult materializer for the pydantic model.

This module provides a materializer for the SearchResult Pydantic model to ensure
proper serialization and deserialization of search results in the pipeline.
"""

from typing import Dict, Type, Any

from utils.pydantic_models import SearchResult
from zenml.enums import ArtifactType
from zenml.materializers import PydanticMaterializer


class SearchResultMaterializer(PydanticMaterializer):
    """Materializer for the SearchResult Pydantic class.
    
    This materializer ensures that the SearchResult Pydantic model can be
    properly serialized and deserialized when passed between pipeline steps.
    """
    
    ASSOCIATED_TYPES = (SearchResult,)
    ASSOCIATED_ARTIFACT_TYPE = ArtifactType.DATA