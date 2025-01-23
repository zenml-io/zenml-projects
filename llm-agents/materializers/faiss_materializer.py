from typing import Any, Type

from langchain_openai import OpenAIEmbeddings
import numpy as np
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.vectorstores.base import VectorStore
from zenml.materializers.base_materializer import BaseMaterializer


class FAISSMaterializer(BaseMaterializer):
    """Materializer for FAISS vector stores."""

    ASSOCIATED_TYPES = (FAISS, VectorStore)

    def save(self, data: FAISS) -> None:
        """Save the FAISS index and documents.
        
        Args:
            data: The FAISS vector store to save
        """
        # Save the index to disk
        data.save_local(self.uri)

    def load(self, data_type: Type[FAISS]) -> FAISS:
        """Load the FAISS index and documents.
        
        Returns:
            The loaded FAISS vector store
        """
        embeddings = OpenAIEmbeddings()
        # Load from disk
        return FAISS.load_local(self.uri, embeddings=embeddings, allow_dangerous_deserialization=True) 