import os
from typing import Type

from langchain_community.vectorstores.faiss import FAISS
from langchain_core.vectorstores.base import VectorStore
from langchain_openai import OpenAIEmbeddings
from zenml.client import Client
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
        # First try to get API key from environment variable
        api_key = os.getenv("OPENAI_API_KEY")

        # If not found in env, fall back to ZenML secret
        if not api_key:
            secret = Client().get_secret("llm_complete")
            api_key = secret.secret_values["openai_api_key"]

        embeddings = OpenAIEmbeddings(openai_api_key=api_key)

        # Load from disk
        return FAISS.load_local(
            self.uri,
            embeddings=embeddings,
            allow_dangerous_deserialization=True,
        )
