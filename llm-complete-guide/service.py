from typing import AsyncGenerator

import bentoml
import litellm
import numpy as np
from constants import (
    EMBEDDINGS_MODEL,
    MODEL_NAME_MAP,
    OPENAI_MODEL,
    SECRET_NAME,
)
from psycopg2.extensions import connection
from rerankers import Reranker
from sentence_transformers import SentenceTransformer
from utils.llm_utils import get_db_conn, get_topn_similar_docs
from utils.openai_utils import get_openai_api_key
from zenml.client import Client


@bentoml.service(
    name="rag-service",
    traffic={
        "timeout": 300,
        "concurrency": 256,
    },
    http={
        "cors": {
            "enabled": True,
            "access_control_allow_origins": [
                "https://cloud.zenml.io"
            ],  # Add your allowed origins
            "access_control_allow_methods": [
                "GET",
                "OPTIONS",
                "POST",
                "HEAD",
                "PUT",
            ],
            "access_control_allow_credentials": True,
            "access_control_allow_headers": ["*"],
            "access_control_max_age": 1200,
            "access_control_expose_headers": ["Content-Length"],
        }
    },
)
class RAGService:
    """RAG service for generating responses using LLM and RAG."""

    def __init__(self):
        """Initialize the RAG service."""
        # Initialize embeddings model
        self.embeddings_model = SentenceTransformer(EMBEDDINGS_MODEL)

        # Initialize reranker
        self.reranker = Reranker("flashrank")

        # Initialize database connection
        self.db_conn = get_db_conn()

    def get_embeddings(self, text: str) -> np.ndarray:
        """Get embeddings for the given text."""
        embeddings = self.embeddings_model.encode(text)
        if embeddings.ndim == 2:
            embeddings = embeddings[0]
        return embeddings

    def get_similar_docs(
        self, query_embedding: np.ndarray, n: int = 20
    ) -> list:
        """Get similar documents for the given query embedding."""
        if query_embedding.ndim == 2:
            query_embedding = query_embedding[0]

        docs = get_topn_similar_docs(
            query_embedding=query_embedding.tolist(),
            conn=self.db_conn,
            n=n,
            include_metadata=True
        )

        return [
            {
                "content": doc[0],
                "url": doc[1],
                "parent_section": doc[2],
            }
            for doc in docs
        ]

    def rerank_documents(self, query: str, documents: list) -> list:
        """Rerank documents using the reranker."""
        docs_texts = [
            f"{doc['content']} PARENT SECTION: {doc['parent_section']}"
            for doc in documents
        ]
        results = self.reranker.rank(query=query, docs=docs_texts)

        reranked_docs = []
        for result in results.results:
            index_val = result.doc_id
            doc = documents[index_val]
            reranked_docs.append((result.text, doc["url"]))
        return reranked_docs[:5]

    async def get_completion(
        self, messages: list, model: str, temperature: float, max_tokens: int
    ) -> AsyncGenerator[str, None]:
        """Handle the completion request and streaming response."""
        try:
            response = await litellm.acompletion(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=get_openai_api_key(),
                stream=True,
            )

            async for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            yield f"Error in completion: {str(e)}"

    @bentoml.api
    async def generate(
        self,
        query: str = "Explain ZenML features",
        temperature: float = 0.4,
        max_tokens: int = 1000,
    ) -> AsyncGenerator[str, None]:
        """Generate responses for the given query."""
        try:
            # Get embeddings for query
            query_embedding = self.get_embeddings(query)

            # Retrieve similar documents
            similar_docs = self.get_similar_docs(query_embedding, n=20)

            # Rerank documents
            reranked_docs = self.rerank_documents(query, similar_docs)

            # Prepare context from reranked documents
            context = "\n\n".join([doc[0] for doc in reranked_docs])

            # Prepare system message
            system_message = """
            You are a friendly chatbot. \
            You can answer questions about ZenML, its features and its use cases. \
            You respond in a concise, technically credible tone. \
            You ONLY use the context from the ZenML documentation to provide relevant answers. \
            You do not make up answers or provide opinions that you don't have information to support. \
            If you are unsure or don't know, just say so. \
            """

            # Prepare messages for LLM
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": query},
                {
                    "role": "assistant",
                    "content": f"Please use the following relevant ZenML documentation to answer the query: \n{context}",
                },
            ]

            # Get completion from LLM using the new async method
            model = MODEL_NAME_MAP.get(OPENAI_MODEL, OPENAI_MODEL)
            async for chunk in self.get_completion(
                messages, model, temperature, max_tokens
            ):
                yield chunk

        except Exception as e:
            yield f"Error occurred: {str(e)}"
