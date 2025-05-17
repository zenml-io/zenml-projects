# Apache Software License 2.0
#
# Copyright (c) ZenML GmbH 2024. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# credit to langchain for the original base implementation of splitting
# functionality
# https://github.com/langchain-ai/langchain/blob/master/libs/text-splitters/langchain_text_splitters/character.py

import logging
import os

import pinecone
from elasticsearch import Elasticsearch
from pinecone import Pinecone, ServerlessSpec
from zenml.client import Client

from utils.openai_utils import get_openai_api_key

# Configure logging levels for specific modules
logging.getLogger("pytorch").setLevel(logging.CRITICAL)
logging.getLogger("sentence-transformers").setLevel(logging.CRITICAL)
logging.getLogger("rerankers").setLevel(logging.CRITICAL)
logging.getLogger("transformers").setLevel(logging.CRITICAL)

# Configure the logging level for the root logger
logging.getLogger().setLevel(logging.ERROR)

import re
from typing import List, Optional, Tuple

import litellm
import numpy as np
import psycopg2
import tiktoken
from constants import (
    DEFAULT_PROMPT,
    EMBEDDING_DIMENSIONALITY,
    EMBEDDINGS_MODEL,
    MODEL_NAME_MAP,
    OPENAI_MODEL,
    SECRET_NAME,
    SECRET_NAME_ELASTICSEARCH,
    ZENML_CHATBOT_MODEL_NAME,
    ZENML_CHATBOT_MODEL_VERSION,
)
from pgvector.psycopg2 import register_vector
from psycopg2.extensions import connection
from rerankers import Reranker
from sentence_transformers import SentenceTransformer
from structures import Document

logger = logging.getLogger(__name__)

# First try to get from environment variables
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST")

# If any are not set, get from ZenML secrets and set the env vars
if not all([LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST]):
    secret = Client().get_secret(SECRET_NAME)

    if not LANGFUSE_PUBLIC_KEY:
        LANGFUSE_PUBLIC_KEY = secret.secret_values.get("langfuse_public_key")
        if LANGFUSE_PUBLIC_KEY:
            os.environ["LANGFUSE_PUBLIC_KEY"] = LANGFUSE_PUBLIC_KEY

    if not LANGFUSE_SECRET_KEY:
        LANGFUSE_SECRET_KEY = secret.secret_values.get("langfuse_secret_key")
        if LANGFUSE_SECRET_KEY:
            os.environ["LANGFUSE_SECRET_KEY"] = LANGFUSE_SECRET_KEY

    if not LANGFUSE_HOST:
        LANGFUSE_HOST = secret.secret_values.get("langfuse_host")
        if LANGFUSE_HOST:
            os.environ["LANGFUSE_HOST"] = LANGFUSE_HOST

# logs all litellm requests to langfuse
litellm.callbacks = ["langfuse"]


def split_text_with_regex(
    text: str, separator: str, keep_separator: bool
) -> List[str]:
    """Splits a given text using a specified separator.

    This function splits the input text using the provided separator. The separator can be included or excluded
    from the resulting splits based on the value of keep_separator.

    Args:
        text (str): The text to be split.
        separator (str): The separator to use for splitting the text.
        keep_separator (bool): If True, the separator is kept in the resulting splits. If False, the separator is removed.

    Returns:
        List[str]: A list of strings resulting from splitting the input text.
    """
    if separator:
        if keep_separator:
            _splits = re.split(f"({separator})", text)
            splits = [
                _splits[i] + _splits[i + 1] for i in range(1, len(_splits), 2)
            ]
            if len(_splits) % 2 == 0:
                splits += _splits[-1:]
            splits = [_splits[0]] + splits
        else:
            splits = re.split(separator, text)
    else:
        splits = list(text)
    return [s for s in splits if s != ""]


def split_text(
    document: Document,
    separator: str = "\n\n",
    chunk_size: int = 4000,
    chunk_overlap: int = 200,
    keep_separator: bool = False,
    strip_whitespace: bool = True,
) -> List[Document]:
    """Splits a given text into chunks of specified size with optional overlap.

    Args:
        document (Document): The document to be split.
        separator (str, optional): The separator to use for splitting the text. Defaults to "\n\n".
        chunk_size (int, optional): The maximum size of each chunk. Defaults to 4000.
        chunk_overlap (int, optional): The size of the overlap between consecutive chunks. Defaults to 200.
        keep_separator (bool, optional): If True, the separator is kept in the resulting splits. If False, the separator is removed. Defaults to False.
        strip_whitespace (bool, optional): If True, leading and trailing whitespace is removed from each split. Defaults to True.

    Raises:
        ValueError: If chunk_overlap is larger than chunk_size.

    Returns:
        List[Document]: A list of documents resulting from splitting the input document into chunks.
    """
    if chunk_overlap > chunk_size:
        raise ValueError(
            f"Got a larger chunk overlap ({chunk_overlap}) than chunk size "
            f"({chunk_size}), should be smaller."
        )

    separator_regex = re.escape(separator)
    splits = split_text_with_regex(
        document.page_content, separator_regex, keep_separator
    )
    _separator = "" if keep_separator else separator

    encoding = tiktoken.get_encoding("cl100k_base")
    chunks = []
    current_chunk = ""

    for split in splits:
        if strip_whitespace:
            split = split.strip()

        if len(current_chunk) + len(split) + len(_separator) <= chunk_size:
            current_chunk += split + _separator
        else:
            if current_chunk:
                token_count = len(
                    encoding.encode(current_chunk.rstrip(_separator))
                )
                chunks.append(
                    Document(
                        page_content=current_chunk.rstrip(_separator),
                        filename=document.filename,
                        parent_section=document.parent_section,
                        url=document.url,
                        token_count=token_count,
                    )
                )
            current_chunk = split + _separator

    if current_chunk:
        token_count = len(encoding.encode(current_chunk.rstrip(_separator)))
        chunks.append(
            Document(
                page_content=current_chunk.rstrip(_separator),
                filename=document.filename,
                parent_section=document.parent_section,
                url=document.url,
                token_count=token_count,
            )
        )

    final_chunks = []
    for i in range(len(chunks)):
        if i == 0:
            final_chunks.append(chunks[i])
        else:
            overlap = chunks[i - 1].page_content[-chunk_overlap:]
            token_count = len(
                encoding.encode(overlap + chunks[i].page_content)
            )
            final_chunks.append(
                Document(
                    page_content=overlap + chunks[i].page_content,
                    filename=document.filename,
                    parent_section=document.parent_section,
                    url=document.url,
                    token_count=token_count,
                )
            )

    return final_chunks


def split_documents(
    documents: List[Document],
    separator: str = "\n\n",
    chunk_size: int = 4000,
    chunk_overlap: int = 200,
    keep_separator: bool = False,
    strip_whitespace: bool = True,
) -> List[Document]:
    """Splits a list of documents into chunks.

    Args:
        documents (List[str]): The list of documents to be split.
        separator (str, optional): The separator to use for splitting the documents. Defaults to "\n\n".
        chunk_size (int, optional): The maximum size of each chunk. Defaults to 4000.
        chunk_overlap (int, optional): The size of the overlap between consecutive chunks. Defaults to 200.
        keep_separator (bool, optional): If True, the separator is kept in the resulting splits. If False, the separator is removed. Defaults to False.
        strip_whitespace (bool, optional): If True, leading and trailing whitespace is removed from each split. Defaults to True.

    Returns:
        List[str]: A list of chunked documents.
    """
    chunked_documents = []
    for doc in documents:
        chunked_documents.extend(
            split_text(
                doc,
                separator=separator,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                keep_separator=keep_separator,
                strip_whitespace=strip_whitespace,
            )
        )
    return chunked_documents


def get_es_client() -> Elasticsearch:
    """Get an Elasticsearch client.

    Returns:
        Elasticsearch: An Elasticsearch client.
    """
    client = Client()
    es_host = client.get_secret(SECRET_NAME_ELASTICSEARCH).secret_values[
        "elasticsearch_host"
    ]
    es_api_key = client.get_secret(SECRET_NAME_ELASTICSEARCH).secret_values[
        "elasticsearch_api_key"
    ]

    es = Elasticsearch(
        es_host,
        api_key=es_api_key,
    )
    return es


def get_db_conn() -> connection:
    """Establishes and returns a connection to the PostgreSQL database.

    This function retrieves the password for the PostgreSQL database from a secret store,
    then uses it along with other connection details to establish a connection.

    Returns:
        connection: A psycopg2 connection object to the PostgreSQL database.
    """
    client = Client()
    try:
        secret = client.get_secret(SECRET_NAME)
        logger.debug(f"Secret keys: {list(secret.secret_values.keys())}")

        CONNECTION_DETAILS = {
            "user": os.getenv("SUPABASE_USER")
            or secret.secret_values["supabase_user"],
            "password": os.getenv("SUPABASE_PASSWORD")
            or secret.secret_values["supabase_password"],
            "host": os.getenv("SUPABASE_HOST")
            or secret.secret_values["supabase_host"],
            "port": os.getenv("SUPABASE_PORT")
            or secret.secret_values["supabase_port"],
            "dbname": "postgres",
        }
        return psycopg2.connect(**CONNECTION_DETAILS)
    except KeyError as e:
        logger.error(f"Missing key in secret: {e}")
        raise


def get_pinecone_client() -> pinecone.Index:
    """Get a Pinecone index client.

    Returns:
        pinecone.Index: A Pinecone index client.
    """
    client = Client()
    pinecone_api_key = (
        os.getenv("PINECONE_API_KEY")
        or client.get_secret(SECRET_NAME).secret_values["pinecone_api_key"]
    )
    pc = Pinecone(api_key=pinecone_api_key)

    # if the model version is staging, we check if any index name is associated as metadata
    # if not, create a new one with the name from the secret and attach it to the metadata
    # if the model version is production, we just use the index name from the metadata attached to it
    # raise error if there is no index name attached to the metadata
    model_version = client.get_model_version(
        model_name_or_id=ZENML_CHATBOT_MODEL_NAME,
    )

    index_name = model_version.name
    index_name = index_name.replace(".", "-")
    if "vector_store" not in model_version.run_metadata:
        model_version.run_metadata["vector_store"] = {}
    model_version.run_metadata["vector_store"]["index_name"] = index_name

    # if not exists, create index
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=EMBEDDING_DIMENSIONALITY,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

    print(f"Pinecone index being used: {index_name}")

    return pc.Index(index_name)


def get_topn_similar_docs_pgvector(
    query_embedding: List[float],
    conn: psycopg2.extensions.connection,
    n: int = 5,
    include_metadata: bool = False,
    only_urls: bool = False,
) -> List[Tuple]:
    """Fetches the top n most similar documents to the given query embedding from the PostgreSQL database.

    Args:
        query_embedding (list): The query embedding to compare against.
        conn (psycopg2.extensions.connection): The database connection object.
        n (int, optional): The number of similar documents to fetch. Defaults to 5.
        include_metadata (bool, optional): Whether to include metadata in the results. Defaults to False.
        only_urls (bool, optional): Whether to only return URLs in the results. Defaults to False.
    """
    embedding_array = np.array(query_embedding)
    register_vector(conn)
    cur = conn.cursor()

    if include_metadata:
        cur.execute(
            f"SELECT content, url, parent_section FROM embeddings ORDER BY embedding <=> %s LIMIT {n}",
            (embedding_array,),
        )
    elif only_urls:
        cur.execute(
            f"SELECT url FROM embeddings ORDER BY embedding <=> %s LIMIT {n}",
            (embedding_array,),
        )
    else:
        cur.execute(
            f"SELECT content FROM embeddings ORDER BY embedding <=> %s LIMIT {n}",
            (embedding_array,),
        )

    return cur.fetchall()


def get_topn_similar_docs_elasticsearch(
    query_embedding: List[float],
    es_client: Elasticsearch,
    n: int = 5,
    include_metadata: bool = False,
    only_urls: bool = False,
) -> List[Tuple]:
    """Fetches the top n most similar documents to the given query embedding from the Elasticsearch index.

    Args:
        query_embedding (list): The query embedding to compare against.
        es_client (Elasticsearch): The Elasticsearch client.
        n (int, optional): The number of similar documents to fetch. Defaults to 5.
        include_metadata (bool, optional): Whether to include metadata in the results. Defaults to False.
        only_urls (bool, optional): Whether to only return URLs in the results. Defaults to False.
    """
    index_name = "zenml_docs"

    response = es_client.search(
        index=index_name,
        knn={
            "field": "embedding",
            "query_vector": query_embedding,
            "num_candidates": 50,
            "k": n,
        },
    )

    results = []
    for hit in response["hits"]["hits"]:
        if only_urls:
            results.append((hit["_source"]["url"],))
        elif include_metadata:
            results.append(
                (
                    hit["_source"]["content"],
                    hit["_source"]["url"],
                    hit["_source"]["parent_section"],
                )
            )
        else:
            results.append((hit["_source"]["content"],))

    return results


def get_topn_similar_docs_pinecone(
    query_embedding: List[float],
    pinecone_index: pinecone.Index,
    n: int = 5,
    include_metadata: bool = False,
    only_urls: bool = False,
) -> List[Tuple]:
    """Get the top N most similar documents from Pinecone.

    Args:
        query_embedding (List[float]): The query embedding vector.
        pinecone_index (pinecone.Index): The Pinecone index client.
        n (int, optional): Number of similar documents to return. Defaults to 5.
        include_metadata (bool, optional): Whether to include metadata in results. Defaults to False.
        only_urls (bool, optional): Whether to return only URLs. Defaults to False.

    Returns:
        List[Tuple]: List of tuples containing the content and metadata (if include_metadata is True)
            of the top n most similar documents.
    """
    # Convert numpy array to list if needed
    if isinstance(query_embedding, np.ndarray):
        query_embedding = query_embedding.tolist()

    # Query the index
    results = pinecone_index.query(
        vector=query_embedding, top_k=n, include_metadata=True
    )

    # Process results
    similar_docs = []
    for match in results.matches:
        metadata = match.metadata

        if only_urls:
            similar_docs.append((metadata["url"],))
        elif include_metadata:
            similar_docs.append(
                (
                    metadata["page_content"],
                    metadata["url"],
                    metadata["parent_section"],
                )
            )
        else:
            similar_docs.append((metadata["page_content"],))

    return similar_docs


def get_topn_similar_docs(
    query_embedding: List[float],
    conn: Optional[psycopg2.extensions.connection] = None,
    es_client: Optional[Elasticsearch] = None,
    pinecone_index: Optional[pinecone.Index] = None,
    n: int = 5,
    include_metadata: bool = False,
    only_urls: bool = False,
) -> List[Tuple]:
    """Get the top N most similar documents from the vector store.

    Args:
        query_embedding (List[float]): The query embedding vector.
        conn (Optional[psycopg2.extensions.connection], optional): PostgreSQL connection. Defaults to None.
        es_client (Optional[Elasticsearch], optional): Elasticsearch client. Defaults to None.
        pinecone_index (Optional[pinecone.Index], optional): Pinecone index client. Defaults to None.
        n (int, optional): Number of similar documents to return. Defaults to 5.
        include_metadata (bool, optional): Whether to include metadata in results. Defaults to False.
        only_urls (bool, optional): Whether to return only URLs. Defaults to False.

    Returns:
        List[Tuple]: List of tuples containing the content and metadata (if include_metadata is True)
            of the top n most similar documents.

    Raises:
        ValueError: If no valid vector store client is provided.
    """
    if es_client is not None:
        return get_topn_similar_docs_elasticsearch(
            query_embedding, es_client, n, include_metadata, only_urls
        )
    elif conn is not None:
        return get_topn_similar_docs_pgvector(
            query_embedding, conn, n, include_metadata, only_urls
        )
    elif pinecone_index is not None:
        return get_topn_similar_docs_pinecone(
            query_embedding, pinecone_index, n, include_metadata, only_urls
        )
    else:
        raise ValueError("No valid vector store client provided")


def get_completion_from_messages(
    messages,
    model=OPENAI_MODEL,
    temperature=0,
    max_tokens=1000,
    tracing_tags: List[str] = [],
) -> str:
    """Generates a completion response from the given messages using the specified model.

    Args:
        messages (list): The list of messages to generate a completion from.
        model (str, optional): The model to use for generating the completion. Defaults to OPENAI_MODEL.
        temperature (float, optional): The temperature to use for the completion. Defaults to 0.4.
        max_tokens (int, optional): The maximum number of tokens to generate.
            Defaults to 1000.
        tracing_tags (List[str], optional): The tags to use for tracing the completion.
            Defaults to an empty list.

    Returns:
        str: The content of the completion response.
    """
    model = MODEL_NAME_MAP.get(model, model)
    completion_response = litellm.completion(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=get_openai_api_key(),
        metadata={
            "project": "llm-complete-guide-rag",
            "tags": tracing_tags,
        },
    )
    return completion_response.choices[0].message.content


def get_embeddings(text):
    """Generates embeddings for the given text using a SentenceTransformer model.

    Args:
        text (str): The text to generate embeddings for.

    Returns:
        np.ndarray: The generated embeddings.
    """
    model = SentenceTransformer(EMBEDDINGS_MODEL)
    return model.encode(text)


def find_vectorstore_name() -> str:
    """Finds the name of the vector store used for the given embeddings model."""
    from zenml.client import Client

    client = Client()
    try:
        model_version = client.get_model_version(
            model_name_or_id=ZENML_CHATBOT_MODEL_NAME,
            model_version_name_or_number_or_id=ZENML_CHATBOT_MODEL_VERSION,
        )
        return model_version.run_metadata["vector_store"]["name"]
    except KeyError:
        logger.error("Vector store metadata not found in model version")
        return "pinecone"  # Fallback to default


def rerank_documents(
    query: str, documents: List[Tuple], reranker_model: str = "flashrank"
) -> List[Tuple[str, str]]:
    """Reranks the given documents based on the given query.

    Args:
        query (str): The query to use for reranking.
        documents (List[Tuple]): The documents to rerank.
        reranker_model (str, optional): The reranker model to use.
            Defaults to "flashrank".

    Returns:
        List[Tuple[str, str]]: A list of tuples containing
            the reranked documents  and their URLs.
    """
    ranker = Reranker(reranker_model)
    docs_texts = [f"{doc[0]} PARENT SECTION: {doc[2]}" for doc in documents]
    results = ranker.rank(query=query, docs=docs_texts)
    # pair the texts with the original urls in `documents`
    # `documents` is a tuple of (content, url)
    # we want the urls to be returned
    reranked_documents_and_urls = []
    for result in results.results:
        # content is a `rerankers` Result object
        index_val = result.doc_id
        doc_text = result.text
        doc_url = documents[index_val][1]
        reranked_documents_and_urls.append((doc_text, doc_url))
    return reranked_documents_and_urls


def process_input_with_retrieval(
    input: str,
    model: str = OPENAI_MODEL,
    n_items_retrieved: int = 20,
    use_reranking: bool = False,
    tracing_tags: List[str] = [],
    prompt: str = DEFAULT_PROMPT,
) -> str:
    """Process the input with retrieval.

    Args:
        input (str): The input to process.
        model (str, optional): The model to use for completion. Defaults to
            OPENAI_MODEL.
        n_items_retrieved (int, optional): The number of items to retrieve from
            the database. Defaults to 5.
        use_reranking (bool, optional): Whether to use reranking. Defaults to
            False.
        model_version_stage (str, optional): The stage of the model version. Defaults to "staging".
        prompt (str, optional): The prompt to use for the retrieval. Defaults to None.
    Returns:
        str: The processed output.
    """
    delimiter = "```"
    # Get embeddings for the query
    query_embedding = get_embeddings(input)

    # Get similar documents based on the vector store being used
    vector_store = find_vectorstore_name()
    if vector_store == "elasticsearch":
        es_client = get_es_client()
        similar_docs = get_topn_similar_docs(
            query_embedding=query_embedding,
            es_client=es_client,
            n=n_items_retrieved,
            include_metadata=True,
        )
    elif vector_store == "pinecone":
        pinecone_index = get_pinecone_client()
        similar_docs = get_topn_similar_docs(
            query_embedding=query_embedding,
            pinecone_index=pinecone_index,
            n=n_items_retrieved,
            include_metadata=True,
        )
    else:  # pgvector
        conn = get_db_conn()
        similar_docs = get_topn_similar_docs(
            query_embedding=query_embedding,
            conn=conn,
            n=n_items_retrieved,
            include_metadata=True,
        )
        conn.close()

    # Rerank documents if enabled
    if use_reranking:
        # Rerank the documents based on the input
        # and take the top 5 only
        context_content = [
            doc[0] for doc in rerank_documents(input, similar_docs)[:5]
        ]
    else:
        context_content = [doc[0] for doc in similar_docs[:5]]

    # Step 2: Get completion from OpenAI API
    # Set system message to help set appropriate tone and context for model
    system_message = f"""
    {prompt}
    """

    # Prepare messages to pass to model
    # We use a delimiter to help the model understand the where the user_input
    # starts and ends

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": f"{delimiter}{input}{delimiter}"},
        {
            "role": "assistant",
            "content": f"Please use the input query and the following relevant ZenML documentation (in order of usefulness for this query) to answer the user query: \n"
            + "\n".join(context_content),
        },
    ]
    logger.debug("CONTEXT USED\n\n", messages[2]["content"], "\n\n")
    return get_completion_from_messages(
        messages,
        model=model,
        tracing_tags=tracing_tags,
    )
