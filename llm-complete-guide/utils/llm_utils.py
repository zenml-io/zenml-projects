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

from zenml.cli import secret

# Configure logging levels for specific modules
logging.getLogger("pytorch").setLevel(logging.CRITICAL)
logging.getLogger("sentence-transformers").setLevel(logging.CRITICAL)
logging.getLogger("rerankers").setLevel(logging.CRITICAL)
logging.getLogger("transformers").setLevel(logging.CRITICAL)

# Configure the logging level for the root logger
logging.getLogger().setLevel(logging.ERROR)

import os
import re
from typing import Dict, List, Tuple

import litellm
import numpy as np
import psycopg2
import tiktoken
from constants import EMBEDDINGS_MODEL, MODEL_NAME_MAP, OPENAI_MODEL
from pgvector.psycopg2 import register_vector
from psycopg2.extensions import connection
from rerankers import Reranker
from sentence_transformers import SentenceTransformer
from structures import Document

logger = logging.getLogger(__name__)


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


def get_db_password(secret_name: str) -> str:
    """Returns the password for the PostgreSQL database.

    Returns:
        str: The password for the PostgreSQL database.
    """
    password = os.getenv("ZENML_POSTGRES_DB_PASSWORD")
    if not password:
        from zenml.client import Client

        password = (
            Client()
            .get_secret(secret_name)
            .secret_values["password"]
        )
    return password


def get_db_user(secret_name: str) -> str:
    """Returns the user for the PostgreSQL database.

    Returns:
        str: The user for the PostgreSQL database.
    """
    user = os.getenv("ZENML_POSTGRES_USER")
    if not user:
        from zenml.client import Client

        user = (
            Client()
            .get_secret(secret_name)
            .secret_values["user"]
        )
    return user


def get_db_host(secret_name: str) -> str:
    """Returns the host for the PostgreSQL database.

    Returns:
        str: The host for the PostgreSQL database.
    """
    host = os.getenv("ZENML_POSTGRES_HOST")
    if not host:
        from zenml.client import Client

        host = (
            Client()
            .get_secret(secret_name)
            .secret_values["host"]
        )
    return host


def get_db_port(secret_name: str) -> str:
    """Returns the port for the PostgreSQL database.

    Returns:
        str: The port for the PostgreSQL database.
    """
    port = os.getenv("ZENML_POSTGRES_DB_PASSWORD")
    if not port:
        from zenml.client import Client

        port = (
            Client()
            .get_secret("supabase_postgres_db")
            .secret_values["port"]
        )
    return port


def get_db_conn() -> connection:
    """Establishes and returns a connection to the PostgreSQL database.

    This function retrieves the password for the PostgreSQL database from a secret store,
    then uses it along with other connection details to establish a connection.

    Returns:
        connection: A psycopg2 connection object to the PostgreSQL database.
    """
    secret_name = os.getenv("ZENML_SUPABASE_SECRET_NAME")

    if not secret_name:
        raise RuntimeError(
            "Please make sure to set the environment variable: ZENML_SUPABASE_SECRET_NAME to point at the secret that "
            "contains your supabase connection details."
        )

    CONNECTION_DETAILS = {
        "user": get_db_user(secret_name),
        "password": get_db_password(secret_name),
        "host": get_db_host(secret_name),
        "port": get_db_port(secret_name),
        "dbname": "postgres",
    }

    return psycopg2.connect(**CONNECTION_DETAILS)


def get_topn_similar_docs(
    query_embedding: List[float],
    conn: psycopg2.extensions.connection,
    n: int = 5,
    include_metadata: bool = False,
    only_urls: bool = False,
) -> List[Tuple]:
    """Fetches the top n most similar documents to the given query embedding from the database.

    Args:
        query_embedding (list): The query embedding to compare against.
        conn (psycopg2.extensions.connection): The database connection object.
        n (int, optional): The number of similar documents to fetch. Defaults to
        5.
        include_metadata (bool, optional): Whether to include metadata in the
        results. Defaults to False.

    Returns:
        list: A list of tuples containing the content and metadata (if include_metadata is True) of the top n most similar documents.
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


def get_completion_from_messages(
    messages, model=OPENAI_MODEL, temperature=0.4, max_tokens=1000
):
    """Generates a completion response from the given messages using the specified model.

    Args:
        messages (list): The list of messages to generate a completion from.
        model (str, optional): The model to use for generating the completion. Defaults to OPENAI_MODEL.
        temperature (float, optional): The temperature to use for the completion. Defaults to 0.4.
        max_tokens (int, optional): The maximum number of tokens to generate. Defaults to 1000.

    Returns:
        str: The content of the completion response.
    """
    model = MODEL_NAME_MAP.get(model, model)
    completion_response = litellm.completion(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
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
            the reranked documents and their URLs.
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

    Returns:
        str: The processed output.
    """
    delimiter = "```"

    # Step 1: Get documents related to the user input from database
    related_docs = get_topn_similar_docs(
        get_embeddings(input),
        get_db_conn(),
        n=n_items_retrieved,
        include_metadata=use_reranking,
    )

    if use_reranking:
        # Rerank the documents based on the input
        # and take the top 5 only
        context_content = [
            doc[0] for doc in rerank_documents(input, related_docs)[:5]
        ]
    else:
        context_content = [doc[0] for doc in related_docs[:5]]

    # Step 2: Get completion from OpenAI API
    # Set system message to help set appropriate tone and context for model
    system_message = f"""
    You are a friendly chatbot. \
    You can answer questions about ZenML, its features and its use cases. \
    You respond in a concise, technically credible tone. \
    You ONLY use the context from the ZenML documentation to provide relevant
    answers. \
    You do not make up answers or provide opinions that you don't have
    information to support. \
    If you are unsure or don't know, just say so. \
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
    return get_completion_from_messages(messages, model=model)
