# credit to langchain for the original base implementation of splitting
# functionality
# https://github.com/langchain-ai/langchain/blob/master/libs/text-splitters/langchain_text_splitters/character.py


import logging
import re
from typing import List

import litellm
import numpy as np
import psycopg2
from constants import EMBEDDINGS_MODEL, OPENAI_MODEL
from pgvector.psycopg2 import register_vector
from psycopg2.extensions import connection
from sentence_transformers import SentenceTransformer
from zenml.client import Client

# Configure the logging level for the root logger
logging.getLogger().setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


def split_text_with_regex(text: str, separator: str, keep_separator: bool) -> List[str]:
    if separator:
        if keep_separator:
            _splits = re.split(f"({separator})", text)
            splits = [_splits[i] + _splits[i + 1] for i in range(1, len(_splits), 2)]
            if len(_splits) % 2 == 0:
                splits += _splits[-1:]
            splits = [_splits[0]] + splits
        else:
            splits = re.split(separator, text)
    else:
        splits = list(text)
    return [s for s in splits if s != ""]


def split_text(
    text: str,
    separator: str = "\n\n",
    chunk_size: int = 4000,
    chunk_overlap: int = 200,
    keep_separator: bool = False,
    strip_whitespace: bool = True,
) -> List[str]:
    if chunk_overlap > chunk_size:
        raise ValueError(
            f"Got a larger chunk overlap ({chunk_overlap}) than chunk size "
            f"({chunk_size}), should be smaller."
        )

    separator_regex = re.escape(separator)
    splits = split_text_with_regex(text, separator_regex, keep_separator)
    _separator = "" if keep_separator else separator

    chunks = []
    current_chunk = ""

    for split in splits:
        if strip_whitespace:
            split = split.strip()

        if len(current_chunk) + len(split) + len(_separator) <= chunk_size:
            current_chunk += split + _separator
        else:
            if current_chunk:
                chunks.append(current_chunk.rstrip(_separator))
            current_chunk = split + _separator

    if current_chunk:
        chunks.append(current_chunk.rstrip(_separator))

    final_chunks = []
    for i in range(len(chunks)):
        if i == 0:
            final_chunks.append(chunks[i])
        else:
            overlap = chunks[i - 1][-chunk_overlap:]
            final_chunks.append(overlap + chunks[i])

    return final_chunks


def split_documents(
    documents: List[str],
    separator: str = "\n\n",
    chunk_size: int = 4000,
    chunk_overlap: int = 200,
    keep_separator: bool = False,
    strip_whitespace: bool = True,
) -> List[str]:
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


def get_db_conn() -> connection:
    pg_password = Client().get_secret("postgres_db").secret_values["password"]

    CONNECTION_DETAILS = {
        "user": "postgres.jjpynzoqhdifcfroyfon",
        "password": pg_password,
        "host": "aws-0-eu-central-1.pooler.supabase.com",
        "port": "5432",
        "dbname": "postgres",
    }

    return psycopg2.connect(**CONNECTION_DETAILS)


def get_topn_similar_docs(query_embedding, conn, n: int = 5):
    embedding_array = np.array(query_embedding)
    register_vector(conn)
    cur = conn.cursor()
    # Get the top n most similar documents using the KNN <=> operator
    cur.execute(
        f"SELECT content FROM embeddings ORDER BY embedding <=> %s LIMIT {n}",
        (embedding_array,),
    )
    return cur.fetchall()


def get_completion_from_messages(
    messages, model=OPENAI_MODEL, temperature=0.4, max_tokens=1000
):
    if model == "gpt4":
        model = "gpt-4-0125-preview"
    elif model == "gpt35":
        model = "gpt-3.5-turbo"
    elif model == "claude3":
        model = "claude-3-opus-20240229"
    elif model == "claudehaiku":
        model = "claude-3-haiku-20240307"
    completion_response = litellm.completion(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return completion_response.choices[0].message.content


# Helper function: get embeddings for a text
def get_embeddings(text):
    model = SentenceTransformer(EMBEDDINGS_MODEL)
    return model.encode(text)


def process_input_with_retrieval(input: str, model: str = OPENAI_MODEL) -> str:
    """Process the input with retrieval.

    Args:
        input (str): The input to process.

    Returns:
        str: The processed output.
    """
    delimiter = "```"

    # Step 1: Get documents related to the user input from database
    related_docs = get_topn_similar_docs(get_embeddings(input), get_db_conn())

    # Step 2: Get completion from OpenAI API
    # Set system message to help set appropriate tone and context for model
    system_message = f"""
    You are a friendly chatbot. \
    You can answer questions about ZenML, its features and its use cases. \
    You respond in a concise, technically credible tone. \
    You ONLY use the context from the ZenML documentation to provide relevant
    answers. \
    You do not make up answers or provide opinions that you don't have information to support. \
    """

    # Prepare messages to pass to model
    # We use a delimiter to help the model understand the where the user_input
    # starts and ends

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": f"{delimiter}{input}{delimiter}"},
        {
            "role": "assistant",
            "content": f"Relevant ZenML documentation: \n"
            + "\n".join(doc[0] for doc in related_docs),
        },
    ]
    logger.debug("CONTEXT USED\n\n", messages[2]["content"], "\n\n")
    return get_completion_from_messages(messages, model=model)
