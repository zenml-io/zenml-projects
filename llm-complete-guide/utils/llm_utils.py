import logging

import numpy as np
import psycopg2
from constants import EMBEDDINGS_MODEL, OPENAI_MODEL
from openai import OpenAI
from pgvector.psycopg2 import register_vector
from psycopg2.extensions import connection
from sentence_transformers import SentenceTransformer
from zenml.client import Client

openai_client = OpenAI()

logger = logging.getLogger(__name__)


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
    completion = openai_client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return completion.choices[0].message.content


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
