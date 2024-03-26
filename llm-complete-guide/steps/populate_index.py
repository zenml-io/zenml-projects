import math
from typing import Annotated, List

import numpy as np
from constants import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_DIMENSIONALITY,
    EMBEDDINGS_MODEL,
)
from langchain.docstore.document import Document
from langchain_text_splitters import CharacterTextSplitter
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer
from utils.llm_utils import get_db_conn
from zenml import ArtifactConfig, log_artifact_metadata, step


@step
def preprocess_documents(
    documents: List[Document],
) -> Annotated[List[Document], ArtifactConfig(name="split_document_chunks")]:
    log_artifact_metadata(
        artifact_name="split_document_chunks",
        metadata={
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
        },
    )
    text_splitter = CharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    return text_splitter.split_documents(documents)


@step
def generate_embeddings(
    split_documents: List[Document],
) -> Annotated[np.ndarray, ArtifactConfig(name="embeddings")]:
    model = SentenceTransformer(EMBEDDINGS_MODEL)

    log_artifact_metadata(
        artifact_name="embeddings",
        metadata={
            "embedding_type": EMBEDDINGS_MODEL,
            "embedding_dimensionality": EMBEDDING_DIMENSIONALITY,
        },
    )
    raw_texts = [doc.page_content for doc in split_documents]
    return model.encode(raw_texts)


@step(enable_cache=False)
def index_generator(
    embeddings: np.ndarray,
    documents: List[Document],
) -> None:
    conn = get_db_conn()
    cur = conn.cursor()

    # Install pgvector if not already installed
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
    conn.commit()

    # Create the embeddings table if it doesn't exist
    table_create_command = f"""
    CREATE TABLE IF NOT EXISTS embeddings (
                id SERIAL PRIMARY KEY,
                title TEXT,
                url TEXT,
                content TEXT,
                tokens INTEGER,
                embedding VECTOR({EMBEDDING_DIMENSIONALITY})
                );
                """
    cur.execute(table_create_command)
    conn.commit()

    register_vector(conn)

    # Insert data only if it doesn't already exist
    for i, doc in enumerate(documents):
        title = doc.metadata.get("title", "")
        url = doc.metadata.get("url", "")
        content = doc.page_content
        tokens = len(
            content.split()
        )  # Approximate token count based on word count
        embedding = embeddings[i].tolist()

        cur.execute(
            "SELECT COUNT(*) FROM embeddings WHERE content = %s", (content,)
        )
        count = cur.fetchone()[0]
        if count == 0:
            cur.execute(
                "INSERT INTO embeddings (title, url, content, tokens, embedding) VALUES (%s, %s, %s, %s, %s)",
                (title, url, content, tokens, embedding),
            )
            conn.commit()

    cur.execute("SELECT COUNT(*) as cnt FROM embeddings;")
    num_records = cur.fetchone()[0]
    print("Number of vector records in table: ", num_records, "\n")

    # calculate the index parameters according to best practices
    num_lists = num_records / 1000
    num_lists = max(num_lists, 10)
    if num_records > 1000000:
        num_lists = math.sqrt(num_records)

    # use the cosine distance measure, which is what we'll later use for querying
    cur.execute(
        f"CREATE INDEX ON embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = {num_lists});"
    )
    conn.commit()

    cur.close()
    conn.close()
