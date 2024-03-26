from typing import Annotated, List

from zenml import ArtifactConfig, step, log_artifact_metadata
from langchain.docstore.document import Document
from langchain_text_splitters import CharacterTextSplitter
from zenml.client import Client
from sentence_transformers import SentenceTransformer
import numpy as np
import psycopg2
from pgvector.psycopg2 import register_vector
import math

EMBEDDINGS_MODEL = "all-distilroberta-v1"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
EMBEDDING_DIMENSIONALITY = (
    768  # Update this to match the dimensionality of the new model
)


@step(enable_cache=False)
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


@step(enable_cache=False)
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
    pg_password = Client().get_secret("postgres_db").secret_values["password"]

    CONNECTION_DETAILS = {
        "user": "postgres.jjpynzoqhdifcfroyfon",
        "password": pg_password,
        "host": "aws-0-eu-central-1.pooler.supabase.com",
        "port": "5432",
        "dbname": "postgres",
    }

    conn = psycopg2.connect(**CONNECTION_DETAILS)
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

    # Prepare the list of tuples to insert
    data_list = []
    for i, doc in enumerate(documents):
        title = doc.metadata.get("title", "")
        url = doc.metadata.get("url", "")
        content = doc.page_content
        tokens = len(
            content.split()
        )  # Approximate token count based on word count
        embedding = embeddings[i].tolist()
        data_list.append((title, url, content, tokens, embedding))

    # Insert data only if it doesn't already exist
    cur.execute("SELECT COUNT(*) FROM embeddings")
    count = cur.fetchone()[0]
    if count == 0:
        psycopg2.extras.execute_values(
            cur,
            "INSERT INTO embeddings (title, url, content, tokens, embedding) VALUES %s",
            data_list,
            template="(%s, %s, %s, %s, %s)",
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
