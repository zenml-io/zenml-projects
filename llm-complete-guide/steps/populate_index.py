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

# credit to
# https://www.timescale.com/blog/postgresql-as-a-vector-database-create-store-and-query-openai-embeddings-with-pgvector/
# for providing the base implementation for this indexing functionality

import logging
import math
from typing import Annotated, List

from constants import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_DIMENSIONALITY,
    EMBEDDINGS_MODEL,
)
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer
from structures import Document
from utils.llm_utils import get_db_conn, split_documents
from zenml import ArtifactConfig, log_artifact_metadata, step

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@step
def preprocess_documents(
    documents: List[Document],
) -> Annotated[List[Document], ArtifactConfig(name="split_chunks")]:
    """
    Preprocesses a list of documents by splitting them into chunks.

    Args:
        documents (List[Document]): A list of documents to be preprocessed.

    Returns:
        Annotated[List[Document], ArtifactConfig(name="split_chunks")]: A list of preprocessed documents annotated with an ArtifactConfig.

    Raises:
        Exception: If an error occurs during preprocessing.
    """
    try:
        log_artifact_metadata(
            artifact_name="split_chunks",
            metadata={
                "chunk_size": CHUNK_SIZE,
                "chunk_overlap": CHUNK_OVERLAP,
            },
        )

        split_docs = split_documents(
            documents, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )
        return split_docs
    except Exception as e:
        logger.error(f"Error in preprocess_documents: {e}")
        raise


@step(enable_cache=False)
def generate_embeddings(
    split_documents: List[Document],
) -> Annotated[
    List[Document], ArtifactConfig(name="documents_with_embeddings")
]:
    """
    Generates embeddings for a list of split documents using a SentenceTransformer model.

    Args:
        split_documents (List[Document]): A list of Document objects that have been split into chunks.

    Returns:
        Annotated[List[Document], ArtifactConfig(name="embeddings")]: The list of Document objects with generated embeddings, annotated with an ArtifactConfig.

    Raises:
        Exception: If an error occurs during the generation of embeddings.
    """
    try:
        model = SentenceTransformer(EMBEDDINGS_MODEL)

        log_artifact_metadata(
            artifact_name="embeddings",
            metadata={
                "embedding_type": EMBEDDINGS_MODEL,
                "embedding_dimensionality": EMBEDDING_DIMENSIONALITY,
            },
        )

        document_texts = [doc.page_content for doc in split_documents]
        embeddings = model.encode(document_texts)

        for doc, embedding in zip(split_documents, embeddings):
            doc.embedding = embedding
        return split_documents
    except Exception as e:
        logger.error(f"Error in generate_embeddings: {e}")
        raise


@step(enable_cache=False)
def index_generator(
    documents: List[Document],
) -> None:
    """
    Generates an index for the given documents.

    This function creates a database connection, installs the pgvector extension if not already installed,
    creates an embeddings table if it doesn't exist, and inserts the embeddings and document metadata into the table.
    It then calculates the index parameters according to best practices and creates an index on the embeddings
    using the cosine distance measure.

    Args:
        documents (List[Document]): The list of Document objects with generated embeddings.

    Raises:
        Exception: If an error occurs during the index generation.
    """
    try:
        conn = get_db_conn()
        with conn.cursor() as cur:
            # Install pgvector if not already installed
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            conn.commit()

            # Create the embeddings table if it doesn't exist
            table_create_command = f"""
            CREATE TABLE IF NOT EXISTS embeddings (
                        id SERIAL PRIMARY KEY,
                        content TEXT,
                        token_count INTEGER,
                        embedding VECTOR({EMBEDDING_DIMENSIONALITY}),
                        filename TEXT,
                        parent_section TEXT,
                        url TEXT
                        );
                        """
            cur.execute(table_create_command)
            conn.commit()

            register_vector(conn)

            # Insert data only if it doesn't already exist
            for doc in documents:
                content = doc.page_content
                token_count = doc.token_count
                embedding = doc.embedding.tolist()
                filename = doc.filename
                parent_section = doc.parent_section
                url = doc.url

                cur.execute(
                    "SELECT COUNT(*) FROM embeddings WHERE content = %s",
                    (content,),
                )
                count = cur.fetchone()[0]
                if count == 0:
                    cur.execute(
                        "INSERT INTO embeddings (content, token_count, embedding, filename, parent_section, url) VALUES (%s, %s, %s, %s, %s, %s)",
                        (
                            content,
                            token_count,
                            embedding,
                            filename,
                            parent_section,
                            url,
                        ),
                    )
                    conn.commit()

            cur.execute("SELECT COUNT(*) as cnt FROM embeddings;")
            num_records = cur.fetchone()[0]
            logger.info(f"Number of vector records in table: {num_records}")

            # calculate the index parameters according to best practices
            num_lists = max(num_records / 1000, 10)
            if num_records > 1000000:
                num_lists = math.sqrt(num_records)

            # use the cosine distance measure, which is what we'll later use for querying
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS embeddings_idx ON embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = {num_lists});"
            )
            conn.commit()

    except Exception as e:
        logger.error(f"Error in index_generator: {e}")
        raise
    finally:
        if conn:
            conn.close()
