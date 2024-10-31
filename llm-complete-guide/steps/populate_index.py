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

import json
import logging
import math
from typing import Annotated, Dict, List, Tuple

from constants import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_DIMENSIONALITY,
    EMBEDDINGS_MODEL,
)
from pgvector.psycopg2 import register_vector
from PIL import Image, ImageDraw, ImageFont
from sentence_transformers import SentenceTransformer
from structures import Document
from utils.llm_utils import get_db_conn, split_documents
from zenml import ArtifactConfig, log_artifact_metadata, step

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_docs_stats(
    total_documents: int, split_docs: List[Document]
) -> Dict[str, Dict[str, int]]:
    """Extracts statistics about the document chunks.

    Args:
        total_documents (int): The total number of original documents before splitting.
        split_docs (List[Document]): The list of document chunks after splitting.

    Returns:
        Dict[str, Dict[str, int]]: A dictionary containing two sub-dictionaries:
            - document_stats: Contains statistics about the chunks including:
                - total_documents: Number of original documents
                - total_chunks: Number of chunks after splitting
                - avg_chunk_size: Average size of chunks in characters
                - min_chunk_size: Size of smallest chunk in characters
                - max_chunk_size: Size of largest chunk in characters
            - chunks_per_section: Maps each document section to number of chunks it contains
    """
    total_documents = total_documents
    total_chunks = len(split_docs)
    chunk_sizes = [len(doc.page_content) for doc in split_docs]
    avg_chunk_size = sum(chunk_sizes) / len(chunk_sizes)
    min_chunk_size = min(chunk_sizes)
    max_chunk_size = max(chunk_sizes)
    chunks_per_section = {}
    for doc in split_docs:
        section = doc.parent_section
        if section not in chunks_per_section:
            chunks_per_section[section] = 0
        chunks_per_section[section] += 1

    return {
        "document_stats": {
            "total_documents": total_documents,
            "total_chunks": total_chunks,
            "avg_chunk_size": avg_chunk_size,
            "min_chunk_size": min_chunk_size,
            "max_chunk_size": max_chunk_size,
        },
        "chunks_per_section": chunks_per_section,
    }


def create_charts(stats: Dict[str, Dict[str, int]]) -> Image.Image:
    """Creates a combined image containing a histogram of chunk sizes and a bar chart of chunk counts per section.

    Args:
        stats (Dict[str, Dict[str, int]]): A dictionary containing the extracted statistics.

    Returns:
        Image.Image: A combined image containing the histogram and bar chart.
    """
    document_stats = stats["document_stats"]
    chunks_per_section = stats["chunks_per_section"]

    # Create a new image with a white background
    image_width = 800
    image_height = 600
    image = Image.new("RGB", (image_width, image_height), color="white")
    draw = ImageDraw.Draw(image)

    # Draw the histogram of chunk sizes
    histogram_width = 600
    histogram_height = 250
    histogram_data = [
        document_stats["min_chunk_size"],
        document_stats["avg_chunk_size"],
        document_stats["max_chunk_size"],
    ]
    histogram_labels = ["Min", "Avg", "Max"]
    histogram_x = (image_width - histogram_width) // 2
    histogram_y = 50
    draw_histogram(
        draw,
        histogram_x,
        histogram_y,
        histogram_width,
        histogram_height,
        histogram_data,
        histogram_labels,
    )

    # Draw the bar chart of chunk counts per section
    bar_chart_width = 600
    bar_chart_height = 250
    bar_chart_data = list(chunks_per_section.values())
    bar_chart_labels = list(chunks_per_section.keys())
    bar_chart_x = (image_width - bar_chart_width) // 2
    bar_chart_y = histogram_y + histogram_height + 50
    draw_bar_chart(
        draw,
        bar_chart_x,
        bar_chart_y,
        bar_chart_width,
        bar_chart_height,
        bar_chart_data,
        bar_chart_labels,
    )

    # Add a title to the combined image
    title_text = "Document Chunk Statistics"
    title_font = ImageFont.load_default(size=24)
    title_bbox = draw.textbbox((0, 0), title_text, font=title_font)
    title_width = title_bbox[2] - title_bbox[0]
    title_height = title_bbox[3] - title_bbox[1]
    title_x = (image_width - title_width) // 2
    title_y = 10
    draw.text((title_x, title_y), title_text, font=title_font, fill="black")

    return image


def draw_histogram(
    draw: ImageDraw.Draw,
    x: int,
    y: int,
    width: int,
    height: int,
    data: List[int],
    labels: List[str],
) -> None:
    """Draws a histogram chart showing the distribution of chunk sizes.

    Args:
        draw (ImageDraw.Draw): The ImageDraw object to draw on
        x (int): The x coordinate of the top-left corner of the histogram
        y (int): The y coordinate of the top-left corner of the histogram
        width (int): The width of the histogram in pixels
        height (int): The height of the histogram in pixels
        data (List[int]): The values to plot in the histogram
        labels (List[str]): The labels for each bar in the histogram
    """
    # Calculate the maximum value in the data
    max_value = max(data)

    # Calculate the bar width and spacing
    bar_width = width // len(data)
    bar_spacing = 10

    # Draw the bars
    for i, value in enumerate(data):
        bar_height = (value / max_value) * height
        bar_x = x + i * (bar_width + bar_spacing)
        bar_y = y + height - bar_height
        draw.rectangle(
            [(bar_x, bar_y), (bar_x + bar_width, y + height)], fill="blue"
        )

        # Draw the label below the bar
        label_text = labels[i]
        label_font = ImageFont.load_default(size=12)
        label_bbox = draw.textbbox((0, 0), label_text, font=label_font)
        label_width = label_bbox[2] - label_bbox[0]
        label_height = label_bbox[3] - label_bbox[1]
        label_x = bar_x + (bar_width - label_width) // 2
        label_y = y + height + 5
        draw.text((label_x, label_y), label_text, font=label_font, fill="black")

    # Draw the title above the histogram
    title_text = "Chunk Size Distribution"
    title_font = ImageFont.load_default(size=16)
    title_bbox = draw.textbbox((0, 0), title_text, font=title_font)
    title_width = title_bbox[2] - title_bbox[0]
    title_height = title_bbox[3] - title_bbox[1]
    title_x = x + (width - title_width) // 2
    title_y = y - title_height - 10
    draw.text((title_x, title_y), title_text, font=title_font, fill="black")


def draw_bar_chart(
    draw: ImageDraw.Draw,
    x: int,
    y: int,
    width: int,
    height: int,
    data: List[int],
    labels: List[str],
) -> None:
    """Draws a bar chart showing the number of chunks per section.

    Args:
        draw (ImageDraw.Draw): The ImageDraw object to draw on
        x (int): The x coordinate of the top-left corner of the bar chart
        y (int): The y coordinate of the top-left corner of the bar chart
        width (int): The width of the bar chart in pixels
        height (int): The height of the bar chart in pixels
        data (List[int]): The values to plot in the bar chart
        labels (List[str]): The labels for each bar in the chart
    """
    # Calculate the maximum value in the data
    max_value = max(data)

    # Calculate the bar width and spacing
    bar_width = width // len(data)
    bar_spacing = 10

    # Draw the bars
    for i, value in enumerate(data):
        bar_height = (value / max_value) * height
        bar_x = x + i * (bar_width + bar_spacing)
        bar_y = y + height - bar_height
        draw.rectangle(
            [(bar_x, bar_y), (bar_x + bar_width, y + height)], fill="green"
        )

        # Draw the label below the bar
        label_text = labels[i]
        label_font = ImageFont.load_default(size=12)
        label_bbox = draw.textbbox((0, 0), label_text, font=label_font)
        label_width = label_bbox[2] - label_bbox[0]
        label_height = label_bbox[3] - label_bbox[1]
        label_x = bar_x + (bar_width - label_width) // 2
        label_y = y + height + 5
        draw.text((label_x, label_y), label_text, font=label_font, fill="black")

    # Draw the title above the bar chart
    title_text = "Chunk Counts per Section"
    title_font = ImageFont.load_default(size=16)
    title_bbox = draw.textbbox((0, 0), title_text, font=title_font)
    title_width = title_bbox[2] - title_bbox[0]
    title_height = title_bbox[3] - title_bbox[1]
    title_x = x + (width - title_width) // 2
    title_y = y - title_height - 10
    draw.text((title_x, title_y), title_text, font=title_font, fill="black")


@step
def preprocess_documents(
    documents: str,
) -> Tuple[
    Annotated[str, ArtifactConfig(name="split_chunks")],
    Annotated[Image.Image, ArtifactConfig(name="doc_stats_chart")],
]:
    """Preprocesses a JSON string of documents by splitting them into chunks.

    Args:
        documents (str): A JSON string containing a list of documents to be preprocessed.

    Returns:
        Annotated[str, ArtifactConfig(name="split_chunks")]: A JSON string containing a list of preprocessed documents annotated with an ArtifactConfig.

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

        document_list: List[Document] = [
            Document(**doc) for doc in json.loads(documents)
        ]
        split_docs: List[Document] = split_documents(
            document_list, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )

        stats: Dict[str, Dict[str, int]] = extract_docs_stats(
            len(document_list), split_docs
        )
        chart: Image.Image = create_charts(stats)

        log_artifact_metadata(
            artifact_name="split_chunks",
            metadata=stats,
        )

        split_docs_json: str = json.dumps([doc.__dict__ for doc in split_docs])

        return split_docs_json, chart
    except Exception as e:
        logger.error(f"Error in preprocess_documents: {e}")
        raise


@step
def generate_embeddings(
    split_documents: str,
) -> Annotated[str, ArtifactConfig(name="documents_with_embeddings")]:
    """
    Generates embeddings for a list of split documents using a SentenceTransformer model.

    Args:
        split_documents (List[Document]): A list of Document objects that have been split into chunks.

    Returns:
        Annotated[str, ArtifactConfig(name="documents_with_embeddings")]: A JSON string containing the Document objects with generated embeddings, annotated with an ArtifactConfig.

    Raises:
        Exception: If an error occurs during the generation of embeddings.
    """
    try:
        model = SentenceTransformer(EMBEDDINGS_MODEL)

        log_artifact_metadata(
            artifact_name="documents_with_embeddings",
            metadata={
                "embedding_type": EMBEDDINGS_MODEL,
                "embedding_dimensionality": EMBEDDING_DIMENSIONALITY,
            },
        )

        # Parse the JSON string into a list of Document objects
        document_list = [
            Document(**doc) for doc in json.loads(split_documents)
        ]

        document_texts = [doc.page_content for doc in document_list]
        embeddings = model.encode(document_texts)

        for doc, embedding in zip(document_list, embeddings):
            doc.embedding = embedding.tolist()

        # Convert the list of Document objects to a JSON string
        documents_json = json.dumps([doc.__dict__ for doc in document_list])

        return documents_json
    except Exception as e:
        logger.error(f"Error in generate_embeddings: {e}")
        raise


@step
def index_generator(
    documents: str,
) -> None:
    """Generates an index for the given documents.

    This function creates a database connection, installs the pgvector extension if not already installed,
    creates an embeddings table if it doesn't exist, and inserts the embeddings and document metadata into the table.
    It then calculates the index parameters according to best practices and creates an index on the embeddings
    using the cosine distance measure.

    Args:
        documents (str): A JSON string containing the Document objects with generated embeddings.

    Raises:
        Exception: If an error occurs during the index generation.
    """
    conn = None
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

            # Parse the JSON string into a list of Document objects
            document_list = [Document(**doc) for doc in json.loads(documents)]

            # Insert data only if it doesn't already exist
            for doc in document_list:
                content = doc.page_content
                token_count = doc.token_count
                embedding = doc.embedding
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
