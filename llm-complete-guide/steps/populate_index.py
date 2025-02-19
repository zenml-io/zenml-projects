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

import hashlib
import json
import logging
import math
import warnings
from enum import Enum
from typing import Annotated, Any, Dict, List, Tuple

# Suppress the specific FutureWarning about clean_up_tokenization_spaces
warnings.filterwarnings(
    "ignore",
    message=".*clean_up_tokenization_spaces.*",
    category=FutureWarning,
    module="transformers.tokenization_utils_base",
)

from constants import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_DIMENSIONALITY,
    EMBEDDINGS_MODEL,
    SECRET_NAME,
    SECRET_NAME_ELASTICSEARCH,
    SECRET_NAME_PINECONE,
)
from pgvector.psycopg2 import register_vector
from PIL import Image, ImageDraw, ImageFont
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from structures import Document
from utils.llm_utils import get_db_conn, get_es_client, get_pinecone_client, split_documents
from zenml import ArtifactConfig, get_step_context, log_metadata, step
from zenml.client import Client
from zenml.metadata.metadata_types import Uri

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def draw_value_label(
    draw: ImageDraw.Draw, value: float, x: int, y: int, bar_width: int
) -> None:
    """Draws a value label above a bar in a chart.

    Args:
        draw: The ImageDraw object to draw on
        value: The value to display
        x: The x coordinate of the bar
        y: The y coordinate of the top of the bar
        bar_width: The width of the bar
    """
    label = str(round(value))
    font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), label, font=font)
    label_width = bbox[2] - bbox[0]
    label_x = x + (bar_width - label_width) // 2
    draw.text((label_x, y - 15), label, font=font, fill="black")


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

    # Add histogram buckets
    num_buckets = 10
    bucket_size = (max_chunk_size - min_chunk_size) / num_buckets
    buckets = [0] * num_buckets
    bucket_ranges = []

    for size in chunk_sizes:
        bucket_index = min(
            int((size - min_chunk_size) / bucket_size), num_buckets - 1
        )
        buckets[bucket_index] += 1

    return {
        "document_stats": {
            "total_documents": total_documents,
            "total_chunks": total_chunks,
            "avg_chunk_size": avg_chunk_size,
            "min_chunk_size": min_chunk_size,
            "max_chunk_size": max_chunk_size,
            "size_distribution": buckets,
            "bucket_size": bucket_size,
        },
        "chunks_per_section": chunks_per_section,
    }


def create_charts(stats: Dict[str, Dict[str, int]]) -> Image.Image:
    """Creates a combined visualization with both a histogram and bar chart.

    Args:
        stats: Dictionary containing statistics about document chunks, including:
            - document_stats: Contains histogram data and chunk size statistics
            - chunks_per_section: Maps document sections to number of chunks

    Returns:
        PIL Image containing both histogram and bar chart visualizations
    """
    document_stats = stats["document_stats"]
    chunks_per_section = stats["chunks_per_section"]

    histogram_width = 600
    histogram_height = 300
    bar_chart_width = 600
    bar_chart_height = 300

    padding = 20
    histogram_y = padding
    bar_chart_y = histogram_y + histogram_height + 60

    image_width = max(histogram_width, bar_chart_width) + 2 * padding
    image_height = histogram_height + bar_chart_height + 100
    image = Image.new("RGB", (image_width, image_height), color="white")
    draw = ImageDraw.Draw(image)

    title_text = "Document Chunk Statistics"
    title_font = ImageFont.load_default(size=24)
    title_bbox = draw.textbbox((0, 0), title_text, font=title_font)
    title_width = title_bbox[2] - title_bbox[0]
    title_x = (image_width - title_width) // 2
    title_y = padding
    draw.text((title_x, title_y), title_text, font=title_font, fill="black")

    histogram_x = (image_width - histogram_width) // 2
    histogram_data = document_stats["size_distribution"]
    histogram_labels = ["Min", "Avg", "Max"]
    histogram_title = "Chunk Size Distribution (Character Count)"
    draw_histogram(
        draw,
        histogram_x,
        histogram_y + 40,
        histogram_width,
        histogram_height,
        histogram_data,
        histogram_labels,
        histogram_title,
    )

    bar_chart_x = (image_width - bar_chart_width) // 2
    bar_chart_data = list(chunks_per_section.values())
    bar_chart_labels = list(chunks_per_section.keys())
    bar_chart_title = "Number of Chunks per Document Section"
    draw_bar_chart(
        draw,
        bar_chart_x,
        bar_chart_y + 40,
        bar_chart_width,
        bar_chart_height,
        bar_chart_data,
        bar_chart_labels,
        bar_chart_title,
    )

    return image


def create_histogram(stats: Dict[str, Dict[str, int]]) -> Image.Image:
    """Creates a histogram visualization showing the distribution of chunk sizes.

    Args:
        stats: Dictionary containing statistics about document chunks, including:
            - document_stats: Contains histogram data and chunk size statistics
            - chunks_per_section: Maps document sections to number of chunks

    Returns:
        PIL Image containing the rendered histogram visualization
    """
    document_stats = stats["document_stats"]

    histogram_width = 600
    histogram_height = 300

    left_padding = 40
    right_padding = 40
    top_padding = 40
    bottom_padding = 40

    image = Image.new(
        "RGB",
        (
            histogram_width + left_padding + right_padding,
            histogram_height + top_padding + bottom_padding,
        ),
        color="white",
    )
    draw = ImageDraw.Draw(image)

    histogram_x = left_padding
    histogram_y = top_padding
    histogram_data = document_stats["size_distribution"]
    histogram_labels = []  # We'll generate these in draw_histogram
    histogram_title = "Chunk Size Distribution (Character Count)"

    draw_histogram(
        draw,
        histogram_x,
        histogram_y,
        histogram_width,
        histogram_height,
        histogram_data,
        histogram_labels,
        histogram_title,
        document_stats,
        image,
    )

    return image


def create_bar_chart(stats: Dict[str, Dict[str, int]]) -> Image.Image:
    """Creates a bar chart showing the number of chunks per document section.

    Args:
        stats: Dictionary containing statistics about the document chunks, including
            a 'chunks_per_section' key mapping to a dict of section names to chunk counts.

    Returns:
        PIL Image containing the rendered bar chart visualization.
    """
    chunks_per_section = stats["chunks_per_section"]

    bar_chart_width = 600
    bar_chart_height = 300
    padding = 20

    image = Image.new(
        "RGB",
        (bar_chart_width + 2 * padding, bar_chart_height + 80),
        color="white",
    )
    draw = ImageDraw.Draw(image)

    bar_chart_x = padding
    bar_chart_y = 40
    bar_chart_data = list(chunks_per_section.values())
    bar_chart_labels = list(chunks_per_section.keys())
    bar_chart_title = "Number of Chunks per Document Section"

    draw_bar_chart(
        draw,
        bar_chart_x,
        bar_chart_y,
        bar_chart_width,
        bar_chart_height,
        bar_chart_data,
        bar_chart_labels,
        bar_chart_title,
    )

    return image


def draw_rotated_text(
    image: Image.Image,
    text: str,
    position: Tuple[int, int],
    font: ImageFont.ImageFont,
) -> None:
    """Helper function to draw rotated text on an image.

    Args:
        image: The image to draw on
        text: The text to draw
        position: (x, y) position to draw the text
        font: The font to use
    """
    # Create a new image for the text with RGBA mode
    bbox = font.getbbox(text)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    # Create a transparent image for the text
    txt_img = Image.new("RGBA", (text_width, text_height), (255, 255, 255, 0))
    txt_draw = ImageDraw.Draw(txt_img)

    # Draw the text onto the image
    txt_draw.text((0, 0), text, font=font, fill="black")

    # Rotate the text image
    rotated = txt_img.rotate(90, expand=True)

    # Create a temporary RGBA version of the main image
    temp_img = image.convert("RGBA")
    temp_img.paste(rotated, position, rotated)

    # Convert back to RGB and update the original image
    rgb_img = temp_img.convert("RGB")
    image.paste(rgb_img)


def draw_histogram(
    draw: ImageDraw.Draw,
    x: int,
    y: int,
    width: int,
    height: int,
    data: List[int],
    labels: List[str],
    title: str,
    document_stats: Dict[str, Any],
    image: Image.Image,
) -> None:
    """Draws a histogram chart on the given image.

    Args:
        draw: The ImageDraw object to draw on
        x: The x coordinate of the top-left corner
        y: The y coordinate of the top-left corner
        width: The total width of the chart area
        height: The total height of the chart area
        data: List of values for each histogram bar
        labels: List of labels for each bar
        title: The title of the chart
        document_stats: Dictionary containing statistics about the document chunks
        image: The PIL Image object to draw on

    Returns:
        None
    """
    # Calculate the maximum value in the data
    max_value = max(data)

    # Adjust margins and positioning (reduced left margin since we removed the label)
    left_margin = 40  # Changed from 80
    right_margin = 40
    top_margin = 40
    bottom_margin = 40
    x += left_margin
    y += top_margin

    # Rest of the function remains the same, but remove the y-axis label drawing code
    usable_width = width - left_margin - right_margin
    usable_height = height - top_margin - bottom_margin
    bar_width = usable_width // len(data)
    bar_spacing = 5

    # Draw y-axis
    draw.line([(x, y), (x, y + usable_height)], fill="black", width=1)

    # Draw y-axis ticks and labels
    num_ticks = 5
    for i in range(num_ticks + 1):
        tick_value = (max_value * i) / num_ticks
        tick_y = y + usable_height - (usable_height * i / num_ticks)

        # Draw tick mark
        draw.line([(x - 5, tick_y), (x, tick_y)], fill="black", width=1)

        # Draw tick label
        label = str(int(tick_value))
        font = ImageFont.load_default(size=10)
        bbox = draw.textbbox((0, 0), label, font=font)
        label_width = bbox[2] - bbox[0]
        draw.text(
            (x - 10 - label_width, tick_y - 5), label, font=font, fill="black"
        )

    # Draw bars with value labels
    for i, value in enumerate(data):
        bar_height = (value / max_value) * usable_height
        bar_x = x + i * (bar_width + bar_spacing)
        bar_y = y + usable_height - bar_height

        # Draw bar
        draw.rectangle(
            [(bar_x, bar_y), (bar_x + bar_width, y + usable_height)],
            fill="#4444FF",
            outline="#000000",
        )

        # Add value label on top
        draw_value_label(draw, value, bar_x, bar_y, bar_width)

    # Draw title
    title_font = ImageFont.load_default(size=16)
    title_bbox = draw.textbbox((0, 0), title, font=title_font)
    title_width = title_bbox[2] - title_bbox[0]
    title_x = x + (usable_width - title_width) // 2
    title_y = y - 10
    draw.text((title_x, title_y), title, font=title_font, fill="black")

    # Draw x-axis labels with actual character count ranges
    label_interval = max(len(data) // 5, 1)
    min_size = document_stats["min_chunk_size"]
    bucket_size = document_stats["bucket_size"]

    for i in range(0, len(data), label_interval):
        bucket_start = min_size + (i * bucket_size)
        bucket_end = bucket_start + bucket_size
        label = f"{int(bucket_start)}-{int(bucket_end)}"
        font = ImageFont.load_default(size=10)
        bbox = draw.textbbox((0, 0), label, font=font)
        label_width = bbox[2] - bbox[0]
        label_x = (
            x + i * (bar_width + bar_spacing) + (bar_width - label_width) // 2
        )
        draw.text(
            (label_x, y + usable_height + 5), label, font=font, fill="black"
        )


def draw_bar_chart(
    draw: ImageDraw.Draw,
    x: int,
    y: int,
    width: int,
    height: int,
    data: List[int],
    labels: List[str],
    title: str,
) -> None:
    """Draws a bar chart on the given image."""
    # Ensure labels is a list, even if empty
    labels = labels or []

    # Skip drawing if no data
    if not data:
        return

    max_value = max(data)
    bar_width = width // len(data)
    bar_spacing = 10

    for i, value in enumerate(data):
        bar_height = (value / max_value) * (height - 40)
        bar_x = x + i * (bar_width + bar_spacing)
        bar_y = y + height - bar_height - 30

        draw.rectangle(
            [(bar_x, bar_y), (bar_x + bar_width, y + height - 30)],
            fill="#00AA00",
            outline="#000000",
        )

        draw_value_label(draw, value, bar_x, bar_y, bar_width)

    title_font = ImageFont.load_default(size=16)
    title_bbox = draw.textbbox((0, 0), title, font=title_font)
    title_width = title_bbox[2] - title_bbox[0]
    title_x = x + (width - title_width) // 2
    title_y = y - 30
    draw.text((title_x, title_y), title, font=title_font, fill="black")

    # Only try to draw labels if they exist
    if labels:
        for i, label in enumerate(labels):
            if label is not None:  # Add null check for individual labels
                font = ImageFont.load_default(size=10)
                bbox = draw.textbbox(
                    (0, 0), str(label), font=font
                )  # Convert to string
                label_width = bbox[2] - bbox[0]
                label_x = (
                    x
                    + i * (bar_width + bar_spacing)
                    + (bar_width - label_width) // 2
                )
                draw.text(
                    (label_x, y + height - 15),
                    str(label),
                    font=font,
                    fill="black",
                )


@step
def preprocess_documents(
    documents: str,
) -> Tuple[
    Annotated[str, ArtifactConfig(name="split_chunks")],
    Annotated[Image.Image, ArtifactConfig(name="histogram_chart")],
    Annotated[Image.Image, ArtifactConfig(name="bar_chart")],
]:
    """Preprocesses a JSON string of documents by splitting them into chunks.

    Args:
        documents (str): A JSON string containing a list of documents to be preprocessed.

    Returns:
        Annotated[str, ArtifactConfig(name="split_chunks")]: A JSON string containing a list of preprocessed documents annotated with an ArtifactConfig.
        Annotated[Image.Image, ArtifactConfig(name="histogram_chart")]: A histogram chart showing the distribution of chunk sizes.
        Annotated[Image.Image, ArtifactConfig(name="bar_chart")]: A bar chart showing the number of chunks per document section.

    Raises:
        Exception: If an error occurs during preprocessing.
    """
    try:
        log_metadata(
            metadata={
                "chunk_size": CHUNK_SIZE,
                "chunk_overlap": CHUNK_OVERLAP,
            },
            artifact_name="split_chunks",
            infer_artifact=True,
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
        histogram_chart: Image.Image = create_histogram(stats)
        bar_chart: Image.Image = create_bar_chart(stats)

        log_metadata(
            artifact_name="split_chunks",
            metadata=stats,
            infer_artifact=True,
        )

        split_docs_json: str = json.dumps([doc.__dict__ for doc in split_docs])

        return split_docs_json, histogram_chart, bar_chart
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
        # Initialize the model
        model = SentenceTransformer(EMBEDDINGS_MODEL)

        # Set clean_up_tokenization_spaces to False on the underlying tokenizer to avoid the warning
        if hasattr(model.tokenizer, "clean_up_tokenization_spaces"):
            model.tokenizer.clean_up_tokenization_spaces = False

        log_metadata(
            metadata={
                "embedding_type": EMBEDDINGS_MODEL,
                "embedding_dimensionality": EMBEDDING_DIMENSIONALITY,
            },
            artifact_name="documents_with_embeddings",
            infer_artifact=True,
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


class IndexType(Enum):
    ELASTICSEARCH = "elasticsearch"
    POSTGRES = "postgres"
    PINECONE = "pinecone"


@step(enable_cache=False)
def index_generator(
    documents: str,
    index_type: IndexType = IndexType.PINECONE,
) -> None:
    """Generates an index for the given documents.

    Args:
        documents (str): JSON string containing the documents to index.
        index_type (IndexType, optional): Type of index to generate. Defaults to IndexType.POSTGRES.
    """
    # get model version 
    context = get_step_context()
    model_version_stage = context.model_version.stage
    if index_type == IndexType.ELASTICSEARCH:
        _index_generator_elastic(documents)
    elif index_type == IndexType.POSTGRES:
        _index_generator_postgres(documents)
    elif index_type == IndexType.PINECONE:
        _index_generator_pinecone(documents, model_version_stage)
    else:
        raise ValueError(f"Unknown index type: {index_type}")

    _log_metadata(index_type)


def _index_generator_elastic(documents: str) -> None:
    """Generates an Elasticsearch index for the given documents."""
    try:
        es = get_es_client()
        index_name = "zenml_docs"

        # Create index with mappings if it doesn't exist
        if not es.indices.exists(index=index_name):
            mappings = {
                "mappings": {
                    "properties": {
                        "doc_id": {"type": "keyword"},
                        "content": {"type": "text"},
                        "token_count": {"type": "integer"},
                        "embedding": {
                            "type": "dense_vector",
                            "dims": EMBEDDING_DIMENSIONALITY,
                            "index": True,
                            "similarity": "cosine",
                        },
                        "filename": {"type": "text"},
                        "parent_section": {"type": "text"},
                        "url": {"type": "text"},
                    }
                }
            }
            # TODO move to using mappings param directly
            es.indices.create(index=index_name, body=mappings)

        # Parse the JSON string into a list of Document objects
        document_list = [Document(**doc) for doc in json.loads(documents)]
        operations = []

        for doc in document_list:
            content_hash = hashlib.md5(
                f"{doc.page_content}{doc.filename}{doc.parent_section}{doc.url}".encode()
            ).hexdigest()

            exists_query = {"query": {"term": {"doc_id": content_hash}}}

            if not es.count(index=index_name, body=exists_query)["count"]:
                operations.append(
                    {"index": {"_index": index_name, "_id": content_hash}}
                )

                operations.append(
                    {
                        "doc_id": content_hash,
                        "content": doc.page_content,
                        "token_count": doc.token_count,
                        "embedding": doc.embedding,
                        "filename": doc.filename,
                        "parent_section": doc.parent_section,
                        "url": doc.url,
                    }
                )

        if operations:
            response = es.bulk(operations=operations, timeout="10m")

            success_count = sum(
                1
                for item in response["items"]
                if "index" in item and item["index"]["status"] == 201
            )
            failed_count = len(response["items"]) - success_count

            logger.info(f"Successfully indexed {success_count} documents")
            if failed_count > 0:
                logger.warning(f"Failed to index {failed_count} documents")
                for item in response["items"]:
                    if "index" in item and item["index"]["status"] != 201:
                        logger.warning(
                            f"Failed to index document: {item['index']['error']}"
                        )
        else:
            logger.info("No new documents to index")

        _log_metadata(index_type=IndexType.ELASTICSEARCH)

    except Exception as e:
        logger.error(f"Error in Elasticsearch indexing: {e}")
        raise


def _index_generator_postgres(documents: str) -> None:
    """Generates a PostgreSQL index for the given documents."""
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

        _log_metadata(index_type=IndexType.POSTGRES)

    except Exception as e:
        logger.error(f"Error in PostgreSQL indexing: {e}")
        raise
    finally:
        if conn:
            conn.close()


def _index_generator_pinecone(documents: str, model_version_stage: str) -> None:
    """Generates a Pinecone index for the given documents.

    Args:
        documents (str): JSON string containing the documents to index.
        model_version (str): Name of the model version.
    """
    index = get_pinecone_client(model_version_stage=model_version_stage)

    # Load documents
    docs = json.loads(documents)

    # Batch size for upserting vectors
    batch_size = 100
    batch = []

    for doc in docs:
        # Create a unique ID for the document
        doc_id = hashlib.sha256(
            f"{doc['filename']}:{doc['parent_section']}:{doc['page_content']}".encode()
        ).hexdigest()

        # Create vector record
        vector_record = {
            "id": doc_id,
            "values": doc["embedding"],
            "metadata": {
                "filename": doc["filename"],
                "parent_section": doc["parent_section"] or "",
                "url": doc["url"],
                "page_content": doc["page_content"],
                "token_count": doc["token_count"],
            },
        }
        batch.append(vector_record)

        # Upsert batch when it reaches the batch size
        if len(batch) >= batch_size:
            index.upsert(vectors=batch)
            batch = []

    # Upsert any remaining vectors
    if batch:
        index.upsert(vectors=batch)

    logger.info(f"Successfully indexed {len(docs)} documents to Pinecone index")


def _log_metadata(index_type: IndexType) -> None:
    """Log metadata about the indexing process."""
    prompt = """
    You are a friendly chatbot. \
    You can answer questions about ZenML, its features and its use cases. \
    You respond in a concise, technically credible tone. \
    You ONLY use the context from the ZenML documentation to provide relevant answers. \
    You do not make up answers or provide opinions that you don't have information to support. \
    If you are unsure or don't know, just say so. \
    """

    client = Client()

    if index_type == IndexType.ELASTICSEARCH:
        es_host = client.get_secret(SECRET_NAME_ELASTICSEARCH).secret_values[
            "elasticsearch_host"
        ]
        connection_details = {
            "host": es_host,
            "api_key": "*********",
        }
        store_name = "elasticsearch"
    elif index_type == IndexType.POSTGRES:
        store_name = "pgvector"
        connection_details = {
            "user": client.get_secret(SECRET_NAME).secret_values[
                "supabase_user"
            ],
            "password": "**********",
            "host": client.get_secret(SECRET_NAME).secret_values[
                "supabase_host"
            ],
            "port": client.get_secret(SECRET_NAME).secret_values[
                "supabase_port"
            ],
            "dbname": "postgres",
        }
    elif index_type == IndexType.PINECONE:
        store_name = "pinecone"
        connection_details = {
            "api_key": "**********",
            "environment": client.get_secret(
                SECRET_NAME_PINECONE
            ).secret_values["pinecone_env"],
        }

    log_metadata(
        metadata={
            "embeddings": {
                "model": EMBEDDINGS_MODEL,
                "dimensionality": EMBEDDING_DIMENSIONALITY,
                "model_url": Uri(f"https://huggingface.co/{EMBEDDINGS_MODEL}"),
            },
            "prompt": {
                "content": prompt,
            },
            "vector_store": {
                "name": store_name,
                "connection_details": connection_details,
            },
        },
        infer_model=True,
    )
