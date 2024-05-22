from typing import Annotated, List

import polars as pl
from constants import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
)
from structures import Document
from utils.llm_utils import split_documents
from zenml import log_artifact_metadata, step
from zenml.logger import get_logger

logger = get_logger(__name__)


def split_and_return_docs(
    documents: List[Document], chunk_size: int, chunk_overlap: int
) -> pl.DataFrame:
    split_docs = split_documents(
        documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    logger.debug(f"Split {len(split_docs)} documents")
    logger.debug(f"Example document: {split_docs[0]}")
    return pl.DataFrame(
        {
            "filename": [doc.filename for doc in split_docs],
            "page_content": [doc.page_content for doc in split_docs],
        }
    )


def split_by_header(documents: List[Document]) -> pl.DataFrame:
    """
    Split documents into sections based on headers.

    This function takes a list of Document objects and splits each document's
    content into sections based on the presence of header lines (lines starting
    with '#'). Code blocks (delimited by '```') are ignored during the splitting
    process.

    Args:
        documents (List[Document]): A list of Document objects to be split.

    Returns:
        pl.DataFrame: A DataFrame containing the split documents, with columns
            'filename' and 'page_content'.
    """
    split_docs = []
    for doc in documents:
        lines = doc.page_content.split("\n")
        sections = []
        current_section = ""
        in_code_block = False

        for line in lines:
            if line.startswith("```"):
                in_code_block = not in_code_block
                current_section += line + "\n"
            elif not in_code_block and line.startswith("#"):
                if current_section:
                    sections.append(current_section.strip())
                current_section = line + "\n"
            else:
                current_section += line + "\n"

        if current_section:
            sections.append(current_section.strip())

        split_docs.extend(
            [
                Document(filename=doc.filename, page_content=section)
                for section in sections
            ]
        )

    return pl.DataFrame(
        {
            "filename": [doc.filename for doc in split_docs],
            "page_content": [doc.page_content for doc in split_docs],
        }
    )


@step
def chunk_documents(
    docs_df: pl.DataFrame, chunking_method: str = "default"
) -> Annotated[pl.DataFrame, "chunked_documents"]:
    """
    Chunk documents into smaller pieces based on the specified chunking method.

    Args:
        docs_df (pl.DataFrame): DataFrame containing the documents to be chunked.
        chunking_method (str): The method to use for chunking the documents.
            Supported methods are:
            - "default": Use the default chunking method with predefined chunk size and overlap.
            - "split-by-document": Split documents by document boundaries.
            - "split-by-header": Split documents by header sections.
            - "chunk-size-1000": Chunk documents with a chunk size of 1000 and overlap of 100.
            - "chunk-size-500": Chunk documents with a chunk size of 500 and overlap of 50.
            - "chunk-size-4000": Chunk documents with a chunk size of 4000 and overlap of 400.

    Returns:
        pl.DataFrame: DataFrame containing the chunked documents.

    Raises:
        ValueError: If an unknown chunking method is provided.
    """
    documents: List[Document] = [
        Document(filename=row["filename"], page_content=row["page_content"])
        for row in docs_df.to_dicts()
    ]

    logger.info(f"Chunking documents using method: {chunking_method}")
    num_docs_before_chunking = len(documents)
    logger.info(
        f"Number of documents before chunking: {num_docs_before_chunking}"
    )

    chunked_docs = chunk_documents_by_method(documents, chunking_method)

    num_docs_after_chunking = len(chunked_docs)
    logger.info(
        f"Number of documents after chunking: {num_docs_after_chunking}"
    )
    log_artifact_metadata(
        artifact_name="chunked_documents",
        metadata={
            "before_chunking_count": num_docs_before_chunking,
            "after_chunking_count": num_docs_after_chunking,
            "chunking_method": chunking_method,
        },
    )

    return chunked_docs


def chunk_documents_by_method(
    documents: List[Document], chunking_method: str
) -> pl.DataFrame:
    """
    Chunk documents based on the specified chunking method.

    Args:
        documents (List[Document]): List of documents to be chunked.
        chunking_method (str): The method to use for chunking the documents.
            Supported methods are:
            - "default": Use the default chunking method with predefined chunk size and overlap.
            - "split-by-document": Split documents by document boundaries.
            - "split-by-header": Split documents by header sections.
            - "chunk-size-1000": Chunk documents with a chunk size of 1000 and overlap of 100.
            - "chunk-size-500": Chunk documents with a chunk size of 500 and overlap of 50.
            - "chunk-size-4000": Chunk documents with a chunk size of 4000 and overlap of 400.

    Returns:
        pl.DataFrame: DataFrame containing the chunked documents.

    Raises:
        ValueError: If an unknown chunking method is provided.
    """
    match chunking_method:
        case "default":
            return split_and_return_docs(
                documents, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
            )
        case "split-by-document":
            return pl.DataFrame(
                {
                    "filename": [doc.filename for doc in documents],
                    "page_content": [doc.page_content for doc in documents],
                }
            )
        case "split-by-header":
            return split_by_header(documents)
        case "chunk-size-1000":
            return split_and_return_docs(
                documents, chunk_size=1000, chunk_overlap=100
            )
        case "chunk-size-500":
            return split_and_return_docs(
                documents, chunk_size=500, chunk_overlap=50
            )
        case "chunk-size-4000":
            return split_and_return_docs(
                documents, chunk_size=4000, chunk_overlap=400
            )
        case _:
            raise ValueError(f"Unknown chunking method: {chunking_method}")
