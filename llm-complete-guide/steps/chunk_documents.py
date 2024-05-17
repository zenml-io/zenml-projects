from typing import List

import polars as pl
from constants import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
)
from structures import Document
from utils.llm_utils import split_documents
from zenml import step
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


@step
def chunk_documents(
    docs_df: pl.DataFrame, chunking_method: str = "default"
) -> pl.DataFrame:
    """Chunk documents."""
    documents: List[Document] = [
        Document(filename=row["filename"], page_content=row["page_content"])
        for row in docs_df.to_dicts()
    ]

    match chunking_method:
        case "default":
            return split_and_return_docs(
                documents, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
            )
        case "split-by-document":
            return docs_df
        case "split_by_header":
            return pl.DataFrame()
        case "chunk_size_1000":
            return split_and_return_docs(
                documents, chunk_size=1000, chunk_overlap=100
            )
        case "chunk_size_500":
            return split_and_return_docs(
                documents, chunk_size=500, chunk_overlap=50
            )
        case "chunk_size_4000":
            return split_and_return_docs(
                documents, chunk_size=4000, chunk_overlap=400
            )
        case _:
            raise ValueError(f"Unknown chunking method: {chunking_method}")
