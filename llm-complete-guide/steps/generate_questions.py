import time
from typing import Annotated, List

import polars as pl
from litellm import completion
from litellm.exceptions import APIConnectionError, Timeout
from rich import print
from structures import Document
from zenml import log_artifact_metadata, step
from zenml.logger import get_logger

logger = get_logger(__name__)
LOCAL_MODEL = "ollama/mixtral"


def generate_question(
    chunk: str, local: bool = False, max_retries: int = 3, retry_delay: int = 5
) -> str:
    """Generate a question from a chunk.

    Args:
        chunk: Text chunk to generate a question from.
        local: Whether to use a local model.
        max_retries: Maximum number of retries.
        retry_delay: Delay in seconds between retries.

    Returns:
        Generated question.
    """
    model = LOCAL_MODEL if local else "gpt-4-turbo"

    for attempt in range(max_retries):
        try:
            response = completion(
                model=model,
                messages=[
                    {
                        "content": f"This is some text from ZenML's documentation. Please generate a question that can be asked about this text: `{chunk}`",
                        "role": "user",
                    }
                ],
                api_base="http://localhost:11434" if local else None,
            )
            return response.choices[0].message.content
        except (Timeout, APIConnectionError) as e:
            if attempt < max_retries - 1:
                print(
                    f"Error generating question (attempt {attempt + 1}/{max_retries}): {e}, retrying in {retry_delay} seconds"
                )
                time.sleep(retry_delay)
            else:
                raise e


@step
def generate_questions(
    docs_df: pl.DataFrame,
    local: bool = False,
    num_generations: int = 3,
    logging_frequency: int = 50,
) -> Annotated[pl.DataFrame, "generated_questions"]:
    """Generates questions from a list of documents.

    Args:
        docs_df: DataFrame containing document data.
        local: Whether to use a local model.
        num_generations: Number of questions to generate per document.
            Default is 3.
        logging_frequency: Frequency of logging. Default is 50.

    Returns:
        DataFrame with generated questions added.
    """
    documents: List[Document] = [
        Document(filename=row["filename"], page_content=row["page_content"])
        for row in docs_df.to_dicts()
    ]
    logger.info("Generating questions for all documents...")
    logger.info(f"Number of documents: {len(documents)}")

    start_time = time.time()

    for i, doc in enumerate(documents, start=1):
        doc.generated_questions = [
            generate_question(doc.page_content, local)
            for _ in range(num_generations)
        ]
        if i % logging_frequency == 0:
            elapsed_time = time.time() - start_time
            docs_processed = i
            docs_remaining = len(documents) - i
            time_per_doc = elapsed_time / docs_processed
            estimated_remaining_time = docs_remaining * time_per_doc

            logger.info(
                f"Generated questions for {i}/{len(documents)} documents"
            )
            logger.info(f"Elapsed time: {elapsed_time:.2f} seconds")
            remaining_hours = int(estimated_remaining_time // 3600)
            remaining_minutes = int((estimated_remaining_time % 3600) // 60)
            logger.info(
                f"Estimated remaining time: {estimated_remaining_time:.2f} seconds ({remaining_hours}h {remaining_minutes}m)"
            )
        # log the estimated completion time and rate every 100 documents
        if i % 100 == 0:
            estimated_completion_time = time.strftime(
                "%H:%M",
                time.localtime(
                    start_time + elapsed_time + estimated_remaining_time
                ),
            )
            docs_processed_last_100 = min(100, i - (i // 100 - 1) * 100)
            time_last_100 = (
                time.time()
                - start_time
                - (elapsed_time - time_per_doc * docs_processed_last_100)
            )
            rate_last_100 = docs_processed_last_100 / (
                time_last_100 / 60
            )  # calculate docs per minute

            logger.info(
                f"Estimated completion time: {estimated_completion_time}, "
                f"Generation rate for last 100 documents: {rate_last_100:.2f} docs/min"
            )

    assert all(doc.generated_questions for doc in documents)

    final_df = pl.DataFrame(
        {
            "filename": [doc.filename for doc in documents],
            "page_content": [doc.page_content for doc in documents],
            "generated_questions": [
                doc.generated_questions for doc in documents
            ],
        }
    )
    logger.info("Generated questions for all documents.")
    logger.info(
        f"Generated {len(final_df)} questions for {len(documents)} documents."
    )

    log_artifact_metadata(
        artifact_name="generated_questions",
        metadata={
            "num_documents": len(documents),
            "num_questions_generated": len(final_df),
            "generations_per_document": num_generations,
            "local_generation": local,
        },
    )

    return final_df