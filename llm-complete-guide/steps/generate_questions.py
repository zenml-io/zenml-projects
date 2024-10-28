#  Copyright (c) ZenML GmbH 2024. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at:
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
#  or implied. See the License for the specific language governing
#  permissions and limitations under the License.

import time
from typing import Annotated, List

import polars as pl
from litellm import completion
from litellm.exceptions import APIConnectionError, Timeout
from rich import print
from structures import Document
from zenml import log_artifact_metadata, step
from zenml.logger import get_logger

from utils.openai_utils import get_openai_api_key

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
    # skip if the chunk is too short
    if len(chunk) < 50:
        return ""
    model = LOCAL_MODEL if local else "gpt-3.5-turbo"

    for attempt in range(max_retries):
        try:
            response = completion(
                model=model,
                temperature=1.2,
                messages=[
                    {
                        "content": f"Here is a chunk of text from some documentation: <chunk>{chunk}</chunk>. Please read this text carefully. Think about what kinds of questions that a user of ZenML would ask where the chunk itself would contain the information needed to answer the question. Now, please generate a single question that can be answered using only the information provided in the chunk of text. Only output this question and nothing else.",
                        "role": "user",
                    }
                ],
                api_base="http://localhost:11434" if local else None,
                api_key=get_openai_api_key(),
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
