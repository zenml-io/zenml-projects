import time
from typing import List

import polars as pl
from litellm import completion
from litellm.exceptions import Timeout
from rich import print
from structures import Document
from zenml import step

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
    model = LOCAL_MODEL if local else "gpt-4o"

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
        except Timeout as e:
            if attempt < max_retries - 1:
                print(
                    f"Error generating question (attempt {attempt + 1}/{max_retries}): {e}, retrying in {retry_delay} seconds"
                )
                time.sleep(retry_delay)
            else:
                raise e


@step
def generate_questions(
    docs_df: pl.DataFrame, local: bool = False, num_generations: int = 3
) -> pl.DataFrame:
    """Generates questions from a list of documents.

    Args:
        docs_df: DataFrame containing document data.
        local: Whether to use a local model.
        num_generations: Number of questions to generate per document. Default is 3.

    Returns:
        DataFrame with generated questions added.
    """
    documents: List[Document] = [
        Document(filename=row["filename"], page_content=row["page_content"])
        for row in docs_df.to_dicts()
    ]

    for doc in documents:
        doc.generated_questions = [
            generate_question(doc.page_content, local)
            for _ in range(num_generations)
        ]

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
    return final_df
