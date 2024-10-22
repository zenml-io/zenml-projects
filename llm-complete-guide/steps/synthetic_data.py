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

from typing import Annotated
import logging
import json

import pandas as pd
from datasets import Dataset
from huggingface_hub import create_repo
from litellm import completion
from structures import Document
from zenml import ArtifactConfig, step
from zenml.client import Client

logger = logging.getLogger(__name__)

LOCAL_MODEL = "ollama/mixtral"


def generate_question(chunk: str, local: bool = False) -> str:
    """Generate a question from a chunk.

    Args:
        chunk: Text chunk to generate a question from.

    Returns:
        Generated question.
    """
    model = LOCAL_MODEL if local else "gpt-4o"
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


@step
def generate_questions_from_chunks(
    docs_with_embeddings: str,
    local: bool = False,
    logging_interval: int = 10,
) -> Annotated[str, ArtifactConfig(name="synthetic_questions")]:
    """Generate questions from chunks.

    Args:
        docs_with_embeddings: JSON string containing a list of Document objects with embeddings.
        local: Whether to run the pipeline with a local LLM.

    Returns:
        JSON string containing a list of documents with generated questions added.
    """
    document_list = [
        Document(**doc) for doc in json.loads(docs_with_embeddings)
    ]

    for i, doc in enumerate(document_list, 1):
        doc.generated_questions = [generate_question(doc.page_content, local)]
        if i % logging_interval == 0:
            logger.info(
                f"Progress: {i}/{len(document_list)} documents processed"
            )
            logger.info(
                f"Generated question for document {i}: {doc.generated_questions[0]}"
            )

    assert all(doc.generated_questions for doc in document_list)

    # Convert List[Document] to DataFrame
    df = pd.DataFrame([doc.__dict__ for doc in document_list])

    # upload the parquet file to a private dataset on the huggingface hub
    client = Client()
    hf_token = client.get_secret("huggingface_datasets").secret_values["token"]

    create_repo(
        "zenml/rag_qa_embedding_questions",
        token=hf_token,
        exist_ok=True,
        repo_type="dataset",
    )

    # add an extra `__pydantic_initialised__` column to the dataframe
    df["__pydantic_initialised__"] = True

    dataset = Dataset.from_pandas(df)
    dataset.push_to_hub(
        repo_id="zenml/rag_qa_embedding_questions",
        token=hf_token,
        create_pr=True,
    )

    return docs_with_embeddings
