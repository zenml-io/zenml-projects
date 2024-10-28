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

import polars as pl
from datasets import Dataset
from huggingface_hub import create_repo
from zenml import step
from zenml.client import Client

from constants import SECRET_NAME


@step
def upload_chunks_dataset_to_huggingface(
    documents: pl.DataFrame, dataset_suffix: str
) -> str:
    """Uploads chunked documents to Hugging Face dataset."""
    client = Client()
    hf_token = client.get_secret(SECRET_NAME).secret_values["hf_token"]

    repo_name = f"zenml/rag_qa_embedding_questions_{dataset_suffix}"

    create_repo(
        repo_name,
        token=hf_token,
        exist_ok=True,
        private=True,
        repo_type="dataset",
    )

    # Convert the list of questions to a single string
    documents = documents.with_columns(
        pl.col("generated_questions").apply(lambda x: "\n".join(x))
    )

    dataset = Dataset(documents.to_arrow())
    dataset.push_to_hub(
        repo_id=repo_name,
        private=True,
        token=hf_token,
    )
    return repo_name
