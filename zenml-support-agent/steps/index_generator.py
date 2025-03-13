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

import os
from typing import List

from langchain.schema.vectorstore import VectorStore
from langchain.text_splitter import (
    CharacterTextSplitter,
)
from langchain_community.docstore.document import Document
from langchain_community.vectorstores.faiss import FAISS
from langchain_openai import OpenAIEmbeddings
from materializers.faiss_materializer import FAISSMaterializer
from typing_extensions import Annotated
from zenml import log_artifact_metadata, step
from zenml.client import Client


@step(output_materializers={"vector_store": FAISSMaterializer})
def index_generator(
    documents: List[Document],
) -> Annotated[VectorStore, "vector_store"]:
    # First try to get API key from environment variable
    api_key = os.getenv("OPENAI_API_KEY")

    # If not found in env, fall back to ZenML secret
    if not api_key:
        secret = Client().get_secret("llm_complete")
        api_key = secret.secret_values["openai_api_key"]

    embeddings = OpenAIEmbeddings(openai_api_key=api_key)

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    compiled_texts = text_splitter.split_documents(documents)

    log_artifact_metadata(
        artifact_name="vector_store",
        metadata={
            "embedding_type": "OpenAIEmbeddings",
            "vector_store_type": "FAISS",
        },
    )

    return FAISS.from_documents(compiled_texts, embeddings)
