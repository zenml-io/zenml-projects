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

from typing import List

from langchain.docstore.document import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema.vectorstore import VectorStore
from langchain.text_splitter import (
    CharacterTextSplitter,
)
from langchain.vectorstores.faiss import FAISS
from typing_extensions import Annotated
from zenml import log_artifact_metadata, step


@step
def index_generator(
    documents: List[Document],
) -> Annotated[VectorStore, "vector_store"]:
    embeddings = OpenAIEmbeddings()

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
