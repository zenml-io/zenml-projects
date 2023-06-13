#  Copyright (c) ZenML GmbH 2023. All Rights Reserved.
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
from langchain.text_splitter import (
    CharacterTextSplitter,
)
from langchain.vectorstores import Weaviate
from zenml.steps import step, BaseParameters


class IndexParameters(BaseParameters):
    """Parameters for the agent."""

    weaviate_settings: dict = {
        "url": "",
    }


@step(enable_cache=True)
def index_generator(
    documents: List[Document], config: IndexParameters
) -> Weaviate:
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    compiled_texts = text_splitter.split_documents(documents)

    return Weaviate.from_documents(
        index_name="documents",
        documents=compiled_texts,
        weaviate_url=config.weaviate_settings["url"],
    )
