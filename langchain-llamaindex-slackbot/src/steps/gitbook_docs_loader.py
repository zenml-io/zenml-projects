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
from langchain.document_loaders import GitbookLoader
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
)
from zenml.steps import BaseParameters, step


class DocsLoaderParameters(BaseParameters):
    docs_uri: str = "https://docs.zenml.io"
    docs_base_url: str = "https://docs.zenml.io"


@step(enable_cache=True)
def docs_loader(params: DocsLoaderParameters) -> List[Document]:
    # langchain loader; returns langchain documents
    loader = GitbookLoader(
        web_page=params.docs_uri,
        base_url=params.docs_base_url,
        load_all_paths=True,
    )
    all_pages_data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
    )
    return text_splitter.split_documents(all_pages_data)
