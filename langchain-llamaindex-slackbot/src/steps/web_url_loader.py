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
from langchain.document_loaders import UnstructuredURLLoader
from zenml.steps import BaseParameters, step


class WebLoaderParameters(BaseParameters):
    docs_urls: List[str] = []
    examples_readme_urls: List[str] = []
    release_notes_url: str = ""


@step(enable_cache=True)
def web_url_loader(params: WebLoaderParameters) -> List[Document]:
    docs_urls = params.docs_urls
    examples_readme_urls = params.examples_readme_urls
    release_notes_url = params.release_notes_url

    # combine all urls into a single list
    urls = docs_urls + examples_readme_urls + [release_notes_url]

    loader = UnstructuredURLLoader(
        urls=urls,
    )
    return loader.load()
