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
from langchain.document_loaders import UnstructuredURLLoader, GitLoader
from zenml import step


@step(enable_cache=True)
def web_url_loader(urls: List[str]) -> List[Document]:
    """Loads documents from a list of URLs.

    Args:
        urls: List of URLs to load documents from.

    Returns:
        List of langchain documents.
    """
    normal_urls_loader = UnstructuredURLLoader(
        urls=urls,
    )
    # github_release_notes_loader = GitLoader(
    #     repo_path="zenml-repo",
    #     clone_url="https://github.com/zenml-io/zenml.git",
    #     branch="main",
    #     file_filter=lambda path: path.endswith("RELEASE_NOTES.md"),
    # )
    return normal_urls_loader.load()
