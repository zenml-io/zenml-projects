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

import nltk
from langchain_community.docstore.document import Document
from langchain_community.document_loaders import UnstructuredURLLoader
from zenml import step


@step
def web_url_loader(urls: List[str]) -> List[Document]:
    """Loads documents from a list of URLs.

    Args:
        urls: List of URLs to load documents from.

    Returns:
        List of langchain documents.
    """
    # Set NLTK data path to a writable directory
    nltk_data_dir = os.path.join(os.getcwd(), "nltk_data")
    os.makedirs(nltk_data_dir, exist_ok=True)
    nltk.data.path.append(nltk_data_dir)

    # Download required NLTK data
    nltk.download("punkt", download_dir=nltk_data_dir)
    nltk.download("wordnet", download_dir=nltk_data_dir)
    nltk.download("omw-1.4", download_dir=nltk_data_dir)
    nltk.download("punkt_tab", download_dir=nltk_data_dir)
    nltk.download("averaged_perceptron_tagger_eng", download_dir=nltk_data_dir)

    loader = UnstructuredURLLoader(
        urls=urls,
    )
    return loader.load()
