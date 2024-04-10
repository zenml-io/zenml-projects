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

from structures import Document
from unstructured.partition.html import partition_html
from zenml import step

from steps.url_scraping_utils import extract_parent_section


@step
def web_url_loader(urls: List[str]) -> List[Document]:
    """Loads documents from a list of URLs.

    Args:
        urls: List of URLs to load documents from.

    Returns:
        List of custom Document objects.
    """
    documents = []
    for url in urls:
        elements = partition_html(url=url)
        text = "\n\n".join([str(el) for el in elements])

        parent_section = extract_parent_section(url)

        document = Document(
            page_content=text,
            url=url,
            filename=url,
            parent_section=parent_section,
        )
        documents.append(document)

    return documents
