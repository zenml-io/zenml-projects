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

import json
from typing import Annotated

from structures import Document
from unstructured.partition.html import partition_html
from zenml import ArtifactConfig, step

from steps.url_scraping_utils import extract_parent_section


@step
def web_url_loader(
    urls: str,
) -> Annotated[str, ArtifactConfig(name="documents_from_urls")]:
    """Loads documents from a list of URLs.

    Args:
        urls: JSON string containing a list of URL strings to load documents from.

    Returns:
        JSON string containing a list of custom Document objects.
    """
    url_list = json.loads(urls)
    documents = []
    for url in url_list:
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

    return json.dumps([doc.__dict__ for doc in documents])
