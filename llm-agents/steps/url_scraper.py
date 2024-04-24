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

from typing_extensions import Annotated
from zenml import log_artifact_metadata, step


@step
def url_scraper(
    docs_url: str = "https://docs.zenml.io",
    repo_url: str = "https://github.com/zenml-io/zenml",
    website_url: str = "https://zenml.io",
) -> Annotated[List[str], "urls"]:
    """Generates a list of relevant URLs to scrape.

    Args:
        docs_url: URL to the documentation.
        repo_url: URL to the repository.
        release_notes_url: URL to the release notes.
        website_url: URL to the website.

    Returns:
        List of URLs to scrape.
    """

    # We comment this out to make this pipeline faster
    # examples_readme_urls = get_nested_readme_urls(repo_url)
    # docs_urls = get_all_pages(docs_url)
    # website_urls = get_all_pages(website_url)
    # all_urls = docs_urls + website_urls + examples_readme_urls
    all_urls = [website_url]
    log_artifact_metadata(
        metadata={
            "count": len(all_urls),
        },
    )
    return all_urls
