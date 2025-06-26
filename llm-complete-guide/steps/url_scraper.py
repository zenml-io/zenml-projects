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

from typing_extensions import Annotated
from zenml import ArtifactConfig, log_metadata, step

from steps.url_scraping_utils import get_all_pages


@step
def url_scraper(
    docs_url: str = "https://docs.zenml.io",
    repo_url: str = "https://github.com/zenml-io/zenml",
    website_url: str = "https://zenml.io",
    use_dev_set: bool = False,
) -> Annotated[str, ArtifactConfig(name="urls")]:
    """Generates a list of relevant URLs to scrape.

    Args:
        docs_url: URL to the documentation.
        repo_url: URL to the repository.
        website_url: URL to the website.

    Returns:
        JSON string containing a list of URLs to scrape.
    """
    # We comment this out to make this pipeline faster
    # examples_readme_urls = get_nested_readme_urls(repo_url)
    if use_dev_set:
        docs_urls = [
            "https://docs.zenml.io/getting-started/system-architectures",
            "https://docs.zenml.io/getting-started/core-concepts",
            "https://docs.zenml.io/user-guides/llmops-guide/rag-with-zenml/rag-85-loc",
            "https://docs.zenml.io/concepts/metadata",
            "https://docs.zenml.io/user-guides/best-practices/debug-and-solve-issues",
            "https://docs.zenml.io/stack-components/step-operators/azureml",
            # "https://docs.zenml.io/concepts/secrets",
            # "https://docs.zenml.io/stacks/service-connectors/service-connectors-guide",
            # "https://docs.zenml.io/stacks/service-connectors/hyperai-service-connector",
            # "https://docs.zenml.io/stack-components/data-validators/evidently",
            # "https://docs.zenml.io/stack-components/data-validators",
            # "https://docs.zenml.io/stack-components/step-operators/sagemaker",
            # "https://docs.zenml.io/stack-components/alerters/slack",
            # "https://docs.zenml.io/stacks/service-connectors/kubernetes-service-connector",
            # "https://docs.zenml.io/stacks/service-connectors/azure-service-connector"
        ]
    else:
        docs_urls = get_all_pages(docs_url)

    # website_urls = get_all_pages(website_url)
    # all_urls = docs_urls + website_urls
    # all_urls = website_urls
    all_urls = docs_urls
    log_metadata(
        metadata={
            "count": len(all_urls),
        },
        infer_artifact=True,
    )
    return json.dumps(all_urls)
