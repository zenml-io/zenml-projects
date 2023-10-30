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
import os

from steps.index_generator import index_generator
from steps.url_scraper import url_scraper
from steps.web_url_loader import web_url_loader
from zenml import pipeline
from zenml.config import DockerSettings

pipeline_name = "zenml_docs_index_generation"
docker_settings = DockerSettings(
    requirements=[
        "langchain==0.0.263",
        "openai==0.27.2",
        "slack-bolt==1.16.2",
        "slack-sdk==3.20.0",
        "fastapi",
        "flask",
        "uvicorn",
        "gcsfs==2023.5.0",
        "faiss-cpu==1.7.3",
        "unstructured==0.5.7",
        "tiktoken",
        "bs4"
    ],
    environment={"OPENAI_API_KEY": os.getenv("OPENAI_API_KEY")}
)

@pipeline(name=pipeline_name, settings={"docker": docker_settings})
def docs_to_index_pipeline(
    docs_url: str = "",
    repo_url: str = "",
    release_notes_url: str = "",
    website_url: str = "",
) -> None:
    """Generate index for ZenML.

    Args:
        docs_url: URL to the documentation.
        repo_url: URL to the repository.
        release_notes_url: URL to the release notes.
        website_url: URL to the website.
    """
    # minimal sources to boost answer quality
    urls = url_scraper(docs_url=docs_url)
    documents = web_url_loader(urls)
    index_generator(documents)
