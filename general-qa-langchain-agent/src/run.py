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

import logging

from langchain.llms import HuggingFaceHub

from pipelines.index_builder import docs_to_index_pipeline
from steps.index_generator import index_generator, IndexParameters
from steps.url_scraper import UrlScraperParameters, url_scraper
from steps.web_url_loader import web_url_loader
from steps.agent_creator import agent_creator, AgentParameters


def main():
    docs_url = "https://docs.zenml.io"
    repo_url = "https://github.com/zenml-io/zenml/tree/main/examples"
    release_notes_url = (
        "https://github.com/zenml-io/zenml/blob/main/RELEASE_NOTES.md"
    )
    weaviate_settings = {"url": "http://34.159.200.12"}

    slackbot_pipeline = docs_to_index_pipeline(
        url_scraper=url_scraper(
            params=UrlScraperParameters(
                docs_url=docs_url,
                repo_url=repo_url,
                release_notes_url=release_notes_url,
            )
        ),
        web_loader=web_url_loader(),
        index_generator=index_generator(
            config=IndexParameters(weaviate_settings=weaviate_settings)
        ),
        agent_creator=agent_creator(
            config=AgentParameters(
                llm=HuggingFaceHub(
                    repo_id="google/flan-t5-xl",
                    # huggingfacehub_api_token="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",
                    model_kwargs={"temperature": 0, "max_length": 500},
                ),
                weaviate_settings=weaviate_settings,
            )
        ),
    )

    slackbot_pipeline.run()


if __name__ == "__main__":
    logging.basicConfig(level="INFO")
    logging.getLogger().setLevel(logging.INFO)
    main()
