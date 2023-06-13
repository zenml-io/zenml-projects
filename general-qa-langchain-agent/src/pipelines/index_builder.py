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


from zenml.pipelines import pipeline

pipeline_name = "zenml_docs_index_generation"


@pipeline(name=pipeline_name)
def docs_to_index_pipeline(
    url_scraper, web_loader, index_generator, agent_creator
):
    urls = url_scraper()
    documents = web_loader(urls)
    vector_store = index_generator(documents)
    agent = agent_creator(vector_store)
    # deploy_agent(agent)
