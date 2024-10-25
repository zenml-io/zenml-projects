# Apache Software License 2.0
#
# Copyright (c) ZenML GmbH 2024. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from steps.populate_index import (
    generate_embeddings,
    index_generator,
    preprocess_documents,
)
from steps.url_scraper import url_scraper
from steps.web_url_loader import web_url_loader
from zenml import pipeline, Model

model_definition = Model(
    name=""
)


@pipeline
def llm_basic_rag(model=model_definition) -> None:
    """Executes the pipeline to train a basic RAG model.

    This function performs the following steps:
    1. Scrapes URLs using the url_scraper function.
    2. Loads documents from the scraped URLs using the web_url_loader function.
    3. Preprocesses the loaded documents using the preprocess_documents function.
    4. Generates embeddings for the preprocessed documents using the generate_embeddings function.
    5. Generates an index for the embeddings and documents using the index_generator function.
    """
    urls = url_scraper()
    docs = web_url_loader(urls=urls)
    processed_docs = preprocess_documents(documents=docs)
    embedded_docs = generate_embeddings(split_documents=processed_docs)
    index_generator(documents=embedded_docs)
