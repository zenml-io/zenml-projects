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

from pipelines import llm_basic_rag, llm_eval
from zenml import pipeline


@pipeline
def llm_index_and_evaluate() -> None:
    """Executes the pipeline to train a basic RAG model.

    This function performs the following steps:
    1. Scrapes URLs using the url_scraper function.
    2. Loads documents from the scraped URLs using the web_url_loader function.
    3. Preprocesses the loaded documents using the preprocess_documents function.
    4. Generates embeddings for the preprocessed documents using the generate_embeddings function.
    5. Generates an index for the embeddings and documents using the index_generator function.
    6. Evaluates the RAG pipeline using the llm_eval pipeline.
    """
    llm_basic_rag()
    llm_eval(after="index_generator")
