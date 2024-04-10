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

import logging
from concurrent.futures import ProcessPoolExecutor as Executor
from concurrent.futures import as_completed
from typing import Annotated

from utils.llm_utils import get_db_conn, get_embeddings, get_topn_similar_docs
from zenml import step

# Adjust logging settings as before
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

question_doc_pairs = [
    {
        "question": "How do I get going with the Label Studio integration? What are the first steps?",
        "url_ending": "stacks-and-components/component-guide/annotators/label-studio",
    },
    {
        "question": "How can I write my own custom materializer?",
        "url_ending": "user-guide/advanced-guide/data-management/handle-custom-data-types",
    },
    {
        "question": "How do I generate embeddings as part of a RAG pipeline when using ZenML?",
        "url_ending": "user-guide/llmops-guide/rag-with-zenml/embeddings-generation",
    },
    {
        "question": "How do I use failure hooks in my ZenML pipeline?",
        "url_ending": "user-guide/advanced-guide/pipelining-features/use-failure-success-hooks",
    },
    {
        "question": "Can I deploy ZenML self-hosted with Helm? How do I do it?",
        "url_ending": "deploying-zenml/zenml-self-hosted/deploy-with-helm",
    },
]


def query_similar_docs(question: str, url_ending: str) -> tuple:
    # This function remains unchanged
    embedded_question = get_embeddings(question)
    db_conn = get_db_conn()
    top_similar_docs_urls = get_topn_similar_docs(
        embedded_question, db_conn, n=5, only_urls=True
    )
    urls = [url[0] for url in top_similar_docs_urls]  # Unpacking URLs
    return (question, url_ending, urls)


def test_retrieved_docs_retrieve_best_url(question_doc_pairs: list) -> float:
    total_tests = len(question_doc_pairs)
    failures = 0
    with Executor(max_workers=4) as executor:
        # Schedule the queries to be executed
        future_to_query = {
            executor.submit(
                query_similar_docs, pair["question"], pair["url_ending"]
            ): pair
            for pair in question_doc_pairs
        }
        for future in as_completed(future_to_query):
            question, url_ending, urls = future.result()
            if all(url_ending not in url for url in urls):
                logging.error(
                    f"Failed for question: {question}. Expected URL ending: {url_ending}. Got: {urls}"
                )
                failures += 1
    logging.info(f"Total tests: {total_tests}. Failures: {failures}")
    failure_rate = (failures / total_tests) * 100
    return round(failure_rate, 2)


@step
def retrieval_evaluation() -> Annotated[float, "failure_rate_retrieval"]:
    """Executes the retrieval evaluation step."""
    failure_rate = test_retrieved_docs_retrieve_best_url(question_doc_pairs)
    logging.info(f"Retrieval failure rate: {failure_rate}%")
    return failure_rate
