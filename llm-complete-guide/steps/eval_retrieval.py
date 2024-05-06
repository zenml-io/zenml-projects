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
from typing import Annotated

from datasets import load_dataset
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
    """Query similar documents for a given question and URL ending.

    Args:
        question: Question to query similar documents for.
        url_ending: URL ending to compare the retrieved documents against.

    Returns:
        Tuple containing the question, URL ending, and retrieved URLs.
    """
    embedded_question = get_embeddings(question)
    db_conn = get_db_conn()
    top_similar_docs_urls = get_topn_similar_docs(
        embedded_question, db_conn, n=5, only_urls=True
    )
    urls = [url[0] for url in top_similar_docs_urls]  # Unpacking URLs
    return (question, url_ending, urls)


def test_retrieved_docs_retrieve_best_url(question_doc_pairs: list) -> float:
    """Test if the retrieved documents contain the best URL ending.

    Args:
        question_doc_pairs: List of dictionaries containing questions and URL
        endings.

    Returns:
        The failure rate of the retrieval test.
    """
    total_tests = len(question_doc_pairs)
    failures = 0

    for pair in question_doc_pairs:
        question, url_ending, urls = query_similar_docs(
            pair["question"], pair["url_ending"]
        )
        if all(url_ending not in url for url in urls):
            logging.error(
                f"Failed for question: {question}. Expected URL ending: {url_ending}. Got: {urls}"
            )
            failures += 1

    logging.info(f"Total tests: {total_tests}. Failures: {failures}")
    failure_rate = (failures / total_tests) * 100
    return round(failure_rate, 2)


@step
def retrieval_evaluation_small() -> (
    Annotated[float, "small_failure_rate_retrieval"]
):
    """Executes the retrieval evaluation step.

    Returns:
        The failure rate of the retrieval test.
    """
    failure_rate = test_retrieved_docs_retrieve_best_url(question_doc_pairs)
    logging.info(f"Retrieval failure rate: {failure_rate}%")
    return failure_rate


@step
def retrieval_evaluation_full(
    sample_size: int = 50,
) -> Annotated[float, "full_failure_rate_retrieval"]:
    """Executes the retrieval evaluation step.

    Args:
        sample_size: Number of samples to use for the evaluation.

    Returns:
        The failure rate of the retrieval test.
    """
    # Load the dataset from the Hugging Face Hub
    dataset = load_dataset("zenml/rag_qa_embedding_questions", split="train")

    # Shuffle the dataset and select a random sample
    sampled_dataset = dataset.shuffle(seed=42).select(range(sample_size))

    total_tests = len(sampled_dataset)
    failures = 0

    for item in sampled_dataset:
        generated_questions = item["generated_questions"]
        question = generated_questions[
            0
        ]  # Assuming only one question per item
        url_ending = item["filename"].split("/")[
            -1
        ]  # Extract the URL ending from the filename

        _, _, urls = query_similar_docs(question, url_ending)

        if all(url_ending not in url for url in urls):
            logging.error(
                f"Failed for question: {question}. Expected URL ending: {url_ending}. Got: {urls}"
            )
            failures += 1

    logging.info(f"Total tests: {total_tests}. Failures: {failures}")
    failure_rate = (failures / total_tests) * 100
    return round(failure_rate, 2)
