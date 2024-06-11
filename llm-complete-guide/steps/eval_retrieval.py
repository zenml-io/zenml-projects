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
from typing import Annotated, List, Tuple

from datasets import load_dataset
from utils.llm_utils import (
    get_db_conn,
    get_embeddings,
    get_topn_similar_docs,
    rerank_documents,
)
from zenml import step

# Adjust logging settings as before
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

question_doc_pairs = [
    {
        "question": "How do I get going with the Label Studio integration? What are the first steps?",
        "url_ending": "stack-components/annotators/label-studio",
    },
    {
        "question": "How can I write my own custom materializer?",
        "url_ending": "how-to/handle-data-artifacts/handle-custom-data-types",
    },
    {
        "question": "How do I generate embeddings as part of a RAG pipeline when using ZenML?",
        "url_ending": "user-guide/llmops-guide/rag-with-zenml/embeddings-generation",
    },
    {
        "question": "How do I use failure hooks in my ZenML pipeline?",
        "url_ending": "how-to/build-pipelines/use-failure-success-hooks",
    },
    {
        "question": "Can I deploy ZenML self-hosted with Helm? How do I do it?",
        "url_ending": "getting-started/deploying-zenml/deploy-with-helm",
    },
]


def query_similar_docs(
    question: str,
    url_ending: str,
    use_reranking: bool = False,
    returned_sample_size: int = 5,
) -> Tuple[str, str, List[str]]:
    """Query similar documents for a given question and URL ending.

    Args:
        question: Question to query similar documents for.
        url_ending: URL ending to compare the retrieved documents against.
        use_reranking: Whether to use reranking to improve retrieval.
        returned_sample_size: Number of documents to return.

    Returns:
        Tuple containing the question, URL ending, and retrieved URLs.
    """
    embedded_question = get_embeddings(question)
    db_conn = get_db_conn()
    num_docs = 20 if use_reranking else returned_sample_size
    # get (content, url) tuples for the top n similar documents
    top_similar_docs = get_topn_similar_docs(
        embedded_question, db_conn, n=num_docs, include_metadata=True
    )

    if use_reranking:
        reranked_docs_and_urls = rerank_documents(question, top_similar_docs)[
            :returned_sample_size
        ]
        urls = [doc[1] for doc in reranked_docs_and_urls]
    else:
        urls = [doc[1] for doc in top_similar_docs]  # Unpacking URLs

    return (question, url_ending, urls)


def test_retrieved_docs_retrieve_best_url(
    question_doc_pairs: list, use_reranking: bool = False
) -> float:
    """Test if the retrieved documents contain the best URL ending.

    Args:
        question_doc_pairs: List of dictionaries containing questions and URL
            endings.
        use_reranking: Whether to use reranking to improve retrieval.

    Returns:
        The failure rate of the retrieval test.
    """
    total_tests = len(question_doc_pairs)
    failures = 0

    for pair in question_doc_pairs:
        question, url_ending, urls = query_similar_docs(
            pair["question"], pair["url_ending"], use_reranking
        )
        if all(url_ending not in url for url in urls):
            logging.error(
                f"Failed for question: {question}. Expected URL ending: {url_ending}. Got: {urls}"
            )
            failures += 1

    logging.info(f"Total tests: {total_tests}. Failures: {failures}")
    failure_rate = (failures / total_tests) * 100
    return round(failure_rate, 2)


def perform_small_retrieval_evaluation(use_reranking: bool) -> float:
    """Helper function to perform the small retrieval evaluation.

    Args:
        use_reranking: Whether to use reranking in the retrieval process.

    Returns:
        The failure rate of the retrieval test.
    """
    failure_rate = test_retrieved_docs_retrieve_best_url(
        question_doc_pairs, use_reranking
    )
    logging.info(
        f"Retrieval failure rate{' with reranking' if use_reranking else ''}: {failure_rate}%"
    )
    return failure_rate


@step
def retrieval_evaluation_small() -> (
    Annotated[float, "small_failure_rate_retrieval"]
):
    """Executes the retrieval evaluation step without reranking.

    Returns:
        The failure rate of the retrieval test.
    """
    return perform_small_retrieval_evaluation(use_reranking=False)


@step
def retrieval_evaluation_small_with_reranking() -> (
    Annotated[float, "small_failure_rate_retrieval_reranking"]
):
    """Executes the retrieval evaluation step with reranking.

    Returns:
        The failure rate of the retrieval test.
    """
    return perform_small_retrieval_evaluation(use_reranking=True)


def perform_retrieval_evaluation(
    sample_size: int, use_reranking: bool
) -> float:
    """Helper function to perform the retrieval evaluation.

    Args:
        sample_size: Number of samples to use for the evaluation.
        use_reranking: Whether to use reranking in the retrieval process.

    Returns:
        The failure rate of the retrieval test.
    """
    dataset = load_dataset("zenml/rag_qa_embedding_questions", split="train")
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

        _, _, urls = query_similar_docs(question, url_ending, use_reranking)

        if all(url_ending not in url for url in urls):
            logging.error(
                f"Failed for question: {question}. Expected URL ending: {url_ending}. Got: {urls}"
            )
            failures += 1

    logging.info(f"Total tests: {total_tests}. Failures: {failures}")
    failure_rate = (failures / total_tests) * 100
    return round(failure_rate, 2)


@step
def retrieval_evaluation_full(
    sample_size: int = 100,
) -> Annotated[float, "full_failure_rate_retrieval"]:
    """Executes the retrieval evaluation step without reranking.

    Args:
        sample_size: Number of samples to use for the evaluation.

    Returns:
        The failure rate of the retrieval test.
    """
    failure_rate = perform_retrieval_evaluation(
        sample_size, use_reranking=False
    )
    logging.info(f"Retrieval failure rate: {failure_rate}%")
    return failure_rate


@step
def retrieval_evaluation_full_with_reranking(
    sample_size: int = 100,
) -> Annotated[float, "full_failure_rate_retrieval_reranking"]:
    """Executes the retrieval evaluation step with reranking.

    Args:
        sample_size: Number of samples to use for the evaluation.

    Returns:
        The failure rate of the retrieval test.
    """
    failure_rate = perform_retrieval_evaluation(
        sample_size, use_reranking=True
    )
    logging.info(f"Retrieval failure rate with reranking: {failure_rate}%")
    return failure_rate
