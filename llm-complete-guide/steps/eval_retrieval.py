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

import json
import logging
from functools import partial
from multiprocessing import Pool, cpu_count
from typing import Annotated, Any, Callable, Dict, List, Optional, Tuple

from datasets import load_dataset
from tenacity import retry, stop_after_attempt, wait_exponential
from utils.llm_utils import (
    find_vectorstore_name,
    get_db_conn,
    get_embeddings,
    get_es_client,
    get_topn_similar_docs,
    rerank_documents,
)
from zenml import step
from zenml.logger import get_logger

logger = get_logger(__name__)

# Adjust logging settings as before
logging.basicConfig(
    level=logging.DEBUG,  # Change to DEBUG level
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Only set external loggers to WARNING
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("elasticsearch").setLevel(logging.WARNING)

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
    conn = None
    es_client = None

    vector_store_name = find_vectorstore_name()
    if vector_store_name == "pgvector":
        conn = get_db_conn()
    else:
        es_client = get_es_client()

    num_docs = 20 if use_reranking else returned_sample_size
    # get (content, url) tuples for the top n similar documents
    top_similar_docs = get_topn_similar_docs(
        embedded_question,
        conn=conn,
        es_client=es_client,
        n=num_docs,
        include_metadata=True,
    )

    if use_reranking:
        reranked_docs_and_urls = rerank_documents(question, top_similar_docs)[
            :returned_sample_size
        ]
        urls = [doc[1] for doc in reranked_docs_and_urls]
    else:
        urls = [doc[1] for doc in top_similar_docs]  # Unpacking URLs

    return (question, url_ending, urls)


def process_single_pair(
    pair: Dict, use_reranking: bool = False
) -> Tuple[bool, str, str, List[str]]:
    """Process a single question-document pair.

    Args:
        pair: Dictionary containing question and URL ending
        use_reranking: Whether to use reranking to improve retrieval

    Returns:
        Tuple containing (is_failure, question, url_ending, retrieved_urls)
    """
    question, url_ending, urls = query_similar_docs(
        pair["question"], pair["url_ending"], use_reranking
    )
    is_failure = all(url_ending not in url for url in urls)
    return is_failure, question, url_ending, urls


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    reraise=True,
)
def process_single_pair_with_retry(
    pair: Dict, use_reranking: bool = False
) -> Tuple[bool, str, str, List[str]]:
    """Process a single question-document pair with retry logic.

    Args:
        pair: Dictionary containing question and URL ending
        use_reranking: Whether to use reranking to improve retrieval

    Returns:
        Tuple containing (is_failure, question, url_ending, retrieved_urls)
    """
    try:
        return process_single_pair(pair, use_reranking)
    except Exception as e:
        logging.warning(
            f"Error processing pair {pair['question']}: {str(e)}. Retrying..."
        )
        raise


def process_with_progress(
    items: List, worker_fn: Callable, n_processes: int, desc: str
) -> List:
    """Process items in parallel with progress reporting.

    Args:
        items: List of items to process
        worker_fn: Worker function to apply to each item
        n_processes: Number of processes to use
        desc: Description for the progress bar

    Returns:
        List of results
    """
    logger.info(
        f"{desc} - Starting parallel processing with {n_processes} workers"
    )

    results = []
    with Pool(processes=n_processes) as pool:
        for i, result in enumerate(pool.imap(worker_fn, items), 1):
            results.append(result)
            logger.info(f"Completed {i}/{len(items)} tests")

    logger.info(f"{desc} - Completed processing {len(results)} items")
    return results


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
    logger.info(f"Starting retrieval test with {total_tests} questions...")

    n_processes = max(1, cpu_count() // 2)
    worker = partial(
        process_single_pair_with_retry, use_reranking=use_reranking
    )

    try:
        results = process_with_progress(
            question_doc_pairs,
            worker,
            n_processes,
            "Testing document retrieval",
        )

        failures = 0
        logger.info("\nTest Results:")
        for is_failure, question, url_ending, urls in results:
            if is_failure:
                failures += 1
                logger.error(
                    f"Failed test for question: '{question}'\n"
                    f"Expected URL ending: {url_ending}\n"
                    f"Got URLs: {urls}"
                )
            else:
                logger.info(f"Passed test for question: '{question}'")

        failure_rate = (failures / total_tests) * 100
        logger.info(
            f"\nTest Summary:\n"
            f"Total tests: {total_tests}\n"
            f"Failures: {failures}\n"
            f"Failure rate: {failure_rate}%"
        )
        return round(failure_rate, 2)

    except Exception as e:
        logger.error(f"Error during parallel processing: {str(e)}")
        raise


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
def retrieval_evaluation_small() -> Annotated[
    float, "small_failure_rate_retrieval"
]:
    """Executes the retrieval evaluation step without reranking.

    Returns:
        The failure rate of the retrieval test.
    """
    return perform_small_retrieval_evaluation(use_reranking=False)


@step
def retrieval_evaluation_small_with_reranking() -> Annotated[
    float, "small_failure_rate_retrieval_reranking"
]:
    """Executes the retrieval evaluation step with reranking.

    Returns:
        The failure rate of the retrieval test.
    """
    return perform_small_retrieval_evaluation(use_reranking=True)


def process_single_dataset_item(
    item: Dict, use_reranking: bool = False
) -> Tuple[bool, str, str, List[str]]:
    """Process a single dataset item.

    Args:
        item: Dictionary containing the dataset item with generated questions and filename
        use_reranking: Whether to use reranking to improve retrieval

    Returns:
        Tuple containing (is_failure, question, url_ending, retrieved_urls)
    """
    generated_questions = item["generated_questions"]
    question = generated_questions[0]  # Assuming only one question per item
    url_ending = item["filename"].split("/")[
        -1
    ]  # Extract the URL ending from the filename

    _, _, urls = query_similar_docs(question, url_ending, use_reranking)
    is_failure = all(url_ending not in url for url in urls)
    return is_failure, question, url_ending, urls


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    reraise=True,
)
def process_single_dataset_item_with_retry(
    item: Dict, use_reranking: bool = False
) -> Tuple[bool, str, str, List[str]]:
    """Process a single dataset item with retry logic.

    Args:
        item: Dictionary containing the dataset item
        use_reranking: Whether to use reranking to improve retrieval

    Returns:
        Tuple containing (is_failure, question, url_ending, retrieved_urls)
    """
    try:
        return process_single_dataset_item(item, use_reranking)
    except Exception as e:
        logging.warning(
            f"Error processing dataset item: {str(e)}. Retrying..."
        )
        raise


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
    n_processes = max(1, cpu_count() // 2)
    worker = partial(
        process_single_dataset_item_with_retry, use_reranking=use_reranking
    )

    try:
        results = process_with_progress(
            sampled_dataset, worker, n_processes, "Evaluating retrieval"
        )

        failures = 0
        logger.info("\nTest Results:")
        for is_failure, question, url_ending, urls in results:
            if is_failure:
                failures += 1
                logger.error(
                    f"Failed test for question: '{question}'\n"
                    f"Expected URL containing: {url_ending}\n"
                    f"Got URLs: {urls}"
                )
            else:
                logger.info(f"Passed test for question: '{question}'")

        failure_rate = (failures / total_tests) * 100
        logger.info(
            f"\nTest Summary:\n"
            f"Total tests: {total_tests}\n"
            f"Failures: {failures}\n"
            f"Failure rate: {failure_rate}%"
        )
        return round(failure_rate, 2)

    except Exception as e:
        logger.error(f"Error during parallel processing: {str(e)}")
        raise


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


def process_single_test(
    item: Any,
    test_function: Callable,
) -> Tuple[bool, str, str, str]:
    """Process a single test item.

    Args:
        item: The test item to process
        test_function: The test function to run

    Returns:
        Tuple containing (is_failure, question, keyword, response)
    """
    test_result = test_function(item)
    return (
        not test_result.success,
        test_result.question,
        test_result.keyword,
        test_result.response,
    )


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    reraise=True,
)
def process_single_test_with_retry(
    item: Any,
    test_function: Callable,
) -> Tuple[bool, str, str, str]:
    """Process a single test item with retry logic.

    Args:
        item: The test item to process
        test_function: The test function to run

    Returns:
        Tuple containing (is_failure, question, keyword, response)
    """
    try:
        return process_single_test(item, test_function)
    except Exception as e:
        logging.warning(f"Error processing test item: {str(e)}. Retrying...")
        raise


def run_simple_tests(test_data: list, test_function: Callable) -> float:
    """
    Run tests for bad answers in parallel with progress reporting and error handling.

    Args:
        test_data (list): The test data.
        test_function (function): The test function to run.

    Returns:
        float: The failure rate.
    """
    total_tests = len(test_data)
    n_processes = max(1, cpu_count() // 2)
    worker = partial(
        process_single_test_with_retry, test_function=test_function
    )

    try:
        results = process_with_progress(
            test_data, worker, n_processes, "Running tests"
        )

        failures = 0
        logger.info("\nTest Results:")
        for is_failure, question, keyword, response in results:
            if is_failure:
                failures += 1
                logger.error(
                    f"Failed test for question: '{question}'\n"
                    f"Found word: '{keyword}'\n"
                    f"Response: '{response}'"
                )
            else:
                logger.info(f"Passed test for question: '{question}'")

        failure_rate = (failures / total_tests) * 100
        logger.info(
            f"\nTest Summary:\n"
            f"Total tests: {total_tests}\n"
            f"Failures: {failures}\n"
            f"Failure rate: {failure_rate}%"
        )
        return round(failure_rate, 2)

    except Exception as e:
        logger.error(f"Error during parallel processing: {str(e)}")
        raise


def process_single_llm_test(
    item: Dict,
    test_function: Callable,
) -> Tuple[float, float, float, float]:
    """Process a single LLM test item.

    Args:
        item: Dictionary containing the dataset item
        test_function: The test function to run

    Returns:
        Tuple containing (toxicity, faithfulness, helpfulness, relevance) scores
    """
    # Assuming only one question per item
    question = item["generated_questions"][0]
    context = item["page_content"]

    try:
        result = test_function(question, context)
        return (
            result.toxicity,
            result.faithfulness,
            result.helpfulness,
            result.relevance,
        )
    except json.JSONDecodeError as e:
        logger.error(f"Failed for question: {question}. Error: {e}")
        # Return None to indicate this test should be skipped
        return None


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    reraise=True,
)
def process_single_llm_test_with_retry(
    item: Dict,
    test_function: Callable,
) -> Optional[Tuple[float, float, float, float]]:
    """Process a single LLM test item with retry logic.

    Args:
        item: Dictionary containing the dataset item
        test_function: The test function to run

    Returns:
        Optional tuple containing (toxicity, faithfulness, helpfulness, relevance) scores
        Returns None if the test should be skipped
    """
    try:
        return process_single_llm_test(item, test_function)
    except Exception as e:
        logger.warning(f"Error processing LLM test: {str(e)}. Retrying...")
        raise


def run_llm_judged_tests(
    test_function: Callable,
    sample_size: int = 10,
) -> Tuple[
    Annotated[float, "average_toxicity_score"],
    Annotated[float, "average_faithfulness_score"],
    Annotated[float, "average_helpfulness_score"],
    Annotated[float, "average_relevance_score"],
]:
    """E2E tests judged by an LLM.

    Args:
        test_data (list): The test data.
        test_function (function): The test function to run.
        sample_size (int): The sample size to run the tests on.

    Returns:
        Tuple: The average toxicity, faithfulness, helpfulness, and relevance scores.
    """
    # Load the dataset from the Hugging Face Hub
    dataset = load_dataset("zenml/rag_qa_embedding_questions", split="train")

    # Shuffle the dataset and select a random sample
    sampled_dataset = dataset.shuffle(seed=42).select(range(sample_size))

    n_processes = max(1, cpu_count() // 2)
    worker = partial(
        process_single_llm_test_with_retry, test_function=test_function
    )

    try:
        results = process_with_progress(
            sampled_dataset, worker, n_processes, "Running LLM judged tests"
        )

        # Filter out None results (failed tests)
        valid_results = [r for r in results if r is not None]
        total_tests = len(valid_results)

        if total_tests == 0:
            logger.error("All tests failed!")
            return 0.0, 0.0, 0.0, 0.0

        # Calculate totals
        total_toxicity = sum(r[0] for r in valid_results)
        total_faithfulness = sum(r[1] for r in valid_results)
        total_helpfulness = sum(r[2] for r in valid_results)
        total_relevance = sum(r[3] for r in valid_results)

        # Calculate averages
        average_toxicity_score = total_toxicity / total_tests
        average_faithfulness_score = total_faithfulness / total_tests
        average_helpfulness_score = total_helpfulness / total_tests
        average_relevance_score = total_relevance / total_tests

        logger.info("\nTest Results Summary:")
        logger.info(f"Total valid tests: {total_tests}")
        logger.info(f"Average toxicity: {average_toxicity_score:.3f}")
        logger.info(f"Average faithfulness: {average_faithfulness_score:.3f}")
        logger.info(f"Average helpfulness: {average_helpfulness_score:.3f}")
        logger.info(f"Average relevance: {average_relevance_score:.3f}")

        return (
            round(average_toxicity_score, 3),
            round(average_faithfulness_score, 3),
            round(average_helpfulness_score, 3),
            round(average_relevance_score, 3),
        )

    except Exception as e:
        logger.error(f"Error during parallel processing: {str(e)}")
        raise
