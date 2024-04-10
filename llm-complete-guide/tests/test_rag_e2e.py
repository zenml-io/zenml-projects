import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

from utils.llm_utils import process_input_with_retrieval

# Configure the logging level for the root logger
logging.getLogger().setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

bad_answers = [
    {
        "question": "What orchestrators does ZenML support?",
        "bad_words": ["AWS Step Functions", "Flyte", "Prefect", "Dagster"],
    },
    {
        "question": "What is the default orchestrator in ZenML?",
        "bad_words": ["Flyte", "AWS Step Functions"],
    },
]

bad_immediate_responses = [
    {
        "question": "Does ZenML support the Flyte orchestrator out of the box?",
        "bad_words": ["Yes"],
    },
]

good_responses = [
    {
        "question": "What are the supported orchestrators in ZenML? Please list as many of the supported ones as possible.",
        "good_words": ["Kubeflow", "Airflow"],
    },
    {
        "question": "What is the default orchestrator in ZenML?",
        "good_words": ["local"],
    },
]


def test_content_for_bad_words(
    item: dict, n_items_retrieved: int = 5
) -> tuple:
    """
    Test if responses contain bad words.

    Args:
        item (dict): The item to test.
        n_items_retrieved (int): The number of items to retrieve.

    Returns:
        tuple: A tuple containing the success status, the question, the bad word found, and the response.
    """
    question = item["question"]
    bad_words = item["bad_words"]
    response = process_input_with_retrieval(
        question, n_items_retrieved=n_items_retrieved
    )
    for word in bad_words:
        if word in response:
            return (False, question, word, response)
    return (True, question, None, response)


def test_response_starts_with_bad_words(
    item: dict, n_items_retrieved: int = 5
) -> tuple:
    """
    Test if responses improperly start with bad words.

    Args:
        item (dict): The item to test.
        n_items_retrieved (int): The number of items to retrieve.

    Returns:
        tuple: A tuple containing the success status, the question, the bad word found, and the response.
    """
    question = item["question"]
    bad_words = item["bad_words"]
    response = process_input_with_retrieval(
        question, n_items_retrieved=n_items_retrieved
    )
    for word in bad_words:
        if response.lower().startswith(word.lower()):
            return (False, question, word, response)
    return (True, question, None, response)


def test_content_contains_good_words(
    item: dict, n_items_retrieved: int = 5
) -> tuple:
    """
    Test if responses properly contain good words.

    Args:
        item (dict): The item to test, containing the question and expected good words.
        n_items_retrieved (int): The number of items to retrieve, defaulted to 5.

    Returns:
        tuple: A tuple containing the success status, question, the good word not found (if any), and the response.
    """
    question = item["question"]
    good_words = item["good_words"]
    response = process_input_with_retrieval(
        question, n_items_retrieved=n_items_retrieved
    )
    for word in good_words:
        if word not in response:
            return (False, question, word, response)
    return (True, question, None, response)


def run_tests(test_data: list, test_function: Callable) -> float:
    """
    Run tests for bad answers.

    Args:
        test_data (list): The test data.
        test_function (function): The test function to run.

    Returns:
        float: The failure rate.
    """
    failures = 0
    total_tests = len(test_data)
    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_test = {
            executor.submit(test_function, item): item for item in test_data
        }
        for future in as_completed(future_to_test):
            success, question, keyword_query_term, response = future.result()
            if not success:
                logging.error(
                    f"Test failed for question: '{question}'. Found word: '{keyword_query_term}'. Response: '{response}'"
                )
                failures += 1
    failure_rate = (failures / total_tests) * 100
    logging.info(
        f"Total tests: {total_tests}. Failures: {failures}. Failure rate: {failure_rate}%"
    )
    return round(failure_rate, 2)


# Then, you integrate it into the main execution flow similar to other tests:
if __name__ == "__main__":
    print("Testing bad answers...")
    failure_rate_bad_answers = run_tests(
        bad_answers, test_content_for_bad_words
    )
    print(f"Bad answers failure rate: {failure_rate_bad_answers}%")

    print("Testing bad immediate responses...")
    failure_rate_bad_immediate_responses = run_tests(
        bad_immediate_responses, test_response_starts_with_bad_words
    )
    print(
        f"Bad immediate responses failure rate: {failure_rate_bad_immediate_responses}%"
    )

    print("Testing good responses...")
    failure_rate_good_responses = run_tests(
        good_responses, test_content_contains_good_words
    )
    print(f"Good responses failure rate: {failure_rate_good_responses}%")
