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
from typing import Annotated, Callable

from pydantic import BaseModel
from utils.llm_utils import process_input_with_retrieval
from zenml import step

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


class TestResult(BaseModel):
    success: bool
    question: str
    keyword: str = ""
    response: str


def test_content_for_bad_words(
    item: dict, n_items_retrieved: int = 5
) -> TestResult:
    """
    Test if responses contain bad words.

    Args:
        item (dict): The item to test.
        n_items_retrieved (int): The number of items to retrieve.

    Returns:
        TestResult: A TestResult object containing the test result information.
    """
    question = item["question"]
    bad_words = item["bad_words"]
    response = process_input_with_retrieval(
        question, n_items_retrieved=n_items_retrieved
    )
    for word in bad_words:
        if word in response:
            return TestResult(
                success=False,
                question=question,
                keyword=word,
                response=response,
            )
    return TestResult(success=True, question=question, response=response)


def test_response_starts_with_bad_words(
    item: dict, n_items_retrieved: int = 5
) -> TestResult:
    """
    Test if responses improperly start with bad words.

    Args:
        item (dict): The item to test.
        n_items_retrieved (int): The number of items to retrieve.

    Returns:
        TestResult: A TestResult object containing the test result information.
    """
    question = item["question"]
    bad_words = item["bad_words"]
    response = process_input_with_retrieval(
        question, n_items_retrieved=n_items_retrieved
    )
    for word in bad_words:
        if response.lower().startswith(word.lower()):
            return TestResult(
                success=False,
                question=question,
                keyword=word,
                response=response,
            )
    return TestResult(success=True, question=question, response=response)


def test_content_contains_good_words(
    item: dict, n_items_retrieved: int = 5
) -> TestResult:
    """
    Test if responses properly contain good words.

    Args:
        item (dict): The item to test, containing the question and expected good words.
        n_items_retrieved (int): The number of items to retrieve, defaulted to 5.

    Returns:
        TestResult: A TestResult object containing the test result information.
    """
    question = item["question"]
    good_words = item["good_words"]
    response = process_input_with_retrieval(
        question, n_items_retrieved=n_items_retrieved
    )
    for word in good_words:
        if word not in response:
            return TestResult(
                success=False,
                question=question,
                keyword=word,
                response=response,
            )
    return TestResult(success=True, question=question, response=response)


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
    for item in test_data:
        test_result = test_function(item)
        if not test_result.success:
            logging.error(
                f"Test failed for question: '{test_result.question}'. Found word: '{test_result.keyword}'. Response: '{test_result.response}'"
            )
            failures += 1
    failure_rate = (failures / total_tests) * 100
    logging.info(
        f"Total tests: {total_tests}. Failures: {failures}. Failure rate: {failure_rate}%"
    )
    return round(failure_rate, 2)


@step
def e2e_evaluation() -> (
    Annotated[float, "failure_rate_bad_answers"],
    Annotated[float, "failure_rate_bad_immediate_responses"],
    Annotated[float, "failure_rate_good_responses"],
):
    """Executes the end-to-end evaluation step."""
    logging.info("Testing bad answers...")
    failure_rate_bad_answers = run_tests(
        bad_answers, test_content_for_bad_words
    )
    logging.info(f"Bad answers failure rate: {failure_rate_bad_answers}%")

    logging.info("Testing bad immediate responses...")
    failure_rate_bad_immediate_responses = run_tests(
        bad_immediate_responses, test_response_starts_with_bad_words
    )
    logging.info(
        f"Bad immediate responses failure rate: {failure_rate_bad_immediate_responses}%"
    )

    logging.info("Testing good responses...")
    failure_rate_good_responses = run_tests(
        good_responses, test_content_contains_good_words
    )
    logging.info(
        f"Good responses failure rate: {failure_rate_good_responses}%"
    )
    return (
        failure_rate_bad_answers,
        failure_rate_bad_immediate_responses,
        failure_rate_good_responses,
    )
