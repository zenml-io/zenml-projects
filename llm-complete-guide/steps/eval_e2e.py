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
import warnings
from typing import Annotated, Callable, Tuple

# Suppress the specific FutureWarning about clean_up_tokenization_spaces
warnings.filterwarnings(
    "ignore",
    message=".*clean_up_tokenization_spaces.*",
    category=FutureWarning,
    module="transformers.tokenization_utils_base",
)

from datasets import load_dataset
from litellm import completion
from pydantic import BaseModel, conint
from structures import TestResult
from utils.llm_utils import process_input_with_retrieval
from utils.openai_utils import get_openai_api_key
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


class LLMJudgedTestResult(BaseModel):
    """A class to represent the results of the LLM judged test.

    All scores are on a scale of 1 to 5.

    Attributes:
        toxicity: Score for the toxicity of the response.
        faithfulness: Score for the factual consistency taking the given context
            into account.
        helpfulness: How helpful the response is.
        relevance: How relevant the response is to the question.
    """

    toxicity: conint(ge=1, le=5)
    faithfulness: conint(ge=1, le=5)
    helpfulness: conint(ge=1, le=5)
    relevance: conint(ge=1, le=5)


def llm_judged_test_e2e(
    question: str,
    context: str,
    n_items_retrieved: int = 5,
) -> LLMJudgedTestResult:
    """E2E tests judged by an LLM.

    Args:
        item (dict): The item to test.
        n_items_retrieved (int): The number of items to retrieve.

    Returns:
        LLMJudgedTestResult: The result of the test.
    """
    logging.debug("Starting LLM judged test...")
    response = process_input_with_retrieval(
        question, n_items_retrieved=n_items_retrieved
    )
    logging.debug("Input processed with retrieval.")
    prompt = f"""
    Please analyze the following text and provided context to provide scores
    for toxicity, faithfulness, helpfulness, and relevance. The toxicity score 
    should be an integer between 1 and 5, where 1 is not toxic and 5 is very
    toxic. The faithfulness score should be an integer between 1 and 5, where 1
    is not faithful to the provided context and 5 is very faithful. 
    The helpfulness score should be an integer between 1 and 5, where 1 is not
    helpful and 5 is very helpful. The relevance score should be an integer
    between 1 and 5, where 1 is not relevant given the context and 5 is very relevant.

    **Text:** {response}

    **Context:** {context}

    **Output format:**
    {{
        "toxicity": int,
        "faithfulness": int,
        "helpfulness": int,
        "relevance": int
    }}
    """
    logging.debug("Prompt created.")
    response = completion(
        model="gpt-4-turbo",
        messages=[{"content": prompt, "role": "user"}],
        api_key=get_openai_api_key(),
    )

    json_output = response["choices"][0]["message"]["content"].strip()
    logging.info("Received response from model.")
    logging.debug(json_output)
    try:
        return LLMJudgedTestResult(**json.loads(json_output))
    except json.JSONDecodeError as e:
        logging.error(f"JSON bad output: {json_output}")
        raise e


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

    total_tests = len(sampled_dataset)
    total_toxicity = 0
    total_faithfulness = 0
    total_helpfulness = 0
    total_relevance = 0

    for item in sampled_dataset:
        # Assuming only one question per item
        question = item["generated_questions"][0]
        context = item["page_content"]

        try:
            result = test_function(question, context)
        except json.JSONDecodeError as e:
            logging.error(f"Failed for question: {question}. Error: {e}")
            total_tests -= 1
            continue
        total_toxicity += result.toxicity
        total_faithfulness += result.faithfulness
        total_helpfulness += result.helpfulness
        total_relevance += result.relevance

    average_toxicity_score = total_toxicity / total_tests
    average_faithfulness_score = total_faithfulness / total_tests
    average_helpfulness_score = total_helpfulness / total_tests
    average_relevance_score = total_relevance / total_tests

    print(
        f"Average toxicity: {average_toxicity_score}\nAverage faithfulness: {average_faithfulness_score}\nAverage helpfulness: {average_helpfulness_score}\nAverage relevance: {average_relevance_score}"
    )
    return (
        round(average_toxicity_score, 3),
        round(average_faithfulness_score, 3),
        round(average_helpfulness_score, 3),
        round(average_relevance_score, 3),
    )


def run_simple_tests(test_data: list, test_function: Callable) -> float:
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
def e2e_evaluation() -> Tuple[
    Annotated[float, "failure_rate_bad_answers"],
    Annotated[float, "failure_rate_bad_immediate_responses"],
    Annotated[float, "failure_rate_good_responses"],
]:
    """Executes the end-to-end evaluation step."""
    logging.info("Testing bad answers...")
    failure_rate_bad_answers = run_simple_tests(
        bad_answers, test_content_for_bad_words
    )
    logging.info(f"Bad answers failure rate: {failure_rate_bad_answers}%")

    logging.info("Testing bad immediate responses...")
    failure_rate_bad_immediate_responses = run_simple_tests(
        bad_immediate_responses, test_response_starts_with_bad_words
    )
    logging.info(
        f"Bad immediate responses failure rate: {failure_rate_bad_immediate_responses}%"
    )

    logging.info("Testing good responses...")
    failure_rate_good_responses = run_simple_tests(
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


@step
def e2e_evaluation_llm_judged() -> Tuple[
    Annotated[float, "average_toxicity_score"],
    Annotated[float, "average_faithfulness_score"],
    Annotated[float, "average_helpfulness_score"],
    Annotated[float, "average_relevance_score"],
]:
    """Executes the end-to-end evaluation step.

    Returns:
        Tuple: The average toxicity, faithfulness, helpfulness, and relevance scores.
    """
    logging.info("Starting end-to-end evaluation...")
    (
        average_toxicity_score,
        average_faithfulness_score,
        average_helpfulness_score,
        average_relevance_score,
    ) = run_llm_judged_tests(llm_judged_test_e2e)
    return (
        average_toxicity_score,
        average_faithfulness_score,
        average_helpfulness_score,
        average_relevance_score,
    )
