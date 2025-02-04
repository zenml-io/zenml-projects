import logging
from typing import Any, Dict, List, Tuple, Optional
from dataclasses import dataclass
import json
import sys
import traceback

from pydantic import BaseModel
from zenml import step
from litellm import completion
import mlflow
from zenml.logger import get_logger

logger = get_logger(__name__)


sys.excepthook = sys.__excepthook__  # Revert to standard tracebacks

SERVICE_CONNECTORS_EVAL_CRITERIA = """
The RAG pipeline sometimes struggles to respond to questions about ZenML's service 
connectors feature. Generally speaking, it should respond in detail with code 
examples when asked about supported service connectors (and should explain how to 
use them together with associated stack components.)
"""

EVALUATION_DATA_PAIRS = [
    {
        "mlflow_experiment_name": "service_connectors",
        "eval_criteria": SERVICE_CONNECTORS_EVAL_CRITERIA,
    }
]


class EvalResult(BaseModel):
    """Pydantic model to capture LLM evaluation outputs."""

    is_good_response: bool
    reasoning: str


@dataclass
class EvaluationInput:
    """Data class to hold all inputs needed for a single evaluation."""

    question: str
    llm_response: str
    eval_criteria: str
    sample_good_response: str
    sample_bad_response: str


def construct_eval_metaprompt(
    eval_criteria: str,
    example_good_response: str = "None provided for this evaluation.",
    example_bad_response: str = "None provided for this evaluation.",
) -> str:
    """Construct a metaprompt for evaluating the performance of an LLM.

    Args:
        eval_criteria (str): The criteria for evaluating the LLM.
        example_good_response (str): An example of a good response.
        example_bad_response (str): An example of a bad response.
    """
    return f"""
# General Instructions

You are an expert at evaluating the performance of a chatbot.

You will be given a question and an LLM-generated output and you should judge
whether this is a good or bad response.

You will also be given a specific area of focus for the evaluation.

# Evaluation Criteria

You should judge the response bearing the following comments in mind:

{eval_criteria}

## A good response

Here is an example of a good response:

{example_good_response}

## A bad response

Here is an example of a bad response:

{example_bad_response}

# Your response

Your response should be a JSON object with the following fields:

- `is_good_response`: (bool) Whether the response is good or bad.
- `reasoning`: (str) A short explanation for your answer.
"""


def build_evaluation_messages(
    eval_input: EvaluationInput,
) -> List[Dict[str, str]]:
    """Construct the messages payload for the LLM evaluation.

    Args:
        eval_input: Container for all evaluation-related inputs

    Returns:
        List of message dictionaries for the LLM API call
    """
    prompt = construct_eval_metaprompt(
        eval_input.eval_criteria,
        eval_input.sample_good_response,
        eval_input.sample_bad_response,
    )

    return [
        {"role": "system", "content": prompt},
        {
            "role": "user",
            "content": f"Question: {eval_input.question}\n\nLLM Response: {eval_input.llm_response}",
        },
    ]


def evaluate_single_response(
    eval_input: EvaluationInput, logger: logging.Logger
) -> Optional[Dict[str, Any]]:
    """Evaluate a single LLM response using the evaluation criteria."""
    try:
        messages = build_evaluation_messages(eval_input)
        logger.debug(
            "Constructed evaluation messages:\n%s",
            json.dumps(messages, indent=2),
        )

        logger.debug(
            "Sending evaluation prompt to LLM for question: %s",
            eval_input.question,
        )

        # Use minimal OpenAI-compatible JSON response format
        response = completion(
            model="gpt-4o",
            messages=messages,
            response_format={"type": "json_object"},
        )

        # Log full response structure for debugging
        logger.debug(
            "Raw LLM response:\n%s", json.dumps(response.dict(), indent=2)
        )

        # Validate and parse the response content
        response_content = response.choices[0].message.content
        evaluation = EvalResult.model_validate_json(response_content)

        result = {
            "question": eval_input.question,
            "evaluation": evaluation.model_dump(),
        }

        logger.debug("Successfully parsed evaluation result")
        logger.debug(
            "Full evaluation result:\n%s", json.dumps(result, indent=2)
        )
        return result

    except Exception as e:
        logger.error(
            "Error during LLM evaluation for question '%s': %s\n%s",
            eval_input.question,
            str(e),
            "Full traceback:\n" + traceback.format_exc(),
        )
        return None


def get_mlflow_dataset(
    mlflow_experiment_name: str,
) -> Tuple[List[Tuple[str, str]], str, str]:
    """Get the MLflow dataset for a given experiment name.

    Args:
        mlflow_experiment_name (str): The name of the MLflow experiment.

    Returns:
        Tuple containing:
        - List of (question, response) tuples from MLflow traces
        - A sample good response for evaluation
        - A sample bad response for evaluation

    Raises:
        ValueError: If no traces are available
    """
    # Get experiment to ensure it exists and get its ID
    experiment = mlflow.get_experiment_by_name(mlflow_experiment_name)
    if not experiment:
        raise ValueError(f"Experiment '{mlflow_experiment_name}' not found")

    logger.info(
        f"Found experiment '{mlflow_experiment_name}' with ID: {experiment.experiment_id}"
    )

    # Get all traces with votes
    traces = mlflow.search_traces(
        experiment_ids=[experiment.experiment_id],
        filter_string="status = 'OK'",
    )

    if traces.empty:
        logger.warning(
            f"No traces found for experiment '{mlflow_experiment_name}'"
        )
        return [], "None provided", "None provided"

    logger.debug(f"Found {len(traces.values)} total traces")

    # Extract question-answer pairs (only from downvoted traces)
    qa_pairs: List[Tuple[str, str]] = []
    good_response = "None provided"
    bad_response = "None provided"

    # Process all traces
    for idx, trace in enumerate(traces.values):
        try:
            logger.debug(f"\nProcessing trace {idx}")

            # The trace metadata is in the last element
            trace_metadata = trace[-1]
            if not isinstance(trace_metadata, dict):
                logger.debug(f"Trace {idx}: last element is not a dict")
                continue

            # The messages data is in element 5
            if len(trace) < 6 or not isinstance(trace[5], dict):
                logger.debug(f"Trace {idx}: invalid structure")
                continue

            messages_data = trace[5]
            vote = trace_metadata.get("vote")
            completion_data = trace[6]  # This contains the actual LLM response

            # Extract question from messages
            question = ""
            for msg in messages_data.get("messages", []):
                if msg.get("role") == "user":
                    question = msg.get("content", "").strip("` \n")
                    break

            # Extract answer from completion choices
            answer = (
                completion_data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                .strip()
            )

            if not question or not answer:
                logger.debug(f"Trace {idx}: missing question/answer")
                continue

            # Format as markdown with headers
            formatted_qa = f"### Question\n{question}\n\n### Answer\n{answer}"

            if vote == "down":
                qa_pairs.append((formatted_qa, "Needs improvement"))
                bad_response = formatted_qa
            elif vote == "up":
                good_response = formatted_qa

        except Exception as e:
            logger.error(f"Error processing trace {idx}: {e}")
            continue

    logger.info(
        f"Found {len(qa_pairs)} question-answer pairs from downvoted traces"
    )
    logger.debug(f"Good response found: {good_response != 'None provided'}")
    logger.debug(f"Bad response found: {bad_response != 'None provided'}")

    if not qa_pairs:
        raise ValueError(
            "No valid question-answer pairs found in downvoted traces"
        )

    return qa_pairs, good_response, bad_response


@step(enable_cache=False)
def fast_eval() -> List[Dict[str, Any]]:
    """
    Step function to evaluate LLM responses by calling LiteLLM in JSON mode.

    Returns:
        List of evaluation results for each question-response pair
    """
    logger = logging.getLogger(__name__)
    results: List[Dict[str, Any]] = []

    for pair in EVALUATION_DATA_PAIRS:
        mlflow_experiment_name = pair["mlflow_experiment_name"]
        eval_criteria = pair["eval_criteria"]

        mlflow_dataset, sample_good_response, sample_bad_response = (
            get_mlflow_dataset(mlflow_experiment_name)
        )

        for question, llm_response in mlflow_dataset:
            eval_input = EvaluationInput(
                question=question,
                llm_response=llm_response,
                eval_criteria=eval_criteria,
                sample_good_response=sample_good_response,
                sample_bad_response=sample_bad_response,
            )

            if result := evaluate_single_response(eval_input, logger):
                results.append(result)

    logger.info("All evaluations completed with %d results", len(results))
    return results
