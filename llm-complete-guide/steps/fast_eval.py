import io
import json
import logging
import sys
import traceback
from collections import defaultdict
from dataclasses import dataclass
from typing import Annotated, Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import mlflow
import numpy as np
from litellm import completion
from PIL import Image
from pydantic import BaseModel
from rich import print
from utils.llm_utils import process_input_with_retrieval
from zenml import ArtifactConfig, get_step_context, log_metadata, step
from zenml.logger import get_logger

logger = get_logger(__name__)


sys.excepthook = sys.__excepthook__  # Revert to standard tracebacks

SERVICE_CONNECTORS_EVAL_CRITERIA = """
The RAG pipeline sometimes struggles to respond to questions about ZenML's service 
connectors feature. Generally speaking, it should respond in detail with code 
examples when asked about supported service connectors (and should explain how to 
use them together with associated stack components.) And if someone is asking
about service connectors then generally speaking it shouldn't then go on to
be all about an orchestrator etc instead of focusing on the service connectors.
"""

MISSING_CODE_SAMPLE_EVAL_CRITERIA = """
The RAG pipeline sometimes doesn't include a code sample in the response.
Of course, a code sample doesn't always need to be included, but generally when
a user asks about how to use a feature, it's usually useful to include a code
sample in the response. (Also note that there are detailed documents about
Evidently and so it's not enough to just say that you can implement Evidently as
a custom step.)
"""

EVALUATION_DATA_PAIRS = [
    {
        "mlflow_experiment_name": "service_connectors",
        "eval_criteria": SERVICE_CONNECTORS_EVAL_CRITERIA,
    },
    {
        "mlflow_experiment_name": "missing_code_sample",
        "eval_criteria": MISSING_CODE_SAMPLE_EVAL_CRITERIA,
    },
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

You will be given:
1. A user question
2. An OLD RESPONSE from a previous version of the chatbot
3. A NEW RESPONSE from the current version of the chatbot

Your task is to compare these responses and determine if the NEW RESPONSE is better
than the OLD RESPONSE based on the evaluation criteria below.

# Evaluation Criteria

You should judge whether the NEW RESPONSE is an improvement over the OLD RESPONSE,
considering these specific criteria:

{eval_criteria}

## Example Responses

Here is an example of a good response format:
{example_good_response}

Here is an example of a response that needs improvement:
{example_bad_response}

# Your response

Your response should be a JSON object with the following fields:

- `is_good_response`: (bool) Whether the NEW RESPONSE is an improvement over the OLD RESPONSE.
- `reasoning`: (str) A brief explanation comparing the two responses and justifying your decision.
    Focus on specific improvements or regressions in the NEW RESPONSE.
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
        print(result)

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
) -> Tuple[List[Dict[str, Any]], str, str]:
    """Get the MLflow dataset for a given experiment name.

    Returns:
        Tuple containing:
        - List of dicts with {'question': str, 'old_response': str} from MLflow traces
        - A sample good response for evaluation
        - A sample bad response for evaluation
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
    qa_pairs: List[Dict[str, Any]] = []
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
                qa_pairs.append({"question": question, "old_response": answer})
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
    """Evaluate LLM responses by comparing old vs new responses.

    Returns:
        List of evaluation results comparing old and new responses
    """
    logger = logging.getLogger(__name__)
    results: List[Dict[str, Any]] = []

    for pair in EVALUATION_DATA_PAIRS:
        mlflow_experiment_name = pair["mlflow_experiment_name"]
        eval_criteria = pair["eval_criteria"]

        mlflow_dataset, sample_good_response, sample_bad_response = (
            get_mlflow_dataset(mlflow_experiment_name)
        )

        for data in mlflow_dataset:
            question = data["question"]
            old_response = data["old_response"]

            # Generate new response using current implementation
            try:
                new_response = process_input_with_retrieval(
                    question, mlflow_experiment_name=mlflow_experiment_name
                )
            except Exception as e:
                logger.error(f"Error generating new response: {e}")
                continue

            eval_input = EvaluationInput(
                question=question,
                llm_response=f"OLD RESPONSE:\n{old_response}\n\nNEW RESPONSE:\n{new_response}",
                eval_criteria=eval_criteria,
                sample_good_response=sample_good_response,
                sample_bad_response=sample_bad_response,
            )

            if result := evaluate_single_response(eval_input, logger):
                result.update(
                    {
                        "experiment_name": mlflow_experiment_name,
                        "old_response": old_response,
                        "new_response": new_response,
                    }
                )
                results.append(result)

    logger.info("All evaluations completed with %d results", len(results))
    return results


@step
def visualize_fast_eval_results(
    results: List[Dict[str, Any]],
) -> Annotated[Image.Image, ArtifactConfig(name="fast_eval_metrics")]:
    """Visualize the results of the fast evaluation.

    Args:
        results: List of evaluation results from the fast_eval step, each containing
                experiment_name, question, and evaluation data

    Returns:
        PIL Image showing the evaluation metrics visualization
    """
    step_context = get_step_context()
    pipeline_run_name = step_context.pipeline_run.name

    # Process results to get metrics per experiment
    experiment_metrics = defaultdict(lambda: {"total": 0, "bad": 0})

    for result in results:
        experiment = result["experiment_name"]
        experiment_metrics[experiment]["total"] += 1
        if not result["evaluation"]["is_good_response"]:
            experiment_metrics[experiment]["bad"] += 1

    # Calculate percentages
    percentages = {}
    for exp, metrics in experiment_metrics.items():
        if metrics["total"] > 0:
            percentages[exp] = (metrics["bad"] / metrics["total"]) * 100

            log_metadata(
                metadata={
                    f"{exp}.total_evaluations": metrics["total"],
                    f"{exp}.bad_responses": metrics["bad"],
                    f"{exp}.bad_response_percentage": percentages[exp],
                }
            )

    # Create visualization
    labels = list(percentages.keys())
    scores = list(percentages.values())

    # Create a new figure and axis
    fig, ax = plt.subplots(
        figsize=(12, max(6, len(labels) * 0.5))
    )  # Adjust height based on number of experiments
    fig.subplots_adjust(
        left=0.4
    )  # Adjust left margin for potentially longer experiment names

    # Plot horizontal bar chart
    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, scores, align="center", color="skyblue")

    # Add value labels on the bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        total = experiment_metrics[labels[i]]["total"]
        bad = experiment_metrics[labels[i]]["bad"]
        ax.text(
            width + 1,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.1f}% ({bad}/{total})",
            ha="left",
            va="center",
        )

    # Customize the plot
    ax.set_yticks(y_pos)
    ax.set_yticklabels([name.replace("_", " ").title() for name in labels])
    ax.invert_yaxis()  # Labels read top-to-bottom
    ax.set_xlabel("Percentage of Bad Responses")
    ax.set_title(
        f"Fast Evaluation Results - Bad Response Rate\n{pipeline_run_name}"
    )
    ax.set_xlim(0, 100)  # Set x-axis limits for percentage

    # Add grid for better readability
    ax.grid(True, axis="x", linestyle="--", alpha=0.7)

    # Add a light gray background for better contrast
    ax.set_facecolor("#f8f8f8")

    # Add total evaluations count to the title
    total_evals = sum(
        metrics["total"] for metrics in experiment_metrics.values()
    )
    plt.suptitle(f"Total Evaluations: {total_evals}", y=0.95, fontsize=10)

    # Adjust layout
    plt.tight_layout()

    # Convert plot to PIL Image
    buf = io.BytesIO()
    plt.savefig(
        buf, format="png", bbox_inches="tight", dpi=300, facecolor="white"
    )
    buf.seek(0)
    image = Image.open(buf)

    logger.info("Generated visualization for fast evaluation results")
    return image
