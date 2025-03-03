import concurrent.futures
import io
import json
import logging
import os
import sys
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass
from typing import Annotated, Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from langfuse import Langfuse
from litellm import completion
from PIL import Image
from pydantic import BaseModel
from rich import print
from utils.llm_utils import process_input_with_retrieval
from zenml import ArtifactConfig, get_step_context, log_metadata, step
from zenml.logger import get_logger

logger = get_logger(__name__)

langfuse = Langfuse()

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
        "langfuse_score_identifier": "service_connectors",
        "eval_criteria": SERVICE_CONNECTORS_EVAL_CRITERIA,
    },
    {
        "langfuse_score_identifier": "missing_code_sample",
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


def get_langfuse_scores(
    langfuse_score_identifier: str,
) -> Tuple[List[Dict[str, Any]], str, str]:
    """Get the Langfuse scores for a given score identifier."""
    all_scores = langfuse.api.score.get(name=langfuse_score_identifier).data
    if not all_scores:
        logger.error(
            f"Score with name '{langfuse_score_identifier}' not found"
        )
        return [], "None provided", "None provided"

    logger.info(
        f"Found {len(all_scores)} scores with name '{langfuse_score_identifier}'"
    )

    # Extract question-answer pairs (only from downvoted traces)
    qa_pairs: List[Dict[str, Any]] = []
    good_response = "None provided"
    bad_response = "None provided"

    for score in all_scores:
        associated_trace = langfuse.get_trace(id=score.trace_id)
        question = associated_trace.input["messages"][1]["content"]
        old_response = associated_trace.output["content"]
        qa_pairs.append({"question": question, "old_response": old_response})
        if score.value == 0:
            bad_response = f"Question: {question}\n\nResponse: {old_response}"
        else:
            good_response = f"Question: {question}\n\nResponse: {old_response}"

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


def process_question_answer_pair(
    question: str,
    old_response: str,
    eval_criteria: str,
    sample_good_response: str,
    sample_bad_response: str,
    langfuse_score_identifier: str,
    prompt: str,
    logger: logging.Logger,
) -> Optional[Dict[str, Any]]:
    """Process a single question-answer pair for evaluation.

    Args:
        question: The question to evaluate
        old_response: The old response to compare against
        eval_criteria: The criteria for evaluation
        sample_good_response: An example of a good response
        sample_bad_response: An example of a bad response
        langfuse_score_identifier: The identifier for the Langfuse score
        prompt: The prompt to use for generating the new response
        logger: The logger to use for logging

    Returns:
        Optional dictionary containing the evaluation result
    """
    try:
        # Generate new response using current implementation
        new_response = process_input_with_retrieval(
            input=question,
            prompt=prompt,
            tracing_tags=["evaluation"],
        )

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
                    "experiment_name": langfuse_score_identifier,
                    "old_response": old_response,
                    "new_response": new_response,
                }
            )
            return result
        return None
    except Exception as e:
        logger.error(f"Error processing question-answer pair: {e}")
        return None


# Default to using 4 workers for parallel processing, but allow overriding via environment variable
DEFAULT_MAX_WORKERS = 5
MAX_WORKERS = int(os.environ.get("EVAL_MAX_WORKERS", DEFAULT_MAX_WORKERS))
# Default to using 2 workers for question-answer pairs, but allow overriding via environment variable
QA_MAX_WORKERS = int(os.environ.get("EVAL_QA_MAX_WORKERS", 4))
# Default timeouts (in seconds)
EVAL_PAIR_TIMEOUT = int(os.environ.get("EVAL_PAIR_TIMEOUT", 600))  # 10 minutes
QA_PAIR_TIMEOUT = int(os.environ.get("QA_PAIR_TIMEOUT", 300))  # 5 minutes


class ProgressTracker:
    """A simple class to track progress of parallel evaluations."""

    def __init__(self, total_pairs: int, total_qa_pairs: Dict[str, int]):
        self.total_pairs = total_pairs
        self.total_qa_pairs = total_qa_pairs
        self.completed_pairs = 0
        self.completed_qa_pairs = defaultdict(int)
        self.start_time = time.time()
        self.pair_start_times = {}

    def start_pair(self, pair_id: str):
        """Mark the start of processing an evaluation pair."""
        self.pair_start_times[pair_id] = time.time()

    def complete_pair(self, pair_id: str, results_count: int):
        """Mark the completion of processing an evaluation pair."""
        self.completed_pairs += 1
        duration = time.time() - self.pair_start_times.get(
            pair_id, self.start_time
        )
        return (
            f"Completed evaluation pair '{pair_id}' with {results_count} results "
            f"({self.completed_pairs}/{self.total_pairs}, {duration:.1f}s)"
        )

    def complete_qa(self, pair_id: str):
        """Mark the completion of processing a question-answer pair."""
        self.completed_qa_pairs[pair_id] += 1
        total = self.total_qa_pairs.get(pair_id, 0)
        completed = self.completed_qa_pairs[pair_id]
        return f"Progress for '{pair_id}': {completed}/{total} ({completed / total * 100:.1f}%)"

    def get_overall_progress(self) -> str:
        """Get the overall progress of the evaluation."""
        elapsed = time.time() - self.start_time
        if self.completed_pairs == 0:
            eta = "unknown"
        else:
            avg_time_per_pair = elapsed / self.completed_pairs
            remaining_pairs = self.total_pairs - self.completed_pairs
            eta_seconds = avg_time_per_pair * remaining_pairs
            eta = f"{eta_seconds:.1f}s"

        return (
            f"Overall progress: {self.completed_pairs}/{self.total_pairs} pairs "
            f"({self.completed_pairs / self.total_pairs * 100:.1f}%), "
            f"elapsed: {elapsed:.1f}s, ETA: {eta}"
        )


def process_evaluation_pair(
    pair: Dict[str, str],
    prompt: str,
    logger: logging.Logger,
    progress_tracker: Optional[ProgressTracker] = None,
) -> List[Dict[str, Any]]:
    """Process a single evaluation pair.

    Args:
        pair: The evaluation pair to process
        prompt: The prompt to use for generating the new response
        logger: The logger to use for logging
        progress_tracker: Optional progress tracker

    Returns:
        List of evaluation results
    """
    results = []
    langfuse_score_identifier = pair["langfuse_score_identifier"]
    eval_criteria = pair["eval_criteria"]

    try:
        langfuse_score_data, sample_good_response, sample_bad_response = (
            get_langfuse_scores(langfuse_score_identifier)
        )

        if progress_tracker:
            progress_tracker.start_pair(langfuse_score_identifier)

        logger.info(
            f"Processing {len(langfuse_score_data)} question-answer pairs for '{langfuse_score_identifier}' with {QA_MAX_WORKERS} workers"
        )

        # Process each question-answer pair in parallel
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=QA_MAX_WORKERS
        ) as executor:
            futures = {}
            for data in langfuse_score_data:
                question = data["question"]
                old_response = data["old_response"]

                future = executor.submit(
                    process_question_answer_pair,
                    question,
                    old_response,
                    eval_criteria,
                    sample_good_response,
                    sample_bad_response,
                    langfuse_score_identifier,
                    prompt,
                    logger,
                )
                futures[future] = question

            # Collect results as they complete
            for future in concurrent.futures.as_completed(futures):
                question = futures[future]
                try:
                    # Add timeout to prevent hanging on a single evaluation
                    result = future.result(timeout=QA_PAIR_TIMEOUT)
                    if result:
                        logger.info(
                            f"Completed evaluation for question: '{question[:50]}...'"
                        )
                        results.append(result)
                    else:
                        logger.warning(
                            f"No result for question: '{question[:50]}...'"
                        )
                except concurrent.futures.TimeoutError:
                    logger.error(
                        f"Timeout processing question '{question[:50]}...' after {QA_PAIR_TIMEOUT} seconds"
                    )
                except Exception as e:
                    logger.error(
                        f"Error processing question '{question[:50]}...': {e}"
                    )

                # Update progress
                if progress_tracker:
                    progress_msg = progress_tracker.complete_qa(
                        langfuse_score_identifier
                    )
                    logger.info(progress_msg)
    except Exception as e:
        logger.error(
            f"Error processing evaluation pair '{langfuse_score_identifier}': {e}"
        )

    logger.info(
        f"Completed processing '{langfuse_score_identifier}' with {len(results)} results"
    )
    return results


@step(enable_cache=False)
def fast_eval(prompt: str) -> List[Dict[str, Any]]:
    """Evaluate LLM responses by comparing old vs new responses in parallel.

    Args:
        prompt: The prompt to use for generating the new response

    Returns:
        List of evaluation results comparing old and new responses
    """
    logger = logging.getLogger(__name__)
    all_results = []

    # Initialize progress tracking
    total_qa_pairs = {}
    for pair in EVALUATION_DATA_PAIRS:
        try:
            langfuse_score_data, _, _ = get_langfuse_scores(
                pair["langfuse_score_identifier"]
            )
            total_qa_pairs[pair["langfuse_score_identifier"]] = len(
                langfuse_score_data
            )
        except Exception:
            total_qa_pairs[pair["langfuse_score_identifier"]] = 0

    progress_tracker = ProgressTracker(
        len(EVALUATION_DATA_PAIRS), total_qa_pairs
    )

    logger.info(
        f"Starting parallel evaluation with {MAX_WORKERS} workers for {len(EVALUATION_DATA_PAIRS)} evaluation pairs"
    )

    # Process each evaluation pair in parallel
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=MAX_WORKERS
    ) as executor:
        futures = {}
        for pair in EVALUATION_DATA_PAIRS:
            future = executor.submit(
                process_evaluation_pair, pair, prompt, logger, progress_tracker
            )
            futures[future] = pair["langfuse_score_identifier"]

        # Collect results as they complete
        for future in concurrent.futures.as_completed(futures):
            identifier = futures[future]
            try:
                # Add timeout to prevent hanging on a single evaluation pair
                results = future.result(timeout=EVAL_PAIR_TIMEOUT)
                progress_msg = progress_tracker.complete_pair(
                    identifier, len(results)
                )
                logger.info(progress_msg)
                logger.info(progress_tracker.get_overall_progress())
                all_results.extend(results)
            except concurrent.futures.TimeoutError:
                logger.error(
                    f"Timeout processing evaluation pair '{identifier}' after {EVAL_PAIR_TIMEOUT} seconds"
                )
            except Exception as e:
                logger.error(
                    f"Error processing evaluation pair '{identifier}': {e}"
                )

    total_time = time.time() - progress_tracker.start_time
    logger.info(
        f"All evaluations completed with {len(all_results)} total results in {total_time:.1f} seconds"
    )
    return all_results


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
