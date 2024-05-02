import io
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from zenml import step
from zenml.client import Client


@step(enable_cache=False)
def visualize_evaluation_results(
    failure_rate_retrieval: float,
    failure_rate_bad_answers: float,
    failure_rate_bad_immediate_responses: float,
    failure_rate_good_responses: float,
    full_failure_rate_retrieval: float,
    e2e_evaluation_llm_judged_results: float,
) -> Optional[Image.Image]:
    """Visualizes the evaluation results."""
    zen_client = Client()
    last_run = zen_client.get_pipeline_run(
        "fbdb9965-b27f-4e76-b656-1e265bcd7aef"
    )
    # try:
    #     last_run = zen_client.get_pipeline("llm_eval").runs
    # except RuntimeError:
    #     return None

    last_run_steps = last_run.steps
    retrieval_results = last_run_steps["retrieval_evaluation"].outputs
    e2e_results = last_run_steps["e2e_evaluation"].outputs

    previous_failure_rate_retrieval = retrieval_results[
        "failure_rate_retrieval"
    ].load()
    previous_failure_rate_bad_answers = e2e_results[
        "failure_rate_bad_answers"
    ].load()
    previous_failure_rate_bad_immediate_responses = e2e_results[
        "failure_rate_bad_immediate_responses"
    ].load()
    previous_failure_rate_good_responses = e2e_results[
        "failure_rate_good_responses"
    ].load()

    # Set up the data
    labels = [
        "Retrieval",
        "Bad Answers",
        "Bad Immediate Responses",
        "Good Responses",
    ]
    previous_values = [
        previous_failure_rate_retrieval,
        previous_failure_rate_bad_answers,
        previous_failure_rate_bad_immediate_responses,
        previous_failure_rate_good_responses,
    ]
    current_values = [
        failure_rate_retrieval,
        failure_rate_bad_answers,
        failure_rate_bad_immediate_responses,
        failure_rate_good_responses,
    ]

    # Set the width of the bars
    bar_width = 0.35

    # Set the positions of the bars on the x-axis
    x = np.arange(len(labels))

    # Create the figure and axes
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create the 'Previous' bars
    ax.bar(
        x - bar_width / 2,
        previous_values,
        width=bar_width,
        label="Previous",
        color="blue",
    )

    # Create the 'Current' bars
    ax.bar(
        x + bar_width / 2,
        current_values,
        width=bar_width,
        label="Current",
        color="green",
    )

    # Add labels and title
    ax.set_xlabel("Metrics")
    ax.set_ylabel("Failure Rate")
    ax.set_title("Evaluation Results Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend()

    # Adjust the layout
    fig.tight_layout()

    # Save the chart to a bytes buffer
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png")
    buffer.seek(0)

    return Image.open(buffer)
