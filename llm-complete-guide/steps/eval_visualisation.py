import io
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from zenml import get_step_context, step


@step(enable_cache=False)
def visualize_evaluation_results(
    small_retrieval_eval_failure_rate: float,
    full_retrieval_eval_failure_rate: float,
    failure_rate_bad_answers: float,
    failure_rate_bad_immediate_responses: float,
    failure_rate_good_responses: float,
    average_toxicity_score: float,
    average_faithfulness_score: float,
    average_helpfulness_score: float,
    average_relevance_score: float,
) -> Optional[Image.Image]:
    """Visualizes the evaluation results."""
    step_context = get_step_context()
    pipeline_run_name = step_context.pipeline_run.name

    normalized_scores = [
        score / 20
        for score in [
            small_retrieval_eval_failure_rate,
            full_retrieval_eval_failure_rate,
            failure_rate_bad_answers,
        ]
    ]

    scores = normalized_scores + [
        failure_rate_bad_immediate_responses,
        failure_rate_good_responses,
        average_toxicity_score,
        average_faithfulness_score,
        average_helpfulness_score,
        average_relevance_score,
    ]

    labels = [
        "Small Retrieval Eval Failure Rate",
        "Full Retrieval Eval Failure Rate",
        "Failure Rate Bad Answers",
        "Failure Rate Bad Immediate Responses",
        "Failure Rate Good Responses",
        "Average Toxicity Score",
        "Average Faithfulness Score",
        "Average Helpfulness Score",
        "Average Relevance Score",
    ]

    # Create a new figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the horizontal bar chart
    y_pos = np.arange(len(labels))
    ax.barh(y_pos, scores, align="center")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()  # Labels read top-to-bottom
    ax.set_xlabel("Score")
    ax.set_xlim(0, 5)
    ax.set_title(f"Evaluation Metrics for {pipeline_run_name}")

    # Adjust the layout
    plt.tight_layout()

    # Save the plot to a BytesIO object
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)

    # Create a PIL Image object from the BytesIO object
    image = Image.open(buf)

    return image
