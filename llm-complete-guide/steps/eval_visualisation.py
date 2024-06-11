import io
from typing import Annotated, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from zenml import ArtifactConfig, get_step_context, step


def create_image(labels: list, scores: list, title: str) -> Image.Image:
    """
    Create a horizontal bar chart image from the given labels, scores, and title.

    Args:
        labels (list): List of labels for the y-axis.
        scores (list): List of scores corresponding to each label.
        title (str): Title of the chart.

    Returns:
        Image.Image: The generated chart image.
    """
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
    ax.set_title(title)

    # Adjust the layout
    plt.tight_layout()

    # Save the plot to a BytesIO object
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)

    # Create a PIL Image object from the BytesIO object
    image = Image.open(buf)

    return image


@step(enable_cache=False)
def visualize_evaluation_results(
    small_retrieval_eval_failure_rate: float,
    small_retrieval_eval_failure_rate_reranking: float,
    full_retrieval_eval_failure_rate: float,
    full_retrieval_eval_failure_rate_reranking: float,
    failure_rate_bad_answers: float,
    failure_rate_bad_immediate_responses: float,
    failure_rate_good_responses: float,
    average_toxicity_score: float,
    average_faithfulness_score: float,
    average_helpfulness_score: float,
    average_relevance_score: float,
) -> Tuple[
    Annotated[Image.Image, ArtifactConfig(name="retrieval_eval_metrics")],
    Annotated[Image.Image, ArtifactConfig(name="generation_eval_basic")],
    Annotated[Image.Image, ArtifactConfig(name="generation_eval_full")],
]:
    """
    Visualize the evaluation results by creating three separate images.

    Args:
        small_retrieval_eval_failure_rate (float): Small retrieval evaluation failure rate.
        small_retrieval_eval_failure_rate_reranking (float): Small retrieval evaluation failure rate with reranking.
        full_retrieval_eval_failure_rate (float): Full retrieval evaluation failure rate.
        full_retrieval_eval_failure_rate_reranking (float): Full retrieval evaluation failure rate with reranking.
        failure_rate_bad_answers (float): Failure rate for bad answers.
        failure_rate_bad_immediate_responses (float): Failure rate for bad immediate responses.
        failure_rate_good_responses (float): Failure rate for good responses.
        average_toxicity_score (float): Average toxicity score.
        average_faithfulness_score (float): Average faithfulness score.
        average_helpfulness_score (float): Average helpfulness score.
        average_relevance_score (float): Average relevance score.

    Returns:
        Tuple[Image.Image, Image.Image, Image.Image]: A tuple of three images visualizing the evaluation results.
    """
    step_context = get_step_context()
    pipeline_run_name = step_context.pipeline_run.name

    normalized_scores = [
        score / 20
        for score in [
            small_retrieval_eval_failure_rate,
            small_retrieval_eval_failure_rate_reranking,
            full_retrieval_eval_failure_rate,
            full_retrieval_eval_failure_rate_reranking,
        ]
    ]

    image1_labels = [
        "Small Retrieval Eval Failure Rate",
        "Small Retrieval Eval Failure Rate Reranking",
        "Full Retrieval Eval Failure Rate",
        "Full Retrieval Eval Failure Rate Reranking",
    ]
    image1_scores = normalized_scores

    image2_labels = [
        "Failure Rate Bad Answers",
        "Failure Rate Bad Immediate Responses",
        "Failure Rate Good Responses",
    ]
    image2_scores = [
        failure_rate_bad_answers,
        failure_rate_bad_immediate_responses,
        failure_rate_good_responses,
    ]

    image3_labels = [
        "Average Toxicity Score",
        "Average Faithfulness Score",
        "Average Helpfulness Score",
        "Average Relevance Score",
    ]
    image3_scores = [
        average_toxicity_score,
        average_faithfulness_score,
        average_helpfulness_score,
        average_relevance_score,
    ]

    image1 = create_image(
        image1_labels,
        image1_scores,
        f"Retrieval Evaluation Metrics for {pipeline_run_name}",
    )
    image2 = create_image(
        image2_labels,
        image2_scores,
        f"Basic Generation Evaluation for {pipeline_run_name}",
    )
    image3 = create_image(
        image3_labels,
        image3_scores,
        f"Generation Evaluation (Average Scores for {pipeline_run_name})",
    )

    return image1, image2, image3
