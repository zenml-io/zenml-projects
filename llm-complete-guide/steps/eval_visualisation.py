#  Copyright (c) ZenML GmbH 2024. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at:
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
#  or implied. See the License for the specific language governing
#  permissions and limitations under the License.

import io
from typing import Annotated, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from zenml import ArtifactConfig, get_step_context, log_metadata, step


def create_image(
    labels: list,
    scores: list,
    title: str,
    alternate_colours: bool = False,
    percentage_scale: bool = False,
) -> Image.Image:
    """
    Create a horizontal bar chart image from the given labels, scores, and title.

    Args:
        labels (list): List of labels for the y-axis.
        scores (list): List of scores corresponding to each label.
        title (str): Title of the chart.
        alternate_colours (bool): Whether to alternate colours for the bars.
        percentage_scale (bool): Whether to use a percentage scale (0-100) for the x-axis.

    Returns:
        Image.Image: The generated chart image.
    """
    # Create a new figure and axis with a smaller left margin
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.subplots_adjust(left=0.2)  # Adjust left margin

    # Plot the horizontal bar chart
    y_pos = np.arange(len(labels))
    if alternate_colours:
        colors = ["blue" if i % 2 == 0 else "red" for i in range(len(labels))]
        ax.barh(y_pos, scores, align="center", color=colors)
    else:
        ax.barh(y_pos, scores, align="center")

    # Display the actual value to the left of each bar, or to the right if value is 0
    for i, v in enumerate(scores):
        if v == 0:
            ax.text(
                0.3,  # Position the text label slightly to the right of 0
                i,
                f"{v:.1f}",
                color="black",
                va="center",
                fontweight="bold",
            )
        else:
            colors[i] if alternate_colours else "blue"
            text_color = "white"
            ax.text(
                v
                - 0.1,  # Adjust the x-position of the text labels to the left
                i,
                f"{v:.1f}",
                color=text_color,
                va="center",
                fontweight="bold",
                ha="right",  # Align the text to the right
            )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()  # Labels read top-to-bottom
    ax.set_xlabel("Score")
    if percentage_scale:
        ax.set_xlim(0, 100)  # Set x-axis limits to 0-100 for percentage scale
        ax.set_xlabel("Percentage")
    else:
        ax.set_xlim(0, 5)  # Set x-axis limits based on maximum score
        ax.set_xlabel("Score")

    ax.set_title(title)

    # Adjust the subplot parameters
    plt.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.1)

    # Save the plot to a BytesIO object
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
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
    Visualize the evaluation results by creating three separate images and logging metrics.

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

    # Log all metrics as metadata for dashboard visualization
    log_metadata(
        metadata={
            # Retrieval metrics
            "retrieval.small_failure_rate": small_retrieval_eval_failure_rate,
            "retrieval.small_failure_rate_reranking": small_retrieval_eval_failure_rate_reranking,
            "retrieval.full_failure_rate": full_retrieval_eval_failure_rate,
            "retrieval.full_failure_rate_reranking": full_retrieval_eval_failure_rate_reranking,
            # Generation failure metrics
            "generation.failure_rate_bad_answers": failure_rate_bad_answers,
            "generation.failure_rate_bad_immediate": failure_rate_bad_immediate_responses,
            "generation.failure_rate_good": failure_rate_good_responses,
            # Quality metrics
            "quality.toxicity": average_toxicity_score,
            "quality.faithfulness": average_faithfulness_score,
            "quality.helpfulness": average_helpfulness_score,
            "quality.relevance": average_relevance_score,
            # Composite scores
            "composite.overall_quality": (
                average_faithfulness_score
                + average_helpfulness_score
                + average_relevance_score
            )
            / 3,
            "composite.retrieval_effectiveness": (
                (1 - small_retrieval_eval_failure_rate)
                + (1 - full_retrieval_eval_failure_rate)
            )
            / 2,
        }
    )

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
        alternate_colours=True,
    )
    image2 = create_image(
        image2_labels,
        image2_scores,
        f"Basic Generation Evaluation for {pipeline_run_name}",
        percentage_scale=True,
    )
    image3 = create_image(
        image3_labels,
        image3_scores,
        f"Generation Evaluation (Average Scores for {pipeline_run_name})",
    )

    return image1, image2, image3
