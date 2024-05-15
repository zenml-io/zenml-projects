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

import io
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import PIL
from zenml.logger import get_logger

logger = get_logger(__name__)


def create_comparison_chart(
    labels: List[str],
    pretrained_similarity: float,
    finetuned_similarity: float,
) -> PIL.Image.Image:
    """Create a horizontal bar chart for model comparison, with pretrained and finetuned similarities represented as percentages.

    Args:
        labels: A list of labels for the chart, specifically ["Pretrained", "Finetuned"].
        pretrained_similarity: The similarity score for the pretrained model, assumed to be in the range 0-1.
        finetuned_similarity: The similarity score for the finetuned model, assumed to be in the range 0-1.

    Returns:
        A PIL Image object of the chart.
    """
    # Convert scores to percentages
    print(f"pretrained_similarity: {pretrained_similarity}, finetuned_similarity: {finetuned_similarity}")
    scores_percent = [pretrained_similarity * 100, finetuned_similarity * 100]

    _, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(labels))

    # Define colors for each bar
    colors = ["blue", "red"]  # blue for Pretrained, red for Finetuned

    bars = ax.barh(y_pos, scores_percent, align="center", color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel("Average Cosine Similarity (%)")
    ax.set_title("Model Comparison on Test Set")

    # Add the data labels to each bar, positioned inside the bar near the right edge
    for bar in bars:
        width = bar.get_width()
        label_x_pos = (
            width - 3
        )  # position the text inside the bar near the right edge
        ax.text(
            label_x_pos,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.1f}%",
            va="center",
            ha="right",  # align text to the right for better visibility
            color="white",  # use a contrasting text color for readability
        )

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    return PIL.Image.open(buf)
