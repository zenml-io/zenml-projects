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
#
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from zenml.client import Client


def main():
    client = Client()

    model_versions = client.list_model_versions(
        model_name_or_id="breast_cancer_classifier", size=27, hydrate=True
    )

    alpha_values = []
    losses = []
    penalties = []
    test_accuracies = []
    train_accuracies = []

    for model_version in model_versions:
        mv_metadata = model_version.run_metadata

        alpha_values.append(mv_metadata.get("alpha_value", None).value)
        losses.append(mv_metadata.get("loss", None).value)
        penalties.append(mv_metadata.get("penalty", None).value)
        test_accuracies.append(mv_metadata.get("test_accuracy", None).value)
        train_accuracies.append(mv_metadata.get("train_accuracy", None).value)

    generate_3d_plot(alpha_values, losses, penalties, test_accuracies)
    generate_2d_plots(alpha_values, losses, penalties, test_accuracies)


def generate_2d_plots(alpha_values, losses, penalties, test_accuracies):
    # Convert the data into a DataFrame
    df = pd.DataFrame(
        {
            "Alpha": alpha_values,
            "Loss": losses,
            "Penalty": penalties,
            "Accuracy": test_accuracies,
        }
    )

    # Get unique values
    unique_penalties = df["Penalty"].unique()

    # Create a figure with subplots for each penalty
    fig, axes = plt.subplots(
        1, len(unique_penalties), figsize=(20, 6), sharey=True
    )
    fig.suptitle("Accuracy Heatmap for Different Penalties", fontsize=16)

    for i, penalty in enumerate(unique_penalties):
        # Filter data for the current penalty
        df_penalty = df[df["Penalty"] == penalty]

        # Create a pivot table
        pivot = df_penalty.pivot(
            index="Loss", columns="Alpha", values="Accuracy"
        )

        # Create heatmap
        sns.heatmap(
            pivot,
            ax=axes[i],
            cmap="viridis",
            annot=True,
            fmt=".3f",
            cbar=False,
        )

        axes[i].set_title(f"Penalty: {penalty}")
        axes[i].set_xlabel("Alpha")

        if i == 0:
            axes[i].set_ylabel("Loss")

    # Add a colorbar to the right of the subplots
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(axes[0].collections[0], cax=cbar_ax, label="Accuracy")

    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.show()


def generate_3d_plot(alpha_values, losses, penalties, test_accuracies):
    # Convert losses and penalties to numerical indices
    unique_losses = list(set(losses))
    unique_penalties = list(set(penalties))

    loss_indices = [unique_losses.index(loss) for loss in losses]
    penalty_indices = [
        unique_penalties.index(penalty) for penalty in penalties
    ]

    # Create a figure and a 3D axis
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Create a scatter plot
    scatter = ax.scatter(
        alpha_values,
        loss_indices,
        penalty_indices,
        c=test_accuracies,
        cmap="viridis",
    )
    # Find the point with the highest accuracy
    max_accuracy_index = np.argmax(test_accuracies)
    max_accuracy = test_accuracies[max_accuracy_index]
    max_alpha = alpha_values[max_accuracy_index]
    max_loss = losses[max_accuracy_index]
    max_penalty = penalties[max_accuracy_index]

    # Highlight the point with the highest accuracy
    ax.scatter(
        [max_alpha],
        [loss_indices[max_accuracy_index]],
        [penalty_indices[max_accuracy_index]],
        c="red",
        s=100,
        edgecolors="black",
        linewidths=2,
        zorder=10,
    )

    # Set labels for each axis
    ax.set_xlabel("Alpha")
    ax.set_ylabel("Loss")
    ax.set_zlabel("Penalty")

    # Set custom ticks for loss and penalty axes
    ax.set_yticks(range(len(unique_losses)))
    ax.set_yticklabels(unique_losses)
    ax.set_zticks(range(len(unique_penalties)))
    ax.set_zticklabels(unique_penalties)

    # Add a color bar
    cbar = plt.colorbar(scatter)
    cbar.set_label("Accuracy")

    # Set a title
    plt.title("Accuracy vs. Alpha, Loss, and Penalty")

    # Adjust the viewing angle
    ax.view_init(elev=20, azim=45)

    # Add legend with highest accuracy point description
    legend_text = f"Highest Accuracy:\nAccuracy: {max_accuracy:.4f}\nAlpha: {max_alpha}\nLoss: {max_loss}\nPenalty: {max_penalty}"
    ax.text2D(
        0.05,
        0.95,
        legend_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # Show the plot
    plt.tight_layout()
    plt.show()
    return


if __name__ == "__main__":
    main()
