import numpy as np
from matplotlib import pyplot as plt
from zenml.client import Client


def main():
    client = Client()

    model_versions = client.list_model_versions(model_name_or_id="breast_cancer_classifier", size=30, hydrate=True)

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

    generate_plot(alpha_values, losses, penalties, test_accuracies)


def generate_plot(alpha_values, losses, penalties, test_accuracies):
    # Convert losses and penalties to numerical indices
    unique_losses = list(set(losses))
    unique_penalties = list(set(penalties))

    loss_indices = [unique_losses.index(loss) for loss in losses]
    penalty_indices = [unique_penalties.index(penalty) for penalty in penalties]

    # Create a figure and a 3D axis
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Create a scatter plot
    scatter = ax.scatter(alpha_values, loss_indices, penalty_indices, c=test_accuracies, cmap='viridis')

    # Set labels for each axis
    ax.set_xlabel('Alpha')
    ax.set_ylabel('Loss')
    ax.set_zlabel('Penalty')

    # Set custom ticks for loss and penalty axes
    ax.set_yticks(range(len(unique_losses)))
    ax.set_yticklabels(unique_losses)
    ax.set_zticks(range(len(unique_penalties)))
    ax.set_zticklabels(unique_penalties)

    # Add a color bar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Accuracy')

    # Set a title
    plt.title('Accuracy vs. Alpha, Loss, and Penalty')

    # Adjust the viewing angle
    ax.view_init(elev=20, azim=45)

    # Show the plot
    plt.tight_layout()
    plt.show()
    return



if __name__ == "__main__":
    main()
