import mlflow
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from sklearn.base import ClassifierMixin
from typing_extensions import Annotated
from zenml import step, get_step_context, log_metadata
from zenml.client import Client
from zenml.logger import get_logger

logger = get_logger(__name__)

experiment_tracker = Client().active_stack.experiment_tracker


@step(experiment_tracker=experiment_tracker.name)
def model_evaluator(
    model: ClassifierMixin,
    dataset_trn: pd.DataFrame,
    dataset_tst: pd.DataFrame,
    target: str,
    min_train_accuracy: float = 0.0,
    min_test_accuracy: float = 0.0,
    fail_on_accuracy_quality_gates: bool = False,
) -> Annotated[Image.Image, "accuracies_plot"]:
    """Evaluate a trained model and return a focused accuracy plot as PIL.Image.Image."""
    # Calculate accuracies and log metadata (same as before)
    trn_acc = model.score(
        dataset_trn.drop(columns=[target]), dataset_trn[target]
    )
    tst_acc = model.score(
        dataset_tst.drop(columns=[target]), dataset_tst[target]
    )

    logger.info(f"Train accuracy={trn_acc*100:.2f}%")
    logger.info(f"Test accuracy={tst_acc*100:.2f}%")
    mlflow.log_metric("testing_accuracy_score", tst_acc)

    step_context = get_step_context()

    log_metadata(
        metadata={
            "evaluation_metrics": {
                "train_accuracy": trn_acc,
                "test_accuracy": tst_acc,
            }
        },
        infer_model=True
    )

    # Fetch previous versions (same as before)
    client = Client()
    previous_versions = []
    for version in client.get_model(step_context.model.name).versions:
        version_obj = client.get_model_version(
            step_context.model.name, version.version
        )
        if "evaluation_metrics" in version_obj.run_metadata:
            test_accuracy = version_obj.run_metadata["evaluation_metrics"].get("test_accuracy")
            if test_accuracy is not None:
                previous_versions.append(
                    (f"v{version.version}", float(test_accuracy))
                )

    # Sort versions by number
    previous_versions.sort(key=lambda x: int(x[0][1:]))

    # Take up to 5 most recent versions, including the current one
    previous_versions = (
        previous_versions[-5:]
        if len(previous_versions) > 5
        else previous_versions
    )

    # Ensure the current version is included
    current_version_tuple = (f"v{step_context.model.version}", tst_acc)
    if current_version_tuple not in previous_versions:
        previous_versions.append(current_version_tuple)
        previous_versions.sort(key=lambda x: int(x[0][1:]))
        previous_versions = previous_versions[-5:]

    # Create a clean image with transparent background
    img_width, img_height = 1400, 800
    img = Image.new("RGBA", (img_width, img_height), (255, 255, 255, 0))

    draw = ImageDraw.Draw(img)

    # Load fonts (use default if custom font not available)
    try:
        title_font = ImageFont.truetype("Arial.ttf", 40)
        main_font = ImageFont.truetype("Arial.ttf", 30)
        small_font = ImageFont.truetype("Arial.ttf", 20)
    except IOError:
        # Use default font with different sizes
        title_font = ImageFont.load_default()
        main_font = ImageFont.load_default()
        small_font = ImageFont.load_default()

    # Draw title
    title = "Accuracy over Time"
    draw.text(
        (img_width // 2, 30),
        title,
        fill="#000000",
        font=title_font,
        anchor="mt",
    )

    # Draw accuracy history graph
    graph_left, graph_top, graph_right, graph_bottom = 100, 100, 1100, 700
    draw.rectangle(
        [graph_left, graph_top, graph_right, graph_bottom],
        outline="#000000",
        width=2,
    )

    # Add grid
    for i in range(1, 5):
        y = graph_top + (graph_bottom - graph_top) * i / 5
        draw.line([graph_left, y, graph_right, y], fill="#CCCCCC", width=1)

    # Plot points with custom icons and lines
    num_versions = len(previous_versions)
    x_step = (
        (graph_right - graph_left) / (num_versions - 1)
        if num_versions > 1
        else 0
    )

    for i, (version, acc) in enumerate(previous_versions):
        x = graph_left + i * x_step
        y = graph_bottom - (acc * (graph_bottom - graph_top))

        # Custom icon (circle)
        icon_size = 10
        draw.ellipse(
            [x - icon_size, y - icon_size, x + icon_size, y + icon_size],
            fill="#FF6B6B",
            outline="#333333",
        )

        if i > 0:
            prev_x = graph_left + (i - 1) * x_step
            prev_y = graph_bottom - (
                previous_versions[i - 1][1] * (graph_bottom - graph_top)
            )
            draw.line([prev_x, prev_y, x, y], fill="#4ECDC4", width=3)

        draw.text(
            (x, y + 25), version, fill="#333333", font=small_font, anchor="mt"
        )
        draw.text(
            (x, y - 25),
            f"{acc:.2%}",
            fill="#333333",
            font=small_font,
            anchor="mb",
        )

    # Add graph labels
    draw.text(
        (graph_left + (graph_right - graph_left) // 2, graph_bottom + 50),
        "Model Versions",
        fill="#333333",
        font=main_font,
        anchor="mt",
    )

    draw.text(
        (graph_left - 50, graph_top + (graph_bottom - graph_top) // 2),
        "Accuracy",
        fill="#333333",
        font=main_font,
        anchor="mm",
        rotation=90,
    )

    # Apply anti-aliasing
    img = img.resize((img_width, img_height), Image.LANCZOS)

    return img
