from typing_extensions import Annotated
import mlflow
import pandas as pd
from sklearn.base import ClassifierMixin
from zenml import step, get_step_context, log_model_metadata
from zenml.client import Client
from zenml.logger import get_logger
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np
import io

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
    """Evaluate a trained model and return a visually stunning accuracy plot as PIL.Image.Image."""
    # Calculate accuracies and log metadata (same as before)
    trn_acc = model.score(dataset_trn.drop(columns=[target]), dataset_trn[target])
    tst_acc = model.score(dataset_tst.drop(columns=[target]), dataset_tst[target])
    
    logger.info(f"Train accuracy={trn_acc*100:.2f}%")
    logger.info(f"Test accuracy={tst_acc*100:.2f}%")
    mlflow.log_metric("testing_accuracy_score", tst_acc)

    step_context = get_step_context()

    log_model_metadata(
        metadata={
            "evaluation_metrics": {
                "train_accuracy": trn_acc,
                "test_accuracy": tst_acc
            }
        },
    )

    # Fetch previous versions (same as before)
    client = Client()
    previous_versions = []    
    for version in client.get_model(step_context.model.name).versions:
        version_obj = client.get_model_version(step_context.model.name, version.version)
        if "evaluation_metrics" in version_obj.run_metadata:
            test_accuracy = version_obj.run_metadata["evaluation_metrics"].value.get("test_accuracy")
            if test_accuracy is not None:
                previous_versions.append((f"v{version.version}", float(test_accuracy)))

    # Sort versions by number
    previous_versions.sort(key=lambda x: int(x[0][1:]))

    # Take up to 5 most recent versions, including the current one
    previous_versions = previous_versions[-5:] if len(previous_versions) > 5 else previous_versions

    # Ensure the current version is included
    current_version_tuple = (f"v{step_context.model.version}", tst_acc)
    if current_version_tuple not in previous_versions:
        previous_versions.append(current_version_tuple)
        # Sort again to ensure the current version is in the correct position
        previous_versions.sort(key=lambda x: int(x[0][1:]))
        # Trim to 5 versions if necessary
        previous_versions = previous_versions[-5:]

    # Create a visually stunning image
    img_width, img_height = 1000, 800
    img = Image.new('RGB', (img_width, img_height))

    # Create gradient background
    for y in range(img_height):
        r = int(25 + (y / img_height) * 50)
        g = int(25 + (y / img_height) * 25)
        b = int(50 + (y / img_height) * 75)
        for x in range(img_width):
            img.putpixel((x, y), (r, g, b))

    draw = ImageDraw.Draw(img)

    # Add a stylized border
    border_width = 10
    draw.rectangle([0, 0, img_width, img_height], outline=(200, 200, 200), width=border_width)

    # Load fonts
    try:
        title_font = ImageFont.truetype("Arial.ttf", 50)
        main_font = ImageFont.truetype("Arial.ttf", 40)
        small_font = ImageFont.truetype("Arial.ttf", 30)
    except IOError:
        title_font = ImageFont.load_default()
        main_font = ImageFont.load_default()
        small_font = ImageFont.load_default()

    # Draw title with shadow effect
    title = "Model Accuracy Dashboard"
    draw.text((img_width//2+2, 32), title, fill=(50, 50, 50), font=title_font, anchor="mt")
    draw.text((img_width//2, 30), title, fill=(250, 250, 250), font=title_font, anchor="mt")

    # Draw radial accuracy meter
    center_x, center_y = 250, 300
    radius = 150
    draw.arc([center_x-radius, center_y-radius, center_x+radius, center_y+radius], 
             start=-210, end=30, fill=(200, 200, 200), width=5)
    angle = -210 + 240 * tst_acc
    end_x = center_x + radius * np.cos(np.radians(angle))
    end_y = center_y + radius * np.sin(np.radians(angle))
    draw.line([center_x, center_y, end_x, end_y], fill=(255, 100, 100), width=5)
    draw.text((center_x, center_y+radius+20), f"Test Accuracy: {tst_acc:.2%}", 
              fill=(250, 250, 250), font=main_font, anchor="mt")

    # Draw accuracy history graph
    graph_left, graph_top, graph_right, graph_bottom = 500, 200, 950, 600
    draw.rectangle([graph_left, graph_top, graph_right, graph_bottom], outline=(200, 200, 200), width=2)

    # Add subtle grid
    for i in range(1, 5):
        y = graph_top + (graph_bottom - graph_top) * i / 5
        draw.line([graph_left, y, graph_right, y], fill=(150, 150, 150), width=1)

    # Plot points with custom icons and lines
    # Adjust x_step based on the number of versions
    num_versions = len(previous_versions)
    x_step = (graph_right - graph_left) / (num_versions - 1) if num_versions > 1 else 0

    for i, (version, acc) in enumerate(previous_versions):
        x = graph_left + i * x_step
        y = graph_bottom - (acc * (graph_bottom - graph_top))
        
        # Custom icon (small hexagon)
        icon_size = 15
        draw.regular_polygon((x, y, icon_size), n_sides=6, rotation=30, 
                             fill=(255, 200, 100), outline=(250, 250, 250))
        
        if i > 0:
            prev_x = graph_left + (i-1) * x_step
            prev_y = graph_bottom - (previous_versions[i-1][1] * (graph_bottom - graph_top))
            draw.line([prev_x, prev_y, x, y], fill=(255, 200, 100), width=3)
        
        draw.text((x, y+25), version, fill=(250, 250, 250), font=small_font, anchor="mt")

    # Add graph labels with shadow effect
    draw.text((graph_left + (graph_right - graph_left)//2+2, graph_bottom + 52), 
              "Model Versions", fill=(50, 50, 50), font=main_font, anchor="mt")
    draw.text((graph_left + (graph_right - graph_left)//2, graph_bottom + 50), 
              "Model Versions", fill=(250, 250, 250), font=main_font, anchor="mt")

    draw.text((graph_left - 52, graph_top + (graph_bottom - graph_top)//2), 
              "Accuracy", fill=(50, 50, 50), font=main_font, anchor="mm", rotation=90)
    draw.text((graph_left - 50, graph_top + (graph_bottom - graph_top)//2), 
              "Accuracy", fill=(250, 250, 250), font=main_font, anchor="mm", rotation=90)

    # Apply a subtle blur for a glow effect
    img = img.filter(ImageFilter.GaussianBlur(radius=1))

    return img
