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
import os
import tempfile
from zenml import step, pipeline, Model, ArtifactConfig, log_artifact_metadata
from zenml.client import Client
from ultralytics import YOLO
from typing_extensions import Annotated
from zenml.logger import get_logger
from typing import Tuple
from utils.split_data import unzip_dataset, split_dataset, generate_yaml
from PIL import Image
import os

logger = get_logger(__name__)


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        if (
            filename.endswith(".png")
            or filename.endswith(".jpg")
            or filename.endswith(".jpeg")
        ):
            with open(os.path.join(folder, filename), "rb") as f:
                img = Image.open(f)
                images.append(img)
    return images


def load_and_split_data(dataset_name: str) -> str:

    annotator = Client().active_stack.annotator
    from zenml.integrations.label_studio.annotators.label_studio_annotator import (
        LabelStudioAnnotator,
    )

    if not isinstance(annotator, LabelStudioAnnotator):
        raise TypeError(
            "This step can only be used with the Label Studio annotator."
        )

    if annotator and annotator._connection_available():
        for dataset in annotator.get_datasets():
            if dataset.get_params()["title"] == dataset_name:
                tmpfile_ = tempfile.NamedTemporaryFile(
                    dir="data", delete=False
                )
                tmpdirname = os.path.basename(tmpfile_.name)
                export_location = os.path.join(tmpdirname, "data.zip")
                extract_location = os.path.join(tmpdirname, "data")
                dataset.export_tasks(
                    export_type="YOLO",
                    export_location=export_location,
                    download_resources=True,
                )
                unzip_dataset(export_location, extract_location)
                split_dataset(
                    extract_location, ratio=(0.7, 0.15, 0.15), seed=42
                )
                return generate_yaml(extract_location)

    raise ValueError(f"Dataset {dataset_name} not found in Label Studio.")


@step
def load_model(
    model_checkpoint: str = "yolov8l.pt",
) -> Annotated[YOLO, ArtifactConfig(name="Raw_YOLO", is_model_artifact=True)]:
    logger.info(f"Loading YOLO checkpoint {model_checkpoint}")
    # return YOLO(model_checkpoint)
    return YOLO()


@step
def train_model(
    model: YOLO,
    dataset_name: str = "zenml_test_project",
) -> Tuple[
    Annotated[
        YOLO, ArtifactConfig(name="Trained_YOLO", is_model_artifact=True)
    ],
    Annotated[Image.Image, "confusion_matrix"],
]:

    data_path = load_and_split_data(dataset_name)
    model.train(data=data_path, epochs=1)

    # model.train(data="coco8.yaml", epochs=3)  # train the model
    metrics = model.val()  # evaluate model performance on the validation set

    log_artifact_metadata(
        artifact_name="Trained_YOLO",
        metadata={"metrics": metrics.results_dict},
    )

    # Read images as PIL images from directory metrics.save_dir for all png and jpg files
    images = load_images_from_folder(metrics.save_dir)

    return model, images[0]


@pipeline(enable_cache=True, model=Model(name="Yolo_Object_Detection"))
def my_pipeline():
    model = load_model()
    train_model(model=model)


if __name__ == "__main__":
    my_pipeline()
