#  Copyright (c) ZenML GmbH 2022. All Rights Reserved.
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
import os
from typing import Annotated, Dict, List

import cv2
from zenml.client import Client
from zenml.steps import step


@step
def inference_loader() -> Annotated[List, "images_path"]:
    """Loads the trained models from previous training pipeline runs."""
    training_pipeline = Client().get_pipeline(
        "sign_language_detection_train_pipeline"
    )
    last_run = training_pipeline.runs[0]

    try:
        inference_data: Dict = (
            last_run.get_step("data_loader").outputs["test_images"].read()
        )
    except KeyError:
        print(
            f"Skipping {last_run.name} as it does not contain the data_loader"
        )
    images = image_saver(inference_data)
    return images


def image_saver(image_set: Dict):
    # Create a temporary directory to store the model
    os.makedirs(os.path.join("inference", "images"), exist_ok=True)
    images = []
    for key, value in image_set.items():
        dim = (768, 1024)
        resized_image = cv2.resize(value[0], dim, interpolation=cv2.INTER_AREA)
        image_path = os.path.join("inference", "images", key.rsplit("/", 1)[1])
        if cv2.imwrite(image_path, resized_image):
            images.append(image_path)
    return images
