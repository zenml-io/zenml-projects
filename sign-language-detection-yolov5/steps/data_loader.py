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
from typing import Any, Dict

import cv2
import numpy as np
from genericpath import isdir
from materializer.dataset_materializer import DatasetMaterializer
from roboflow import Roboflow
from zenml.steps import BaseParameters, Output, step


class TrainerParameters(BaseParameters):
    """Trainer params"""

    api_key: str = "YOUR_API_KEY"
    workspace: str = "WORKSPACE"
    project: str = "american-sign-language-letters"
    annotation_type: str = "yolov5"


def roboflow_download(
    api_key: str, workspace: str, project: str, annotation_type: str
) -> Any:
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace).project(project)
    dataset = project.version(6).download(annotation_type)
    return dataset.location


@step(
    output_materializers={
        "train_images": DatasetMaterializer,
        "val_images": DatasetMaterializer,
        "test_images": DatasetMaterializer,
    }
)
def data_loader(
    params: TrainerParameters,
) -> Output(train_images=Dict, val_images=Dict, test_images=Dict):
    """Loads data from Roboflow"""
    images: dict(str, list(np.ndarray, list)) = {}
    train_images: dict(str, list(np.ndarray, list)) = {}
    valid_images: dict(str, list(np.ndarray, list)) = {}
    test_images: dict(str, list(np.ndarray, list)) = {}
    dataset_path = roboflow_download(
        params.api_key,
        params.workspace,
        params.project,
        params.annotation_type,
    )
    for folder in os.listdir(dataset_path):
        if isdir(os.path.join(dataset_path, folder)):
            folder_path = os.path.join(dataset_path, folder)
            for filename in os.listdir(os.path.join(folder_path, "images")):
                img_array = cv2.imread(os.path.join(folder_path, "images", filename))
                load_bboxes = np.genfromtxt(
                    os.path.join(folder_path, "labels", f"{filename[:-4]}.txt")
                )
                load_bboxes = list(load_bboxes)
                images[os.path.join(folder, filename)] = [
                    img_array,
                    load_bboxes,
                ]

            # save each of the sets into different dictionary
            if folder == "train":
                train_images = images
            elif folder == "valid":
                valid_images = images
            else:
                test_images = images
            images = {}
    return train_images, valid_images, test_images
