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
from genericpath import isdir
from typing import Annotated, Any, Dict, List, Tuple

import cv2
import numpy as np
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
) -> Tuple[
    Annotated[Dict, "train_images"],
    Annotated[Dict, "val_images"],
    Annotated[Dict, "test_images"],
]:
    """Loads data from Roboflow"""
    images: Dict[str, List[np.ndarray, List]] = {}
    train_images: Dict[str, List[np.ndarray, List]] = {}
    valid_images: Dict[str, List[np.ndarray, List]] = {}
    test_images: Dict[str, List[np.ndarray, List]] = {}
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
                img_array = cv2.imread(
                    os.path.join(folder_path, "images", filename)
                )
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
