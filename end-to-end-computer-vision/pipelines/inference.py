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
from zenml import pipeline, step
from zenml.logger import get_logger

from zenml.client import Client
import os

os.environ["YOLO_VERBOSE"] = "False"

import fiftyone as fo

logger = get_logger(__name__)


import fiftyone as fo

DATASET_NAME = "ships"
DATASET_DIR = "data/ships/subset"


@step
def predict_over_images() -> fo.Dataset:

    # model = Model(
    #     name="Yolo_Object_Detection",
    #     version="staging",
    # )
    # model.get_model_artifact(name="staging").load()
    # model_artifact = model_version.get_model_artifact(name="Raw_YOLO")
    artifact = Client().get_artifact_version(
        "105c768c-8c86-465a-b018-b1a800ad4e19"
    )
    yolo_model = artifact.load()
    # results = yolo_model(DATASET_DIR, half=True, conf=0.6)

    dataset = fo.Dataset.from_dir(
        dataset_dir=DATASET_DIR,
        dataset_type=fo.types.ImageDirectory,
        name=DATASET_NAME,
    )
    dataset.persistent = True
    # # View summary info about the dataset
    # print(dataset)

    # # Print the first few samples in the dataset
    # print(dataset.head())

    dataset.apply_model(yolo_model, label_field="boxes")
    return dataset


@pipeline
def inference():
    dataset = predict_over_images()
