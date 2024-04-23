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
from typing import Annotated

import fiftyone as fo
from zenml import log_artifact_metadata, step
from zenml.client import Client
from zenml.logger import get_logger

from utils.constants import (
    DATASET_DIR,
    DATASET_NAME,
    PREDICTIONS_DATASET_ARTIFACT_NAME,
    TRAINED_MODEL_NAME,
)

logger = get_logger(__name__)

os.environ["YOLO_VERBOSE"] = "False"


@step
def create_fiftyone_dataset() -> (
    Annotated[str, PREDICTIONS_DATASET_ARTIFACT_NAME]
):
    c = Client()
    model_artifact = c.get_artifact_version(
        name_id_or_prefix=TRAINED_MODEL_NAME
    )
    yolo_model = model_artifact.load()

    dataset = fo.Dataset.from_dir(
        dataset_dir=DATASET_DIR,
        dataset_type=fo.types.ImageDirectory,
        name=DATASET_NAME,
        overwrite=True,
        persistent=True,
    )

    dataset.apply_model(yolo_model, label_field="boxes")

    log_artifact_metadata(
        artifact_name="predictions_dataset_json",
        metadata={
            "summary_info": dataset.summary(),
            "persistence": dataset.persistent,
        },
    )

    return dataset.to_json()
