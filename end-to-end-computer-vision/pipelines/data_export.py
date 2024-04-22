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
from typing import Any, Annotated

from zenml import pipeline, step
from zenml.client import Client
from zenml.logger import get_logger

from materializers.label_studio_yolo_dataset_materializer import \
    LabelStudioYOLODataset

logger = get_logger(__name__)



@step
def load_data_from_label_studio(dataset_name: str) -> LabelStudioYOLODataset:
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
                ls_dataset = LabelStudioYOLODataset()
                ls_dataset.dataset = dataset
                return ls_dataset
    else:
        raise TypeError(
            "This step can only be used with an active Label Studio annotator."
        )

@pipeline
def data_export(dataset_name: str = 'polution'):
    load_data_from_label_studio(dataset_name=dataset_name)
