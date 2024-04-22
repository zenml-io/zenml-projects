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
from typing import Any, Annotated, List, Tuple
from zenml import pipeline, step
from zenml.client import Client
from zenml.logger import get_logger
from zenml import log_artifact_metadata
from zenml import get_step_context
from materializers.label_studio_yolo_dataset_materializer import  LabelStudioYOLODataset, LabelStudioYOLODatasetMaterializer
logger = get_logger(__name__)



@step(output_materializers=LabelStudioYOLODatasetMaterializer)
def load_data_from_label_studio(dataset_name: str) -> Tuple[Annotated[LabelStudioYOLODataset, "yolo_dataset"], Annotated[List[int], "task_ids"]]:
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
                
                c = Client()
                step_context = get_step_context()

                last_run = c.get_pipeline().last_successful_run
                last_task_ids = last_run.steps[-1].outputs['task_ids'].read()
                latest_task_ids = dataset.get_task_ids()
                new_task_ids = list(set(latest_task_ids) - set(last_task_ids))

                return ls_dataset, new_task_ids
    else:
        raise TypeError(
            "This step can only be used with an active Label Studio annotator."
        )

@pipeline
def data_export(dataset_name: str = 'polution'):
    load_data_from_label_studio(dataset_name=dataset_name)
