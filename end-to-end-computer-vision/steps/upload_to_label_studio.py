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
from typing import Any, Dict

from zenml import step
from zenml.client import Client


@step
def upload_labels_to_label_studio(
    labels_dict: Dict[str, Any], ls_project_id: int = 7
):
    """Uploads ground truth labels for images to label studio.

    Args:
        labels_dict: Dictionary mapping image filepath to list of bbox labels
        ls_project_id: Project id of teh Label Studio project
    """
    annotator = Client().active_stack.annotator
    from zenml.integrations.label_studio.annotators.label_studio_annotator import (
        LabelStudioAnnotator,
    )

    if not isinstance(annotator, LabelStudioAnnotator):
        raise TypeError(
            "This step can only be used with the Label Studio annotator."
        )

    lsc = annotator._get_client()

    project = lsc.get_project(ls_project_id)

    tasks = project.get_tasks()

    for task in tasks:
        filename = task["storage_filename"]
        print(filename)
        project.create_annotation(
            task["id"], result=labels_dict[filename], ground_truth=True
        )
