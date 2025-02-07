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
from typing import Annotated, List, Tuple

from zenml import log_metadata, step
from zenml.client import Client
from zenml.logger import get_logger

from materializers.label_studio_export_materializer import (
    LabelStudioAnnotationExport,
    LabelStudioAnnotationMaterializer,
)
from utils.constants import LABELED_DATASET_NAME
from zenml import log_artifact_metadata, step
from zenml.client import Client
from zenml.logger import get_logger

logger = get_logger(__name__)



@step(
    output_materializers={
        LABELED_DATASET_NAME: LabelStudioAnnotationMaterializer
    }
)
def load_data_from_label_studio(
    dataset_name: str,
) -> Tuple[
    Annotated[LabelStudioAnnotationExport, LABELED_DATASET_NAME],
    Annotated[List[int], "new_ids"],
]:
    """Loads data from Label Studio.

    Args:
        dataset_name: Name of the dataset to load.

    Returns:
        Tuple of the loaded dataset and the Label Studio task IDs.
    """
    annotator = Client().active_stack.annotator
    from zenml.integrations.label_studio.annotators.label_studio_annotator import (
        LabelStudioAnnotator,
    )

    if not isinstance(annotator, LabelStudioAnnotator):
        raise TypeError(
            "This step can only be used with the Label Studio annotator."
        )

    if annotator:
        try:
            dataset = annotator.get_dataset(dataset_name=dataset_name)
            ls_dataset = LabelStudioAnnotationExport()
            ls_dataset.dataset = dataset

            current_labeled_task_ids = dataset.get_labeled_tasks_ids()

            ls_dataset.task_ids = current_labeled_task_ids
            log_metadata(
                metadata={
                    "num_images": len(current_labeled_task_ids),
                },
                artifact_name=LABELED_DATASET_NAME,
                infer_artifact=True,
            )
            return ls_dataset, current_labeled_task_ids
        except:
            raise ValueError(
                f"Dataset {dataset_name} not found in Label Studio."
            )

    else:
        raise TypeError(
            "This step can only be used with an active Label Studio annotator."
        )
