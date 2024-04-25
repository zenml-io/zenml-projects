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
"""Implementation of the PyTorch DataLoader materializer."""

import os
import tempfile
from typing import Any, ClassVar, List, Optional, Type

from zenml.io import fileio
from zenml.materializers.base_materializer import BaseMaterializer

DEFAULT_FILENAME = "data.zip"


class LabelStudioYOLODataset:
    dataset: Optional[Any]
    filepath: Optional[str]
    task_ids: Optional[List[int]]

    def download_dataset(self):
        tmpfile_ = tempfile.NamedTemporaryFile(dir="data", delete=False)
        tmpdirname = os.path.basename(tmpfile_.name)
        self.filepath = os.path.join(tmpdirname, DEFAULT_FILENAME)
        self.dataset.export_tasks(
            export_type="YOLO",
            export_location=self.filepath,
            download_resources=False,
            ids=self.task_ids,
        )


class LabelStudioYOLODatasetMaterializer(BaseMaterializer):
    """Base class for Label Studio YOLO dataset models."""

    FILENAME: ClassVar[str] = DEFAULT_FILENAME
    SKIP_REGISTRATION: ClassVar[bool] = True
    ASSOCIATED_TYPES = (LabelStudioYOLODataset,)

    def load(self, data_type: Type[Any]) -> LabelStudioYOLODataset:
        """Reads a ultralytics YOLO model from a serialized JSON file.

        Args:
            data_type: A ultralytics YOLO type.

        Returns:
            A ultralytics YOLO object.
        """
        filepath = os.path.join(self.uri, DEFAULT_FILENAME)

        # Create a temporary folder
        tmpfile_ = tempfile.NamedTemporaryFile(
            dir="data", delete=False, suffix=".zip"
        )

        # Copy from artifact store to temporary file
        fileio.copy(filepath, tmpfile_.name, overwrite=True)
        dataset = LabelStudioYOLODataset()
        dataset.filepath = tmpfile_.name

        return dataset

    def save(self, dataset: LabelStudioYOLODataset) -> None:
        """Creates a JSON serialization for a label studio YOLO dataset model.

        Args:
            dataset: A label studio YOLO dataset model.
        """
        dataset.download_dataset()

        filepath = os.path.join(self.uri, DEFAULT_FILENAME)

        fileio.copy(dataset.filepath, filepath)
