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
from typing import Any, ClassVar, Type, Optional
import tempfile

from pydantic import BaseModel
from ultralytics import YOLO

from zenml.io import fileio
from zenml.integrations.pytorch.materializers.pytorch_module_materializer import (
    PyTorchModuleMaterializer,
)
from zenml.materializers.base_materializer import BaseMaterializer

DEFAULT_FILENAME = "data.zip"


class LabelStudioYOLODataset:
    dataset: Optional[Any]
    filepath: Optional[str]

    def download_dataset(self):
        tmpfile_ = tempfile.NamedTemporaryFile(
            dir="data", delete=False
        )
        tmpdirname = os.path.basename(tmpfile_.name)
        self.filepath = os.path.join(tmpdirname, DEFAULT_FILENAME)
        self.dataset.export_tasks(
            export_type="YOLO",
            export_location=self.filepath,
            download_resources=True,
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
        with tempfile.TemporaryDirectory(prefix="zenml-temp-") as temp_dir:
            temp_file = os.path.join(str(temp_dir), DEFAULT_FILENAME)

            # Copy from artifact store to temporary file
            fileio.copy(filepath, temp_file)
            dataset = LabelStudioYOLODataset(filepath=temp_file)

            return dataset

    def save(self, dataset: LabelStudioYOLODataset) -> None:
        """Creates a JSON serialization for a label studio YOLO dataset model.

        Args:
            dataset: A label studio YOLO dataset model.
        """
        dataset.download_dataset()

        filepath = os.path.join(self.uri, DEFAULT_FILENAME)

        # Make a temporary phantom artifact
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as f:
            # Copy it into artifact store
            fileio.copy(dataset.filepath, filepath)
