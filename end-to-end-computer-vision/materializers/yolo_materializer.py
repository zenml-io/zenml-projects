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
import uuid
from typing import Any, ClassVar, Type

from ultralytics import YOLO
from zenml.enums import ArtifactType
from zenml.integrations.pytorch.materializers.pytorch_module_materializer import (
    PyTorchModuleMaterializer,
)
from zenml.io import fileio

DEFAULT_FILENAME = "obj.onnx"


# class UltralyticsMaterializer(BaseMaterializer):
class UltralyticsMaterializer(PyTorchModuleMaterializer):
    """Base class for ultralytics YOLO models."""

    ASSOCIATED_ARTIFACT_TYPE: ClassVar[ArtifactType] = ArtifactType.MODEL
    FILENAME: ClassVar[str] = DEFAULT_FILENAME
    SKIP_REGISTRATION: ClassVar[bool] = True
    ASSOCIATED_TYPES = (YOLO,)

    def load(self, data_type: Type[Any]) -> YOLO:
        """Reads a ultralytics YOLO model from a serialized JSON file.

        Args:
            data_type: A ultralytics YOLO type.

        Returns:
            A ultralytics YOLO object.
        """
        filepath = os.path.join(self.uri, DEFAULT_FILENAME)

        # Create a temporary folder
        # make a random string I can use as a prefix for the folder name
        random_string = str(uuid.uuid4().hex)
        temp_dir = os.path.join("/tmp", random_string)
        # make the directory if it doesn't already exist
        os.makedirs(temp_dir, exist_ok=True)
        temp_file = os.path.join(temp_dir, DEFAULT_FILENAME)

        # Copy from artifact store to temporary file
        fileio.copy(filepath, temp_file)
        model = YOLO(temp_file, task="detect")

        return model

    def save(self, model: YOLO) -> None:
        """Creates a JSON serialization for a ultralytics YOLO model.

        Args:
            model: A ultralytics YOLO model.
        """
        filepath = os.path.join(self.uri, DEFAULT_FILENAME)
        filename = model.export(format="onnx")

        # Copy it into artifact store
        fileio.copy(filename, filepath)
