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
from typing import TYPE_CHECKING, Any, ClassVar, Type

from zenml.io import fileio
from zenml.materializers.base_materializer import BaseMaterializer

DEFAULT_FILENAME = "annotations.zip"

if TYPE_CHECKING:
    from label_studio_sdk import Project


class LabelStudioAnnotationExport:
    def __init__(self, dataset: "Project" = None, filepath: str = None):
        """
        Initialize LabelStudioAnnotationExport object with optional parameters.

        Parameters:
        dataset: Label-studio dataset object.
        filepath: A string representing the file path. Defaults to None.
        """
        self.dataset = dataset
        self.filepath = filepath

    def download_annotations(self):
        """Downloads the annotations from label-studio to the local fs."""
        tmpfile_ = tempfile.NamedTemporaryFile(dir="data", delete=False)
        tmpdirname = os.path.basename(tmpfile_.name)
        self.filepath = os.path.join(tmpdirname, DEFAULT_FILENAME)
        self.dataset.export_tasks(
            export_type="YOLO",
            export_location=self.filepath,
            download_resources=False,
        )


class LabelStudioAnnotationMaterializer(BaseMaterializer):
    """Base class for Label Studio annotation models."""

    FILENAME: ClassVar[str] = DEFAULT_FILENAME
    SKIP_REGISTRATION: ClassVar[bool] = True
    ASSOCIATED_TYPES = (LabelStudioAnnotationExport,)

    def load(self, data_type: Type[Any]) -> LabelStudioAnnotationExport:
        """Loads the serialized JSON file containing the annotations.

        Args:
            data_type: A ultralytics YOLO type.

        Returns:
            A ultralytics YOLO object.
        """
        # Recreate the filepath of the file
        filepath = os.path.join(self.uri, DEFAULT_FILENAME)

        # Create a temporary file
        tmpfile_ = tempfile.NamedTemporaryFile(
            dir="data", delete=False, suffix=".zip"
        )

        # Copy from artifact store to temporary file
        fileio.copy(filepath, tmpfile_.name, overwrite=True)

        # Re-instantiate the LabelStudioAnnotationExport model
        dataset = LabelStudioAnnotationExport(filepath=tmpfile_.name)

        return dataset

    def save(self, dataset: LabelStudioAnnotationExport) -> None:
        """Creates a JSON serialization for a label studio YOLO dataset model.

        Args:
            dataset: A label studio YOLO dataset model.
        """
        # Downloads the annotations into the local fs
        dataset.download_annotations()

        # create the destination path for the exported annotations
        filepath = os.path.join(self.uri, DEFAULT_FILENAME)

        # copies the files from local fs into the artifact store
        fileio.copy(dataset.filepath, filepath)
