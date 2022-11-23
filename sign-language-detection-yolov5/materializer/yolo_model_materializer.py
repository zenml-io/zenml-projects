#  Copyright (c) ZenML GmbH 2021. All Rights Reserved.
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
"""Materializer for Yolov5 Trained Model."""

import os
import sys
import tempfile
from typing import Type

import torch
from zenml.artifacts import ModelArtifact
from zenml.io import fileio
from zenml.logger import get_logger
from zenml.materializers.base_materializer import BaseMaterializer
from zenml.utils import io_utils, source_utils

logger = get_logger(__name__)

DEFAULT_YOLOV5_MODEL_FILENAME = "model.pt"


class Yolov5ModelMaterializer(BaseMaterializer):
    """Materializer for Yolo Trained Model."""

    ASSOCIATED_TYPES = (dict,)
    ASSOCIATED_ARTIFACT_TYPES = (ModelArtifact,)

    def handle_input(self, data_type: Type[dict]) -> dict:
        """Read from artifact store and return a Dict object.

        Args:
            data_type: An Dict type.

        Returns:
            An Dict object.
        """
        super().handle_input(data_type)

        # Create a temporary directory to store the model
        temp_dir = tempfile.TemporaryDirectory()

        # Copy from artifact store to temporary directory
        io_utils.copy_dir(self.artifact.uri, temp_dir.name)

        # Load YoloV5 modules from the local repository
        yolov5_module_path = os.path.join(
            source_utils.get_source_root_path(), "yolov5"
        )
        if os.path.isdir(yolov5_module_path):
            sys.path.insert(0, yolov5_module_path)
        else:
            raise FileNotFoundError(
                f"Could not find yolov5 module at {yolov5_module_path}"
            )

        # Load the model from the temporary directory
        yolov5_model = torch.load(
            os.path.join(temp_dir.name, DEFAULT_YOLOV5_MODEL_FILENAME)
        )
        return yolov5_model

    def handle_return(self, ckpt: dict) -> None:
        """Write to artifact store.

        Args:
            ckpt: A Dict contains information regarding yolov5 model.
        """
        super().handle_return(ckpt)

        # Create a temporary directory to store the model
        temp_dir = tempfile.TemporaryDirectory(prefix="zenml-temp-")
        temp_ckpt_path = os.path.join(
            temp_dir.name, DEFAULT_YOLOV5_MODEL_FILENAME
        )

        # save the image in a temporary directory
        torch.save(ckpt, temp_ckpt_path)

        # copy the saved image to the artifact store
        io_utils.copy_dir(temp_dir.name, self.artifact.uri)

        # Remove the temporary directory
        fileio.rmtree(temp_dir.name)
