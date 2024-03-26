#  Copyright (c) ZenML GmbH 2024. All Rights Reserved.
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
"""Implementation of the Huggingface Trainer materializer."""

import importlib
import os
from tempfile import TemporaryDirectory
from typing import Any, ClassVar, Dict, Tuple, Type

from transformers import (  # type: ignore [import-untyped]
    AutoConfig,
    Trainer,
    TrainingArguments,
)
from zenml.enums import ArtifactType
from zenml.materializers.base_materializer import BaseMaterializer
from zenml.metadata.metadata_types import DType, MetadataType
from zenml.utils import io_utils, yaml_utils

DEFAULT_TRAINER_MODEL_DIR = "hf_model"
DEFAULT_ARGS_JSON = "args.json"


class HFTrainerMaterializer(BaseMaterializer):
    """Materializer to read torch model to and from huggingface pretrained model."""

    ASSOCIATED_TYPES: ClassVar[Tuple[Type[Any], ...]] = (Trainer,)
    ASSOCIATED_ARTIFACT_TYPE: ClassVar[ArtifactType] = ArtifactType.MODEL

    def load(self, data_type: Type[Trainer]) -> Trainer:
        """Reads HFModel.

        Args:
            data_type: The type of the model to read.

        Returns:
            The model read from the specified dir.
        """
        temp_dir = TemporaryDirectory()
        io_utils.copy_dir(
            os.path.join(self.uri, DEFAULT_TRAINER_MODEL_DIR), temp_dir.name
        )
        args = yaml_utils.read_json(self.uri, DEFAULT_ARGS_JSON)
        config = AutoConfig.from_pretrained(temp_dir.name)
        architecture = config.architectures[0]
        model_cls = getattr(importlib.import_module("transformers"), architecture)
        model = model_cls.from_pretrained(temp_dir.name)
        return Trainer(
            model=model,
            args=TrainingArguments(**args),
        )

    def save(self, trainer: Trainer) -> None:
        """Writes a Model to the specified dir.

        Args:
            model: The Torch Model to write.
        """
        temp_dir = TemporaryDirectory()
        trainer.save_model(temp_dir.name)
        io_utils.copy_dir(
            temp_dir.name,
            os.path.join(self.uri, DEFAULT_TRAINER_MODEL_DIR),
        )
        yaml_utils.write_json(
            os.path.join(self.uri, DEFAULT_ARGS_JSON), trainer.args.to_dict()
        )

    def extract_metadata(self, trainer: Trainer) -> Dict[str, "MetadataType"]:
        """Extract metadata from the given `PreTrainedModel` object.

        Args:
            model: The `PreTrainedModel` object to extract metadata from.

        Returns:
            The extracted metadata as a dictionary.
        """

        return {
            **trainer.args.to_dict(),
            "dtype": DType(str(trainer.model.dtype)),
            "device": str(trainer.model.device),
        }
