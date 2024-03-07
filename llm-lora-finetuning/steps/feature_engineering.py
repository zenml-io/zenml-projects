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
import importlib
import json
import os
from dataclasses import asdict
from pathlib import Path
from tempfile import mkdtemp
from typing import Annotated, Any, ClassVar, Dict, Tuple, Type

from lit_gpt import Config
from pydantic import BaseModel
from zenml import log_artifact_metadata, step
from zenml.enums import ArtifactType
from zenml.io import fileio
from zenml.materializers.base_materializer import BaseMaterializer

from scripts.download import download_from_hub
from steps.utils import get_huggingface_access_token


class DirectoryMaterializer(BaseMaterializer):
    ASSOCIATED_TYPES: ClassVar[Tuple[Type[Any], ...]] = (Path,)
    ASSOCIATED_ARTIFACT_TYPE: ClassVar[ArtifactType] = ArtifactType.DATA

    def load(self, data_type: Type[Any]) -> Any:
        """Write logic here to load the data of an artifact.

        Args:
            data_type: What type the artifact data should be loaded as.

        Returns:
        """
        directory = mkdtemp(prefix="zenml-artifact")
        self._copy_directory(src=self.uri, dst=directory)
        return Path(directory)

    def save(self, data: Any) -> None:
        """Write logic here to save the data of an artifact.

        Args:
            data: The data of the artifact to save.
        """
        assert isinstance(data, Path)
        self._copy_directory(src=str(data), dst=self.uri)

    @staticmethod
    def _copy_directory(src: str, dst: str) -> None:
        for src_dir, _, files in fileio.walk(src):
            dst_dir = os.path.join(dst, os.path.relpath(src_dir, src))
            fileio.makedirs(dst_dir)

            for file in files:
                src_file = os.path.join(src_dir, file)
                dst_file = os.path.join(dst_dir, file)
                fileio.copy(src_file, dst_file)


class FeatureEngineeringParameters(BaseModel):
    model_repo: str
    dataset_name: str

    prepare_kwargs: Dict[str, Any] = {}


@step(output_materializers=DirectoryMaterializer)
def feature_engineering(
    config: FeatureEngineeringParameters,
) -> Annotated[Path, "data"]:
    """Prepare the dataset.

    Args:
        config: Configuration for this step.
    """
    access_token = get_huggingface_access_token()

    checkpoint_root_dir = Path("checkpoints")
    download_from_hub(
        repo_id=config.model_repo,
        tokenizer_only=True,
        checkpoint_dir=checkpoint_root_dir,
        access_token=access_token,
    )

    checkpoint_dir = checkpoint_root_dir / config.model_repo

    model_name = checkpoint_dir.name
    config = Config.from_name(model_name)
    config_dict = asdict(config)
    with open(checkpoint_dir / "lit_config.json", "w") as json_config:
        json.dump(config_dict, json_config)

    log_artifact_metadata(
        metadata={
            "model_name": model_name,
            "model_config": config_dict,
            "dataset_name": config.dataset_name,
        }
    )
    destination_dir = Path("data") / config.dataset_name

    helper_module = importlib.import_module(
        f"scripts.prepare_{config.dataset_name}"
    )
    prepare_function = getattr(helper_module, "prepare")

    prepare_function(
        checkpoint_dir=checkpoint_dir,
        destination_path=destination_dir,
        **config.prepare_kwargs,
    )
    return destination_dir
