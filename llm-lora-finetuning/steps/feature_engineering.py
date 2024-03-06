# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
import importlib
import json
import os
from dataclasses import asdict
from pathlib import Path
from tempfile import mkdtemp
from typing import Any, ClassVar, Tuple, Type

from lit_gpt import Config
from zenml import step
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


@step(output_materializers=DirectoryMaterializer)
def feature_engineering(model_repo: str, dataset_name: str) -> Path:
    access_token = get_huggingface_access_token()

    checkpoint_root_dir = Path("checkpoints")
    download_from_hub(
        repo_id=model_repo,
        tokenizer_only=True,
        checkpoint_dir=checkpoint_root_dir,
        access_token=access_token,
    )

    checkpoint_dir = checkpoint_root_dir / model_repo

    model_name = checkpoint_dir.name
    config = Config.from_name(model_name)
    config_dict = asdict(config)
    with open(checkpoint_dir / "lit_config.json", "w") as json_config:
        json.dump(config_dict, json_config)

    destination_dir = Path("data") / dataset_name

    helper_module = importlib.import_module(f"scripts.prepare_{dataset_name}")
    prepare_function = getattr(helper_module, "prepare")

    prepare_function(
        checkpoint_dir=checkpoint_dir, destination_path=destination_dir
    )
    return destination_dir
