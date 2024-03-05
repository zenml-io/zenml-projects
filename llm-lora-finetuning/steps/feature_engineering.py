# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
import importlib
from pathlib import Path
from tempfile import mkdtemp
from typing import Any, ClassVar, Tuple, Type

from zenml import step
from zenml.enums import ArtifactType
from zenml.io import fileio
from zenml.materializers.base_materializer import BaseMaterializer

from scripts.download import download_from_hub


class LocalDirectoryMaterializer(BaseMaterializer):
    ASSOCIATED_TYPES: ClassVar[Tuple[Type[Any], ...]] = (Path,)
    ASSOCIATED_ARTIFACT_TYPE: ClassVar[ArtifactType] = ArtifactType.DATA

    def load(self, data_type: Type[Any]) -> Any:
        """Write logic here to load the data of an artifact.

        Args:
            data_type: What type the artifact data should be loaded as.

        Returns:
        """
        directory = mkdtemp(prefix="zenml-artifact")
        fileio.copy(self.uri, directory)
        return Path(directory)

    def save(self, data: Any) -> None:
        """Write logic here to save the data of an artifact.

        Args:
            data: The data of the artifact to save.
        """
        assert isinstance(data, Path)
        fileio.copy(str(data), self.uri)


@step(output_materializers=LocalDirectoryMaterializer)
def feature_engineering(model_repo: str, dataset_name: str) -> Path:
    checkpoint_dir = Path("checkpoints")
    download_from_hub(
        repo_id=model_repo, tokenizer_only=True, checkpoint_dir=checkpoint_dir
    )

    destination_dir = Path("data") / dataset_name

    helper_module = importlib.import_module(f"scripts/prepare_{dataset_name}")
    prepare_function = getattr(helper_module, "prepare")

    prepare_function(
        checkpoint_dir=checkpoint_dir, destination_path=destination_dir
    )
    return destination_dir
