# Apache Software License 2.0
#
# Copyright (c) ZenML GmbH 2025. All rights reserved.
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

from pathlib import Path
from typing import Optional

from datasets import Dataset, load_from_disk
from typing_extensions import Annotated

from schemas import zenml_project
from utils import logger
from zenml import ArtifactConfig, step
from zenml.artifacts.utils import load_artifact
from zenml.client import Client


@step(model=zenml_project)
def load_test_set(
    source_type: str = "artifact",
    path: Optional[str] = None,
    artifact_name: Optional[str] = None,
    version: Optional[int] = None,
) -> Annotated[
    Dataset,
    ArtifactConfig(name="test_set_artifact", run_metadata={"source": "training_pipeline"}),
]:
    """Load a test dataset from either disk or ZenML artifact.

    Args:
        source_type: Type of data source ('disk' or 'artifact')
        path: Path to dataset on disk (required when source_type is 'disk')
        artifact_name: Name of the ZenML artifact (required when source_type is 'artifact')

    Returns:
        Dataset: The loaded dataset

    Raises:
        ValueError: If the parameters are invalid or the dataset cannot be loaded
    """
    source_type = source_type.lower()
    if source_type not in ["disk", "artifact"]:
        raise ValueError(f"Invalid source_type: {source_type}. Must be 'disk' or 'artifact'")

    try:
        if source_type == "disk":
            if not path:
                raise ValueError("Path must be provided when loading from disk")
            logger.info(f"Loading dataset from disk: {path}")
            if not Path(path).exists():
                raise FileNotFoundError(f"Dataset path does not exist: {path}")
            dataset = load_from_disk(path)

        else:  # artifact
            if not artifact_name:
                raise ValueError("Artifact name must be provided when loading from artifact")
            client = Client()
            artifact = client.get_artifact_version(name_id_or_prefix=artifact_name, version=version)
            dataset = load_artifact(artifact.id)

        logger.info(f"Successfully loaded dataset with {len(dataset)} samples")
        return dataset

    except Exception as e:
        logger.error(f"Failed to load dataset: {str(e)}")
        raise
