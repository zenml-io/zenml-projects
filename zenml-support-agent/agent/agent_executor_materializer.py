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
"""Implementation of ZenML's pickle materializer."""

import os
import pickle
from typing import Any, ClassVar, Tuple, Type

from zenml.enums import ArtifactType
from zenml.environment import Environment
from zenml.io import fileio
from zenml.logger import get_logger
from zenml.materializers.base_materializer import BaseMaterializer
from zenml.utils.io_utils import (
    read_file_contents_as_string,
    write_file_contents_as_string,
)

logger = get_logger(__name__)

DEFAULT_FILENAME = "agent.pkl"
DEFAULT_PYTHON_VERSION_FILENAME = "python_version.txt"


class AgentExecutorMaterializer(BaseMaterializer):
    """Materializer using pickle.

    This materializer can materialize (almost) any object, but does so in a
    non-reproducble way since artifacts cannot be loaded from other Python
    versions. It is recommended to use this materializer only as a last resort.

    That is also why it has `SKIP_REGISTRATION` set to True and is currently
    only used as a fallback materializer inside the materializer registry.
    """

    ASSOCIATED_TYPES: ClassVar[Tuple[Type[Any], ...]] = (object,)
    ASSOCIATED_ARTIFACT_TYPE: ClassVar[ArtifactType] = ArtifactType.DATA
    SKIP_REGISTRATION: ClassVar[bool] = True

    def load(self, data_type: Type[Any]) -> Any:
        """Reads an artifact from a pickle file.

        Args:
            data_type: The data type of the artifact.

        Returns:
            The loaded artifact data.
        """
        # validate python version
        source_python_version = self._load_python_version()
        current_python_version = Environment().python_version()
        if source_python_version != current_python_version:
            logger.warning(
                f"Your artifact was materialized under Python version "
                f"'{source_python_version}' but you are currently using "
                f"'{current_python_version}'. This might cause unexpected "
                "behavior since pickle is not reproducible across Python "
                "versions. Attempting to load anyway..."
            )

        # load data
        filepath = os.path.join(self.uri, DEFAULT_FILENAME)
        with fileio.open(filepath, "rb") as fid:
            data = pickle.load(fid)
        return data

    def _load_python_version(self) -> str:
        """Loads the Python version that was used to materialize the artifact.

        Returns:
            The Python version that was used to materialize the artifact.
        """
        filepath = os.path.join(self.uri, DEFAULT_PYTHON_VERSION_FILENAME)
        if os.path.exists(filepath):
            return read_file_contents_as_string(filepath)
        return "unknown"

    def save(self, data: Any) -> None:
        """Saves an artifact to a pickle file.

        Args:
            data: The data to save.
        """
        # save python version for validation on loading
        self._save_python_version()

        # save data
        filepath = os.path.join(self.uri, DEFAULT_FILENAME)
        with fileio.open(filepath, "wb") as fid:
            pickle.dump(data, fid)

    def _save_python_version(self) -> None:
        """Saves the Python version used to materialize the artifact."""
        filepath = os.path.join(self.uri, DEFAULT_PYTHON_VERSION_FILENAME)
        current_python_version = Environment().python_version()
        write_file_contents_as_string(filepath, current_python_version)
