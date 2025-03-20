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

# materializers/url_materializer.py
import os
from typing import Type

from zenml.enums import ArtifactType
from zenml.materializers.base_materializer import BaseMaterializer

try:
    from pydantic_core import Url
except ImportError:
    from pydantic_core._pydantic_core import Url  # fallback if needed


class UrlMaterializer(BaseMaterializer):
    """Materializer for pydantic's URL type."""

    ASSOCIATED_TYPES = (Url,)
    # Use DATA instead of STRING
    ASSOCIATED_ARTIFACT_TYPE = ArtifactType.DATA

    def load(self, data_type: Type[Url]) -> Url:
        # Read the URL as a string from the file
        with self.artifact_store.open(
            os.path.join(self.uri, "data.txt"), "r"
        ) as f:
            url_str = f.read()
        # Return the URL string (if Url is a subclass of str, otherwise wrap it)
        return url_str  # or: return Url(url_str)

    def save(self, data: Url) -> None:
        # Write the URL (as a string) to disk
        with self.artifact_store.open(
            os.path.join(self.uri, "data.txt"), "w"
        ) as f:
            f.write(str(data))
