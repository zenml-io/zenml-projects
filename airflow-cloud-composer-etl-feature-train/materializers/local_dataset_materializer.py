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

import json
import os
from typing import Type

from local_dataset import LocalDataset
from zenml.enums import ArtifactType
from zenml.io import fileio
from zenml.materializers.base_materializer import BaseMaterializer


class LocalDatasetMaterializer(BaseMaterializer):
    ASSOCIATED_TYPES = (LocalDataset,)
    ASSOCIATED_ARTIFACT_TYPE = ArtifactType.DATA

    def load(self, data_type: Type[LocalDataset]) -> LocalDataset:
        with fileio.open(os.path.join(self.uri, "metadata.json"), "r") as f:
            metadata = json.load(f)
        return LocalDataset(metadata["data_path"])

    def save(self, local_dataset: LocalDataset) -> None:
        metadata = {"data_path": local_dataset.data_path}
        with fileio.open(os.path.join(self.uri, "metadata.json"), "w") as f:
            json.dump(metadata, f)
