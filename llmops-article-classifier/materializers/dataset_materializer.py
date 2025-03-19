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

# materializers/dataset_materializer.py
from typing import Type
from datasets import Dataset, load_from_disk
from zenml.materializers.base_materializer import BaseMaterializer
from zenml.enums import ArtifactType


class DatasetMaterializer(BaseMaterializer):
    ASSOCIATED_TYPES = (Dataset,)
    ASSOCIATED_ARTIFACT_TYPE = ArtifactType.DATA

    def load(self, data_type: Type[Dataset]) -> Dataset:
        return load_from_disk(self.uri)

    def save(self, data: Dataset) -> None:
        data.save_to_disk(self.uri)
