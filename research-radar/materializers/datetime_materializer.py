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

# materializers/datetime_materializer.py
import os
from datetime import datetime
from typing import Type

from zenml.enums import ArtifactType
from zenml.materializers.base_materializer import BaseMaterializer


class DatetimeMaterializer(BaseMaterializer):
    ASSOCIATED_TYPES = (datetime,)
    ASSOCIATED_ARTIFACT_TYPE = ArtifactType.DATA

    def load(self, data_type: Type[datetime]) -> datetime:
        with self.artifact_store.open(
            os.path.join(self.uri, "data.txt"), "r"
        ) as f:
            iso_str = f.read()
        return datetime.fromisoformat(iso_str)

    def save(self, data: datetime) -> None:
        with self.artifact_store.open(
            os.path.join(self.uri, "data.txt"), "w"
        ) as f:
            f.write(data.isoformat())
