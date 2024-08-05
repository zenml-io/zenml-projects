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

from materializers import CSVDataset
from zenml.enums import ArtifactType
from zenml.io import fileio
from zenml.logger import get_logger
from zenml.materializers.base_materializer import BaseMaterializer

logger = get_logger(__name__)


class CSVDatasetMaterializer(BaseMaterializer):
    ASSOCIATED_TYPES = (CSVDataset,)
    ASSOCIATED_ARTIFACT_TYPE = ArtifactType.DATA

    def load(self, data_type: Type[CSVDataset]) -> CSVDataset:
        logger.info(f"Loading CSVDataset from {self.uri}")
        with fileio.open(os.path.join(self.uri, "metadata.json"), "r") as f:
            metadata = json.load(f)
        return CSVDataset(metadata["data_path"])

    def save(self, csv_dataset: CSVDataset) -> None:
        logger.info(f"Saving CSVDataset to {self.uri}")
        metadata = {"data_path": csv_dataset.data_path}
        with fileio.open(os.path.join(self.uri, "metadata.json"), "w") as f:
            json.dump(metadata, f)

        # Write the dataframe if it is not None
        if csv_dataset.df is not None:
            logger.info(f"Writing CSV file {csv_dataset.data_path}")
            with fileio.open(csv_dataset.data_path, "w") as f:
                csv_dataset.df.to_csv(f)
