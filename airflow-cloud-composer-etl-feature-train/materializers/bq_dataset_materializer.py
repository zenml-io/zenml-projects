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

from materializers import BigQueryDataset
from zenml.enums import ArtifactType
from zenml.io import fileio
from zenml.logger import get_logger
from zenml.materializers.base_materializer import BaseMaterializer

logger = get_logger(__name__)


class BigQueryDatasetMaterializer(BaseMaterializer):
    ASSOCIATED_TYPES = (BigQueryDataset,)
    ASSOCIATED_ARTIFACT_TYPE = ArtifactType.DATA

    def load(self, data_type: Type[BigQueryDataset]) -> BigQueryDataset:
        logger.info(f"Loading BigQueryDataset from {self.uri}")
        with fileio.open(os.path.join(self.uri, "metadata.json"), "r") as f:
            metadata = json.load(f)
        dataset = BigQueryDataset(
            table_id=metadata["table_id"],
            write_disposition=metadata["write_disposition"],
            project=metadata.get("project"),
            dataset=metadata.get("dataset"),
        )
        dataset.read_data()
        return dataset

    def save(self, bq_dataset: BigQueryDataset) -> None:
        logger.info(f"Saving BigQueryDataset to {self.uri}")
        metadata = {
            "table_id": bq_dataset.table_id,
            "write_disposition": bq_dataset.write_disposition,
            "project": bq_dataset.project,
            "dataset": bq_dataset.dataset,
        }
        with fileio.open(os.path.join(self.uri, "metadata.json"), "w") as f:
            json.dump(metadata, f)

        # Write the table if the dataframe is not None
        if bq_dataset.df is not None:
            logger.info(f"Writing BigQuery table {bq_dataset.table_id}")
            bq_dataset.write_data()
