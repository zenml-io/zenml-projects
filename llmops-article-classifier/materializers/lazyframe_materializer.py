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

# materializers/lazyframe_materializer.py
import os
from typing import Type
import polars as pl
from zenml.materializers.base_materializer import BaseMaterializer
from zenml.enums import ArtifactType


class LazyFrameMaterializer(BaseMaterializer):
    """Materializer for Polars LazyFrame."""

    ASSOCIATED_TYPES = (pl.LazyFrame,)
    ASSOCIATED_ARTIFACT_TYPE = ArtifactType.DATA

    def load(self, data_type: Type[pl.LazyFrame]) -> pl.LazyFrame:
        """Load the LazyFrame from a parquet file."""
        return pl.scan_parquet(os.path.join(self.uri, "data.parquet"))

    def save(self, data: pl.LazyFrame) -> None:
        """Save the LazyFrame to a parquet file."""
        # Collect and write to parquet
        data.collect().write_parquet(os.path.join(self.uri, "data.parquet"))
