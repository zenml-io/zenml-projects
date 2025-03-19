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

# materializers/register_materializers.py
from datetime import datetime
from datasets import Dataset
from pydantic_core._pydantic_core import Url
import polars as pl
from zenml.materializers.materializer_registry import materializer_registry
from materializers import (
    DatetimeMaterializer,
    DatasetMaterializer,
    UrlMaterializer,
    LazyFrameMaterializer,
)


def register_materializers() -> None:
    """
    Registers custom materializers with ZenML's MaterializerRegistry.
    """
    materializer_registry.register_and_overwrite_type(key=datetime, type_=DatetimeMaterializer)
    materializer_registry.register_and_overwrite_type(key=Dataset, type_=DatasetMaterializer)
    materializer_registry.register_and_overwrite_type(key=Url, type_=UrlMaterializer)
    materializer_registry.register_and_overwrite_type(key=pl.LazyFrame, type_=LazyFrameMaterializer)
