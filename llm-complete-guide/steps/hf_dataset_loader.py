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

from typing import Annotated, Tuple

from constants import DATASET_NAME_DEFAULT
from datasets import Dataset, load_dataset
from zenml import step
from zenml.integrations.huggingface.materializers.huggingface_datasets_materializer import (
    HFDatasetMaterializer,
)


@step(output_materializers=HFDatasetMaterializer)
def load_hf_dataset() -> (
    Tuple[Annotated[Dataset, "train"], Annotated[Dataset, "test"]]
):
    train_dataset = load_dataset(DATASET_NAME_DEFAULT, split="train")
    test_dataset = load_dataset(DATASET_NAME_DEFAULT, split="test")
    return train_dataset, test_dataset


load_hf_dataset()
