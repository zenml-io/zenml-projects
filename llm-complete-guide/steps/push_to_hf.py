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
from zenml.client import Client

from constants import DATASET_NAME_DISTILABEL, SECRET_NAME
from datasets import Dataset, DatasetDict
from zenml import step


@step
def push_to_hf(train_dataset: Dataset, test_dataset: Dataset):
    zenml_client = Client()

    combined_dataset = DatasetDict(
        {"train": train_dataset, "test": test_dataset}
    )
    combined_dataset.push_to_hub(
        DATASET_NAME_DISTILABEL,
        token=zenml_client.get_secret(SECRET_NAME).secret_values["hf_token"],
    )
