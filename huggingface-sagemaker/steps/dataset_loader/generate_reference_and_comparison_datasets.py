# Apache Software License 2.0
#
# Copyright (c) ZenML GmbH 2023. All rights reserved.
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

import pandas as pd
from datasets import DatasetDict
from typing_extensions import Annotated, Tuple
from zenml import step
from zenml.logger import get_logger

logger = get_logger(__name__)


@step
def generate_reference_and_comparison_datasets(
    dataset: DatasetDict,
) -> Tuple[
    Annotated[pd.DataFrame, "reference_dataset"],
    Annotated[pd.DataFrame, "comparison_dataset"],
]:
    """A step to generate reference and comparison datasets.

    Args:
        dataset: A dataset dictionary.

    Returns:
        reference_dataset: A reference dataset.
        comparison_dataset: A comparison dataset.
    """
    reference_dataset = pd.DataFrame(
        {"label": dataset["train"]["label"], "text": dataset["train"]["text"]}
    )
    comparison_dataset = pd.DataFrame(
        {"label": dataset["test"]["label"], "text": dataset["test"]["text"]}
    )
    return reference_dataset, comparison_dataset
