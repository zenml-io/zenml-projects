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

from typing import Tuple

from datasets import Dataset
from typing_extensions import Annotated
from zenml import step


@step
def data_splitter(
    dataset: Dataset,
    test_size: float,
    validation_size: float,
    seed: int = 42,
) -> Tuple[
    Annotated[Dataset, "train_set"],
    Annotated[Dataset, "validation_set"],
    Annotated[Dataset, "test_set"],
]:
    """Performs stratified dataset splitting.

    Args:
        dataset: Input dataset to split
        test_size: Fraction for test+validation
        validation_size: Fraction of test for validation
        seed: Random seed for reproducibility

    Returns:
        Tuple[Dataset, Dataset, Dataset]: Train, validation and test splits
    """
    train_test = dataset.train_test_split(
        test_size=test_size,
        stratify_by_column="label",
        seed=seed,
    )

    test_valid = train_test["test"].train_test_split(
        test_size=validation_size,
        stratify_by_column="label",
        seed=seed,
    )

    return (
        train_test["train"],  # train set
        test_valid["train"],  # validation set
        test_valid["test"],  # test set
    )
