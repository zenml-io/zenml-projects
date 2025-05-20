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

from typing import Annotated, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from zenml import step


@step
def data_splitter(
    dataset: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[
    Annotated[pd.DataFrame, "raw_dataset_trn"],
    Annotated[pd.DataFrame, "raw_dataset_tst"],
]:
    """Dataset splitter step.

    Split the dataset into train and test sets.

    Args:
        dataset: Dataset read from source.
        test_size: 0.0..1.0 defining portion of test set.
        random_state: Random state for reproducibility.
        sample_fraction: Optional fraction of data to sample for inference.

    Returns:
        The split dataset: dataset_trn, dataset_tst.
    """
    dataset_trn, dataset_tst = train_test_split(
        dataset,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )
    dataset_trn = pd.DataFrame(dataset_trn, columns=dataset.columns)
    dataset_tst = pd.DataFrame(dataset_tst, columns=dataset.columns)
    return dataset_trn, dataset_tst
