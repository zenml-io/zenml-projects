#  Copyright (c) ZenML GmbH 2023. All Rights Reserved.
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

"""Data loader steps for the Iris classification pipeline."""

from enum import Enum
from typing import Optional
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split
from zenml.steps import BaseParameters, Output, step
import requests


DATASET_TARGET_COLUMN_NAME = "target"


def download_dataframe(
    version: str,
    bucket: str = "zenmlpublicdata",
) -> pd.DataFrame:
    url = f"https://{bucket}.s3.eu-central-1.amazonaws.com/gitflow_data/{version}.csv"
    df = pd.read_csv(url)
    return df


class DataLoaderStepParameters(BaseParameters):
    """Parameters for the data_loader step.

    Attributes:
        version: data version to load. If not set, the step loads the original
            dataset shipped with scikit-learn. If a version is supplied, the
            step loads the datasets with the given version stored in the public
            S3 bucket.
    """

    version: Optional[str] = None


@step
def data_loader(
    params: DataLoaderStepParameters,
) -> pd.DataFrame:
    """Load the dataset with the indicated version.
    
    Args:
        params: Parameters for the data_loader step (data version to load).

    Returns:
        The dataset with the indicated version.
    """
    if params.version is None:
        # We use the original data shipped with scikit-learn for experimentation
        dataset = load_breast_cancer(as_frame=True).frame
        return dataset

    else:
        # We use data stored in the public S3 bucket for specified versions
        dataset = download_dataframe(version=params.version)
        return dataset


class DataSplitterStepParameters(BaseParameters):
    """Parameters for the data_splitter step.

    Attributes:
        test_size: Proportion of the dataset to include in the test split.
        shuffle: Whether or not to shuffle the data before splitting.
        random_state: Controls the shuffling applied to the data before
            applying the split. Pass an int for reproducible and cached output
            across multiple step runs.
    """

    test_size: float = 0.2
    shuffle: bool = True
    random_state: int = 42


@step
def data_splitter(
    params: DataSplitterStepParameters,
    dataset: pd.DataFrame, 
) -> Output(train=pd.DataFrame, test=pd.DataFrame,):
    """Split the dataset into train and test (validation) subsets.

    Args:
        params: Parameters for the data_splitter step (split proportions,
            shuffling, random state).
        dataset: The dataset to split.
    
    Returns:
        The train and test (validation) subsets of the dataset.
    """
    train, test = train_test_split(
        dataset,
        test_size=params.test_size,
        shuffle=params.shuffle,
        random_state=params.random_state,
    )
    return train, test
