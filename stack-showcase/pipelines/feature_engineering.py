# {% include 'template/license_header' %}

import random
from typing import List, Optional

from steps import (
    data_loader,
    data_preprocessor,
    data_splitter,
)
from zenml import pipeline
from zenml.logger import get_logger

logger = get_logger(__name__)


@pipeline
def feature_engineering(
    test_size: float = 0.2,
    drop_na: Optional[bool] = None,
    normalize: Optional[bool] = None,
    drop_columns: Optional[List[str]] = None,
    target: Optional[str] = "target",
):
    """
    Feature engineering pipeline.

    This is a pipeline that loads the data, processes it and splits
    it into train and test sets.

    Args:
        test_size: Size of holdout set for training 0.0..1.0
        drop_na: If `True` NA values will be removed from dataset
        normalize: If `True` dataset will be normalized with MinMaxScaler
        drop_columns: List of columns to drop from dataset
        target: Name of target column in dataset
    """
    ### ADD YOUR OWN CODE HERE - THIS IS JUST AN EXAMPLE ###
    # Link all the steps together by calling them and passing the output
    # of one step as the input of the next step.
    raw_data = data_loader(random_state=random.randint(0, 100), target=target)
    dataset_trn, dataset_tst = data_splitter(
        dataset=raw_data,
        test_size=test_size,
    )
    dataset_trn, dataset_tst, _ = data_preprocessor(
        dataset_trn=dataset_trn,
        dataset_tst=dataset_tst,
        drop_na=drop_na,
        normalize=normalize,
        drop_columns=drop_columns,
        target=target,
    )
    return dataset_trn, dataset_tst
