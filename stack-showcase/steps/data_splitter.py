# {% include 'template/license_header' %}

from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from typing_extensions import Annotated
from zenml import step


@step
def data_splitter(
    dataset: pd.DataFrame, test_size: float = 0.2
) -> Tuple[
    Annotated[pd.DataFrame, "raw_dataset_trn"],
    Annotated[pd.DataFrame, "raw_dataset_tst"],
]:
    """Dataset splitter step.

    This is an example of a dataset splitter step that splits the data
    into train and test set before passing it to ML model.

    This step is parameterized, which allows you to configure the step
    independently of the step code, before running it in a pipeline.
    In this example, the step can be configured to use different test
    set sizes. See the documentation for more information:

        https://docs.zenml.io/user-guide/advanced-guide/configure-steps-pipelines

    Args:
        dataset: Dataset read from source.
        test_size: 0.0..1.0 defining portion of test set.

    Returns:
        The split dataset: dataset_trn, dataset_tst.
    """
    ### ADD YOUR OWN CODE HERE - THIS IS JUST AN EXAMPLE ###
    dataset_trn, dataset_tst = train_test_split(
        dataset,
        test_size=test_size,
        random_state=42,
        shuffle=True,
    )
    dataset_trn = pd.DataFrame(dataset_trn, columns=dataset.columns)
    dataset_tst = pd.DataFrame(dataset_tst, columns=dataset.columns)
    ### YOUR CODE ENDS HERE ###
    return dataset_trn, dataset_tst
