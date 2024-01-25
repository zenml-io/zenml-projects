# {% include 'template/license_header' %}

import pandas as pd
from sklearn.datasets import load_breast_cancer
from typing_extensions import Annotated
from zenml import log_artifact_metadata, step
from zenml.logger import get_logger

logger = get_logger(__name__)


@step
def data_loader(
    random_state: int, is_inference: bool = False, target: str = "target"
) -> Annotated[pd.DataFrame, "dataset"]:
    """Dataset reader step.

    This is an example of a dataset reader step that load Breast Cancer dataset.

    This step is parameterized, which allows you to configure the step
    independently of the step code, before running it in a pipeline.
    In this example, the step can be configured with number of rows and logic
    to drop target column or not. See the documentation for more information:

        https://docs.zenml.io/user-guide/advanced-guide/configure-steps-pipelines

    Args:
        is_inference: If `True` subset will be returned and target column
            will be removed from dataset.
        random_state: Random state for sampling
        target: Name of target columns in dataset.

    Returns:
        The dataset artifact as Pandas DataFrame and name of target column.
    """
    ### ADD YOUR OWN CODE HERE - THIS IS JUST AN EXAMPLE ###
    dataset = load_breast_cancer(as_frame=True)
    inference_size = int(len(dataset.target) * 0.05)
    dataset: pd.DataFrame = dataset.frame
    inference_subset = dataset.sample(inference_size, random_state=random_state)
    if is_inference:
        dataset = inference_subset
        dataset.drop(columns=target, inplace=True)
    else:
        dataset.drop(inference_subset.index, inplace=True)
    dataset.reset_index(drop=True, inplace=True)
    logger.info(f"Dataset with {len(dataset)} records loaded!")

    # Recording metadata for this dataset
    log_artifact_metadata(metadata={"random_state": random_state, target: target})

    ### YOUR CODE ENDS HERE ###
    return dataset
