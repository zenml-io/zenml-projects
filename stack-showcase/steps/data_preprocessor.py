# {% include 'template/license_header' %}

from typing import Union
import pandas as pd
from typing import List, Optional, Tuple

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from typing_extensions import Annotated
from zenml import log_artifact_metadata, step


class NADropper:
    """Support class to drop NA values in sklearn Pipeline."""

    def fit(self, *args, **kwargs):
        return self

    def transform(self, X: Union[pd.DataFrame, pd.Series]):
        return X.dropna()


class ColumnsDropper:
    """Support class to drop specific columns in sklearn Pipeline."""

    def __init__(self, columns):
        self.columns = columns

    def fit(self, *args, **kwargs):
        return self

    def transform(self, X: Union[pd.DataFrame, pd.Series]):
        return X.drop(columns=self.columns)


class DataFrameCaster:
    """Support class to cast type back to pd.DataFrame in sklearn Pipeline."""

    def __init__(self, columns):
        self.columns = columns

    def fit(self, *args, **kwargs):
        return self

    def transform(self, X):
        return pd.DataFrame(X, columns=self.columns)


@step
def data_preprocessor(
    dataset_trn: pd.DataFrame,
    dataset_tst: pd.DataFrame,
    drop_na: Optional[bool] = None,
    normalize: Optional[bool] = None,
    drop_columns: Optional[List[str]] = None,
    target: Optional[str] = "target",
) -> Tuple[
    Annotated[pd.DataFrame, "dataset_trn"],
    Annotated[pd.DataFrame, "dataset_tst"],
    Annotated[Pipeline, "preprocess_pipeline"],
]:
    """Data preprocessor step.

    This is an example of a data processor step that prepares the data so that
    it is suitable for model training. It takes in a dataset as an input step
    artifact and performs any necessary preprocessing steps like cleaning,
    feature engineering, feature selection, etc. It then returns the processed
    dataset as an step output artifact.

    This step is parameterized, which allows you to configure the step
    independently of the step code, before running it in a pipeline.
    In this example, the step can be configured to drop NA values, drop some
    columns and normalize numerical columns. See the documentation for more
    information:

        https://docs.zenml.io/user-guide/advanced-guide/configure-steps-pipelines

    Args:
        dataset_trn: The train dataset.
        dataset_tst: The test dataset.
        drop_na: If `True` all NA rows will be dropped.
        normalize: If `True` all numeric fields will be normalized.
        drop_columns: List of column names to drop.

    Returns:
        The processed datasets (dataset_trn, dataset_tst) and fitted `Pipeline` object.
    """
    ### ADD YOUR OWN CODE HERE - THIS IS JUST AN EXAMPLE ###
    # We use the sklearn pipeline to chain together multiple preprocessing steps
    preprocess_pipeline = Pipeline([("passthrough", "passthrough")])
    if drop_na:
        preprocess_pipeline.steps.append(("drop_na", NADropper()))
    if drop_columns:
        # Drop columns
        preprocess_pipeline.steps.append(("drop_columns", ColumnsDropper(drop_columns)))
    if normalize:
        # Normalize the data
        preprocess_pipeline.steps.append(("normalize", MinMaxScaler()))
    preprocess_pipeline.steps.append(("cast", DataFrameCaster(dataset_trn.columns)))
    dataset_trn = preprocess_pipeline.fit_transform(dataset_trn)
    dataset_tst = preprocess_pipeline.transform(dataset_tst)

    # Log metadata of target to both datasets
    log_artifact_metadata(
        artifact_name="dataset_trn",
        metadata={"target": target},
    )
    log_artifact_metadata(
        artifact_name="dataset_tst",
        metadata={"target": target},
    )

    ### YOUR CODE ENDS HERE ###
    return dataset_trn, dataset_tst, preprocess_pipeline
