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
from typing import Annotated, Any, Dict, List, Optional, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import StandardScaler
from zenml import step


@step
def data_preprocessor(
    dataset_trn: pd.DataFrame,
    dataset_tst: pd.DataFrame,
    target: str = "target",
    normalize: bool = True,
    drop_na: bool = True,
    drop_columns: Optional[List[str]] = None,
    random_state: int = 42,
) -> Tuple[
    Annotated[pd.DataFrame, "train_df"],
    Annotated[pd.DataFrame, "test_df"],
    Annotated[SkPipeline, "preprocess_pipeline"],
    Annotated[Dict[str, Any], "preprocessing_metadata"],
]:
    """Data preprocessor step that focuses purely on data transformation.

    Uses sklearn's ColumnTransformer for preprocessing operations.

    Args:
        dataset_trn: Training dataset
        dataset_tst: Test dataset
        target: Target column name
        normalize: Whether to normalize numerical features
        drop_na: Whether to drop rows with missing values
        drop_columns: List of columns to drop
        random_state: Random state for reproducibility

    Returns:
        Processed datasets, the fitted pipeline, and preprocessing metadata
    """
    # Track preprocessing steps
    preprocessing_steps = []
    preprocessing_start = pd.Timestamp.now()

    # Initial checksums
    train_initial_checksum = pd.util.hash_pandas_object(dataset_trn).sum()
    test_initial_checksum = pd.util.hash_pandas_object(dataset_tst).sum()

    # 1) Handle missing values if requested
    if drop_na:
        dataset_trn = dataset_trn.dropna()
        dataset_tst = dataset_tst.dropna()
        preprocessing_steps.append(
            {"operation": "drop_na", "timestamp": pd.Timestamp.now().isoformat()}
        )

    # 2) Handle column dropping
    if drop_columns is None:
        drop_columns = []

    # Always drop wallet_address for privacy
    if "wallet_address" in dataset_trn.columns and "wallet_address" not in drop_columns:
        drop_columns.append("wallet_address")

    # Never drop the target
    drop_columns = [c for c in drop_columns if c != target and c in dataset_trn.columns]

    if drop_columns:
        dataset_trn = dataset_trn.drop(columns=drop_columns, errors="ignore")
        dataset_tst = dataset_tst.drop(columns=drop_columns, errors="ignore")
        preprocessing_steps.append(
            {
                "operation": "drop_columns",
                "columns": drop_columns,
                "timestamp": pd.Timestamp.now().isoformat(),
            }
        )

    # 3) Build the preprocessing pipeline
    transformers = []

    # Add normalization if requested
    if normalize:
        numeric_selector = make_column_selector(
            dtype_include=["int64", "float64"],
            pattern=f"^(?!{target}$).*",  # Don't normalize target
        )
        transformers.append(("scale", StandardScaler(), numeric_selector))
        preprocessing_steps.append(
            {
                "operation": "normalize",
                "method": "StandardScaler",
                "timestamp": pd.Timestamp.now().isoformat(),
            }
        )

    # Create the pipeline
    ct = ColumnTransformer(
        transformers=transformers,
        remainder="passthrough",
        verbose_feature_names_out=True,
    )
    pipeline = SkPipeline([("ct", ct)])

    # 4) Fit the pipeline and transform the data
    pipeline.fit(dataset_trn)
    feature_names = pipeline.named_steps["ct"].get_feature_names_out(dataset_trn.columns)

    train_df = pd.DataFrame(pipeline.transform(dataset_trn), columns=feature_names)
    test_df = pd.DataFrame(pipeline.transform(dataset_tst), columns=feature_names)

    # 5) Final checksums
    train_final_checksum = pd.util.hash_pandas_object(train_df).sum()
    test_final_checksum = pd.util.hash_pandas_object(test_df).sum()

    # Prepare metadata for compliance step
    preprocessing_metadata = {
        "preprocessing_start": preprocessing_start.isoformat(),
        "preprocessing_end": pd.Timestamp.now().isoformat(),
        "steps": preprocessing_steps,
        "initial_shapes": {
            "train": dataset_trn.shape,
            "test": dataset_tst.shape,
        },
        "final_shapes": {
            "train": train_df.shape,
            "test": test_df.shape,
        },
        "checksums": {
            "train_initial": str(train_initial_checksum),
            "test_initial": str(test_initial_checksum),
            "train_final": str(train_final_checksum),
            "test_final": str(test_final_checksum),
        },
        "pipeline_steps": [step[0] for step in pipeline.steps],
        "feature_names": feature_names.tolist(),
        "target": target,
        "random_state": random_state,
    }

    return train_df, test_df, pipeline, preprocessing_metadata
