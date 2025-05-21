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
from datetime import datetime
from typing import Annotated, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from zenml import log_metadata, step
from zenml.logger import get_logger

from src.constants import (
    PREPROCESS_PIPELINE_NAME,
    TEST_DATASET_NAME,
    TRAIN_DATASET_NAME,
)
from src.utils import save_artifact_to_modal
from src.utils.preprocess import DeriveAgeFeatures, DropIDColumn

logger = get_logger(__name__)


def log_op(name, **extra):
    """Log an operation with a timestamp and additional metadata."""
    return {"op": name, "timestamp": datetime.now().isoformat(), **extra}


@step
def data_preprocessor(
    dataset_trn: pd.DataFrame,
    dataset_tst: pd.DataFrame,
    target: str = "TARGET",
    normalize: bool = True,
    preprocess_pipeline_path: str = "pipelines/preprocess_pipeline.pkl",
) -> Tuple[
    Annotated[pd.DataFrame, TRAIN_DATASET_NAME],
    Annotated[pd.DataFrame, TEST_DATASET_NAME],
    Annotated[SkPipeline, PREPROCESS_PIPELINE_NAME],
]:
    """Data preprocessor step that focuses purely on data transformation.

    Implements transformations with EU AI Act compliance documentation.

    Args:
        dataset_trn: Training dataset
        dataset_tst: Test dataset
        target: Target column name
        normalize: Whether to normalize numerical features
        preprocess_pipeline_path: Path to save the preprocess pipeline

    Returns:
        Processed datasets, the fitted pipeline, and preprocessing metadata
    """
    # Initialize processing log for compliance documentation
    log = []
    start = datetime.now().isoformat()

    # -- 1. Drop the ID -------------------------------------------------------
    dropper = DropIDColumn()
    dataset_trn = dropper.transform(dataset_trn)
    dataset_tst = dropper.transform(dataset_tst)
    log.append(log_op("drop_id"))

    # -- 2. Derive age/employment ---------------------------------------------
    age_transformer = DeriveAgeFeatures()
    dataset_trn = age_transformer.transform(dataset_trn)
    dataset_tst = age_transformer.transform(dataset_tst)
    log.append(log_op("derive_age"))

    # -- 3. Build preprocessing pipeline --------------------------------------
    numeric_selector = make_column_selector(
        dtype_include=["int64", "float64"],
        pattern=f"^(?!{target}$).*",  # Don't process target
    )
    categorical_selector = make_column_selector(
        dtype_include=["object", "category"]
    )

    # Get column lists for logging
    num_cols = numeric_selector(dataset_trn)
    cat_cols = categorical_selector(dataset_trn)

    # Add transformers
    transformers = []

    # Numeric pipeline with imputation and optional scaling
    num_pipeline_steps = [("impute", SimpleImputer(strategy="mean"))]
    if normalize:
        num_pipeline_steps.append(("scale", StandardScaler()))
        log.append(log_op("scale", columns=num_cols))

    if num_cols:
        transformers.append(
            ("num", SkPipeline(num_pipeline_steps), numeric_selector)
        )
        log.append(log_op("impute_numeric", columns=num_cols))

    # Categorical pipeline with imputation and encoding
    if cat_cols:
        transformers.append(
            (
                "cat",
                SkPipeline(
                    [
                        (
                            "impute",
                            SimpleImputer(
                                strategy="constant", fill_value="missing"
                            ),
                        ),
                        (
                            "ohe",
                            OneHotEncoder(
                                sparse_output=False, handle_unknown="ignore"
                            ),
                        ),
                    ]
                ),
                categorical_selector,
            )
        )
        log.append(log_op("one_hot", columns=cat_cols))

    # Create column transformer
    ct = ColumnTransformer(
        transformers,
        remainder="passthrough",
        verbose_feature_names_out=True,
    )

    # -- 4. Fit & transform ---------------------------------------------------
    sk_pipeline = SkPipeline([("preprocessor", ct)])
    sk_pipeline.fit(dataset_trn)

    # Transform data
    train_transformed = sk_pipeline.transform(dataset_trn)
    test_transformed = sk_pipeline.transform(dataset_tst)

    # Get feature names
    feature_names = sk_pipeline.named_steps[
        "preprocessor"
    ].get_feature_names_out(dataset_trn.columns)

    # Create dataframes
    train_df = pd.DataFrame(train_transformed, columns=feature_names)
    test_df = pd.DataFrame(test_transformed, columns=feature_names)

    # Log data types after transformation for verification
    logger.info(f"Data types after transformation: {train_df.dtypes}")

    # Verify no object/string columns remain
    remaining_cat_cols = train_df.select_dtypes(
        include=["object"]
    ).columns.tolist()
    if remaining_cat_cols:
        logger.warning(
            f"Warning: These columns are still categorical after transformation: {remaining_cat_cols}"
        )

    # -- 5. Log metadata for Annex IV -----------------------------------------
    preprocessing_metadata = {
        "preprocessing_start": start,
        "preprocessing_end": datetime.now().isoformat(),
        "steps": log,
        "train_shape": tuple(int(x) for x in train_df.shape),
        "test_shape": tuple(int(x) for x in test_df.shape),
        "features": feature_names.tolist(),
        "target": target,
        "numeric_columns": num_cols,
        "categorical_columns": cat_cols,
        "missing": {
            "train": int(train_df.isna().sum().sum()),
            "test": int(test_df.isna().sum().sum()),
        },
    }
    log_metadata(metadata=preprocessing_metadata)

    # -- 6. Persist pipeline for deployment ------------------------------------
    save_artifact_to_modal(
        artifact=sk_pipeline,
        artifact_path=preprocess_pipeline_path,
    )

    return train_df, test_df, sk_pipeline
