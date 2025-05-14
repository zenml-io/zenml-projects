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
from typing import Annotated, Any, Dict, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline as SkPipeline
from zenml import log_metadata, step

from src.constants import (
    MODAL_PREPROCESS_PIPELINE_PATH,
    PREPROCESS_METADATA_NAME,
    PREPROCESS_PIPELINE_NAME,
    TARGET_COLUMN,
    TEST_DATASET_NAME,
    TRAIN_DATASET_NAME,
)
from src.utils import save_artifact_to_modal
from src.utils.preprocess import DeriveAgeFeatures, DropIDColumn, SimpleScaler


@step
def data_preprocessor(
    dataset_trn: pd.DataFrame,
    dataset_tst: pd.DataFrame,
    target: str = TARGET_COLUMN,
    normalize: bool = True,
) -> Tuple[
    Annotated[pd.DataFrame, TRAIN_DATASET_NAME],
    Annotated[pd.DataFrame, TEST_DATASET_NAME],
    Annotated[SkPipeline, PREPROCESS_PIPELINE_NAME],
    Annotated[Dict[str, Any], PREPROCESS_METADATA_NAME],
]:
    """Data preprocessor step that focuses purely on data transformation.

    Implements transformations with EU AI Act compliance documentation.

    Args:
        dataset_trn: Training dataset
        dataset_tst: Test dataset
        target: Target column name
        normalize: Whether to normalize numerical features

    Returns:
        Processed datasets, the fitted pipeline, and preprocessing metadata
    """
    # Initialize processing log for compliance documentation
    log = []
    start = datetime.now().isoformat()

    # 1. Drop the ID
    dropper = DropIDColumn()
    dataset_trn = dropper.transform(dataset_trn)
    dataset_tst = dropper.transform(dataset_tst)
    log.append({"op": "drop_id", "at": datetime.now().isoformat()})

    # 2. Derive age/employment
    age_transformer = DeriveAgeFeatures()
    dataset_trn = age_transformer.transform(dataset_trn)
    dataset_tst = age_transformer.transform(dataset_tst)
    log.append({"op": "derive_age", "at": datetime.now().isoformat()})

    # 3. Build scaling step if asked
    transformers = []
    if normalize:
        transformers.append(
            (
                "scale",
                SimpleScaler(exclude=[target]),
                make_column_selector(dtype_include=["number"], pattern=f"^(?!{target}$).*"),
            )
        )
    log.append({"op": "scale", "at": datetime.now().isoformat()})

    ct = ColumnTransformer(transformers, remainder="passthrough", verbose_feature_names_out=True)
    pipeline = SkPipeline([("column_transformer", ct)])
    pipeline.fit(dataset_trn)

    # 4. Transform & collect feature names
    cols = ct.get_feature_names_out(dataset_trn.columns)
    train_df = pd.DataFrame(pipeline.transform(dataset_trn), columns=cols)
    test_df = pd.DataFrame(pipeline.transform(dataset_tst), columns=cols)

    #  5. Log metadata for Annex IV
    metadata = {
        "preprocessing_start": start,
        "preprocessing_end": datetime.now().isoformat(),
        "steps": log,
        "train_shape": train_df.shape,
        "test_shape": test_df.shape,
        "features": cols,
        "target": target,
    }
    log_metadata(metadata)

    # 6. Persist pipeline for deployment
    save_artifact_to_modal(
        artifact=pipeline,
        artifact_path=MODAL_PREPROCESS_PIPELINE_PATH,
    )

    return train_df, test_df, pipeline, metadata
