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

from typing import List, Optional

from zenml.pipelines import pipeline

from src.constants import (
    FEATURE_ENGINEERING_PIPELINE_NAME,
    TARGET_COLUMN,
)
from src.steps import (
    data_preprocessor,
    data_splitter,
    generate_compliance_metadata,
    ingest,
)


@pipeline(name=FEATURE_ENGINEERING_PIPELINE_NAME)
def feature_engineering(
    test_size: float = 0.2,
    normalize: bool = True,
    target: str = TARGET_COLUMN,
    random_state: int = 42,
    sample_fraction: Optional[float] = None,
):
    """End-to-end pipeline for credit scoring with EU AI Act compliance.

    This pipeline handles:
    1. Data loading with provenance tracking (Article 10)
    2. Data splitting into train and test sets
    3. Data preprocessing with documented decisions (Article 10)
    """
    # Initialize volume metadata if not provided

    # Load the data
    raw_data, data_profile, whylogs_visualization = ingest(
        target=target,
        random_state=random_state,
        sample_fraction=sample_fraction,
    )

    # Split the data into train and test sets
    dataset_trn, dataset_tst = data_splitter(
        random_state=random_state,
        dataset=raw_data,
        test_size=test_size,
    )

    # Preprocess the data
    train_df, test_df, sk_pipeline, preprocessing_metadata = data_preprocessor(
        dataset_trn=dataset_trn,
        dataset_tst=dataset_tst,
        target=target,
        normalize=normalize,
    )

    # Generate compliance documentation
    compliance_info = generate_compliance_metadata(
        train_df=train_df,
        test_df=test_df,
        preprocessing_metadata=preprocessing_metadata,
        target=target,
        data_profile=data_profile,
    )

    return train_df, test_df, sk_pipeline, compliance_info
