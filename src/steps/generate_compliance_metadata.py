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
from typing import Annotated, Any, Dict, Optional

import pandas as pd
from whylogs.core import DatasetProfileView
from zenml import log_metadata, step

from src.constants import (
    COMPLIANCE_METADATA_NAME,
    DATA_PROFILE_NAME,
    SENSITIVE_ATTRIBUTES,
    TEST_DATASET_NAME,
    TRAIN_DATASET_NAME,
)


@step
def generate_compliance_metadata(
    train_df: Annotated[pd.DataFrame, TRAIN_DATASET_NAME],
    test_df: Annotated[pd.DataFrame, TEST_DATASET_NAME],
    preprocessing_metadata: Dict[str, Any],
    target: str,
    data_profile: Annotated[Optional[DatasetProfileView], DATA_PROFILE_NAME] = None,
) -> Annotated[Dict[str, Any], COMPLIANCE_METADATA_NAME]:
    """Generate compliance documentation for EU AI Act requirements.

      Records:
      - Feature list & types (Article 10)
      - Pre/post shapes & missing‐value counts (Articles 10 & 12)
      - Data quality summary (Article 15)

    Args:
        train_df: Preprocessed training dataframe
        test_df: Preprocessed test dataframe
        preprocessing_metadata: Metadata from the preprocessing step
        target: Name of target column
        data_profile: WhyLogs data profile (optional)

    Returns:
        Compliance metadata
    """
    # 1. Dataset stats
    stats = {
        "train_shape": train_df.shape,
        "test_shape": test_df.shape,
        "missing": {
            "train": int(train_df.isna().sum().sum()),
            "test": int(test_df.isna().sum().sum()),
        },
    }

    # 2. Feature metadata
    features = []
    for col in train_df.columns:
        if col == target:
            continue
        col_ser = train_df[col]
        entry = {
            "name": col,
            "dtype": str(col_ser.dtype),
            "type": "numerical" if pd.api.types.is_numeric_dtype(col_ser) else "categorical",
        }
        features.append(entry)

    # 3. Optional WhyLogs summary
    profile_summary = None
    if data_profile:
        profile_summary = {
            c: {
                "non_null": data_profile.column_profile(c).counts.get("non_null", 0),
                "unique": data_profile.column_profile(c).metrics.get("unique_count"),
            }
            for c in data_profile.columns
        }

    # 4. Assemble compliance record
    compliance_record = {
        "timestamp": datetime.now().isoformat(),
        "preprocessing": preprocessing_metadata,
        "stats": stats,
        "features": features,
        "target": target,
    }

    if profile_summary:
        compliance_record["data_profile_summary"] = profile_summary

    # 5. Log metadata for Annex IV
    log_metadata(
        metadata={
            COMPLIANCE_METADATA_NAME: compliance_record,
        }
    )

    return compliance_record
