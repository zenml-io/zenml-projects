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

import hashlib
from datetime import datetime
from typing import Annotated, Optional, Tuple

import pandas as pd
import whylogs as why
from datasets import load_dataset
from whylogs.core import DatasetProfileView
from zenml import log_metadata, step

from src.constants import (
    CREDIT_SCORING_CSV_PATH,
    DATA_PROFILE_NAME,
    SENSITIVE_ATTRIBUTES,
    TARGET_COLUMN,
)


@step
def ingest(
    random_state: int = 42,
    target: str = TARGET_COLUMN,
    sample_fraction: Optional[float] = None,
    log_data_profile: bool = True,
) -> Tuple[
    Annotated[pd.DataFrame, "credit_scoring_df"],
    Annotated[Optional[DatasetProfileView], DATA_PROFILE_NAME],
]:
    """Ingest local credit_scoring.csv and log compliance metadata.

    EU AI Act Article 10 (Data Governance) and Article 12 (Record-keeping)
    compliance is implemented by:
    1. Calculating and storing dataset SHA-256 hash for provenance
    2. Creating WhyLogs profile for data quality documentation
    3. Capturing and storing detailed dataset metadata

    Args:
        random_state: Random state for sampling.
        target: Target column name, default is 'target'
        sample_fraction: Fraction of data to sample for inference.
        log_data_profile: Whether to log a WhyLogs profile of the data

    Returns:
        dataset: The loaded dataset
        profile_view: WhyLogs profile for data quality documentation
    """
    # Record start time for logging
    start_time = datetime.now()
    print(f"Ingesting data from {CREDIT_SCORING_CSV_PATH} at {start_time}")

    #  load the CSV
    df = pd.read_csv(CREDIT_SCORING_CSV_PATH)

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in {CREDIT_SCORING_CSV_PATH}")

    # optional stratified sample
    if sample_fraction and 0 < sample_fraction < 1:
        df = (
            df.groupby(target, group_keys=False)
            .apply(lambda g: g.sample(frac=sample_fraction, random_state=random_state))
            .reset_index(drop=True)
        )
        print(f"→ Stratified sample: {len(df)} rows")

    # provenance hash of file bytes (not object hash for scale)
    with open(CREDIT_SCORING_CSV_PATH, "rb") as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()

    dataset_stats = {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "missing_values": int(df.isna().sum().sum()),
        "memory_bytes": int(df.memory_usage(deep=True).sum()),
    }

    # identify sensitive columns by substring (for fairness checks)
    sensitive_cols = [col for col in df.columns for term in SENSITIVE_ATTRIBUTES if term in col]

    # dataset info for compliance documentation
    dataset_info = {
        "name": CREDIT_SCORING_CSV_PATH,
        "source": CREDIT_SCORING_CSV_PATH,
        "ingestion_time": start_time.isoformat(),
        "sha256": file_hash,
        **dataset_stats,
        "column_names": df.columns.tolist(),
        "sensitive_attributes": sensitive_cols,
    }

    # Get a timestamp string for the metadata key
    timestamp = start_time.strftime("%Y%m%d_%H%M%S")

    # WhyLogs profile for data quality documentation
    profile: DatasetProfileView | None = None
    if log_data_profile:
        profile = why.log(df).view()

    log_metadata(
        metadata={
            "timestamp": timestamp,
            "dataset_info": dataset_info,
            "whylogs_profile": str(profile) if profile else None,
        }
    )

    print(f"Ingestion completed at {datetime.now()}, SHA-256: {file_hash}")

    return df, profile if profile else None
