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
from typing import Annotated, List, Optional, Tuple

import pandas as pd
import whylogs as why
from whylogs.core import DatasetProfileView
from zenml import log_metadata, step
from zenml.types import HTMLString

from src.constants import WHYLOGS_VISUALIZATION_NAME
from src.utils import generate_whylogs_visualization


@step(enable_cache=False)
def ingest(
    dataset_path: str,
    random_state: int = 42,
    target: str = "TARGET",
    sample_fraction: Optional[float] = None,
    sensitive_attributes: List[str] = None,
) -> Tuple[
    Annotated[pd.DataFrame, "credit_scoring_df"],
    Annotated[HTMLString, WHYLOGS_VISUALIZATION_NAME],
]:
    """Ingest local credit_scoring.csv and log compliance metadata.

    EU AI Act Article 10 (Data Governance) and Article 12 (Record-keeping)
    compliance is implemented by:
    1. Calculating and storing dataset SHA-256 hash for provenance
    2. Creating WhyLogs profile for data quality documentation
    3. Capturing and storing detailed dataset metadata

    Args:
        dataset_path: Path to the dataset to ingest.
        random_state: Random state for sampling.
        target: Target column name, default is 'target'
        sample_fraction: Fraction of data to sample for inference.
        sensitive_attributes: List of sensitive attributes to check for.

    Returns:
        dataset: The loaded dataset
        profile_view: WhyLogs profile for data quality documentation
    """
    # Record start time for logging
    start_time = datetime.now()
    print(f"Ingesting data from {dataset_path} at {start_time}")

    #  load the CSV
    df = pd.read_csv(dataset_path, low_memory=False)

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in {dataset_path}")

    # Clean data by removing rows with all or most values missing
    if "SK_ID_CURR" in df.columns:
        df = df.dropna(subset=["SK_ID_CURR"])

    # optional stratified sample
    if sample_fraction and 0 < sample_fraction < 1:
        df = (
            df.groupby(target, group_keys=False)
            .apply(lambda g: g.sample(frac=sample_fraction, random_state=random_state))
            .reset_index(drop=True)
        )
        print(f"→ Stratified sample: {len(df)} rows")

    # provenance hash of file bytes (not object hash for scale)
    with open(dataset_path, "rb") as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()

    dataset_stats = {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "missing_values": int(df.isna().sum().sum()),
        "memory_bytes": int(df.memory_usage(deep=True).sum()),
    }

    # identify sensitive columns by substring (for fairness checks)
    sensitive_cols = [col for col in df.columns for term in sensitive_attributes if term in col]

    # dataset info for compliance documentation
    dataset_info = {
        "name": dataset_path,
        "source": dataset_path,
        "ingestion_time": start_time.isoformat(),
        "sha256": file_hash,
        **dataset_stats,
        "column_names": df.columns.tolist(),
        "sensitive_attributes": sensitive_cols,
    }

    # WhyLogs profile for data quality documentation
    data_profile: DatasetProfileView | None = None
    data_profile = why.log(df).view()

    # Generate WhyLogs visualization
    whylogs_visualization = generate_whylogs_visualization(
        data_profile=data_profile,
        dataset_info=dataset_info,
    )

    log_metadata(
        metadata={
            "timestamp": start_time.strftime("%Y%m%d_%H%M%S"),
            "dataset_info": dataset_info,
        }
    )

    print(f"Ingestion completed at {datetime.now()}, SHA-256: {file_hash}")

    return df, whylogs_visualization
