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
import os
from datetime import datetime
from pathlib import Path
from typing import Annotated, Optional

import pandas as pd
import whylogs as why
from datasets import load_dataset
from whylogs.core import DatasetProfileView
from zenml import log_metadata, step

from constants import HF_DATASET_FILE, HF_DATASET_NAME
from utils.preprocess import to_native

# ---------- Named output aliases -------------------------------------------
# RawDF = Annotated[pd.DataFrame, "dataset"]
# DatasetInfo = Annotated[Dict[str, Any], "dataset_info"]
# WhyProfile = Annotated[DatasetProfileView, "dataset_profile"]
# ---------------------------------------------------------------------------


@step
def data_loader(
    random_state: int = 42,
    target: str = "target",
    sample_fraction: Optional[float] = None,
    log_data_profile: bool = True,
) -> Annotated[pd.DataFrame, "credit_scoring_df"]:
    """Ingests credit scoring dataset and logs compliance metadata.

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
        dataset_info: Metadata about the dataset for compliance documentation
        dataset_profile: WhyLogs profile for data quality documentation
    """
    # Record start time for logging
    start_time = datetime.now()
    print(f"Loading dataset {HF_DATASET_NAME}/{HF_DATASET_FILE} at {start_time}")

    # --- load --------------------------------------------------------------
    # df = pd.read_parquet()
    ds = load_dataset(HF_DATASET_NAME, split="train")
    df = ds.to_pandas()

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in {HF_DATASET_NAME}")

    # ---------- optional sampling -----------------------------------------
    if sample_fraction and 0 < sample_fraction < 1:
        # Use stratified sampling to maintain class distribution
        df = (
            df.groupby(target, group_keys=False)
            .apply(lambda g: g.sample(frac=sample_fraction, random_state=random_state))
            .reset_index(drop=True)
        )
        print(f"Performed stratified sampling, new size: {len(df)} rows")

    # ---------- hash + basic stats ----------------------------------------
    sha256 = hashlib.sha256(pd.util.hash_pandas_object(df).values).hexdigest()
    dataset_stats = {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "missing_values": int(df.isna().sum().sum()),
        "memory_bytes": int(df.memory_usage(deep=True).sum()),
    }

    # --- potential sensitive attributes (for fairness checks)  -------------
    sensitive_attrs = [
        c
        for c in df.columns
        if c.lower() in {"gender", "race", "ethnicity", "age", "nationality", "marital_status"}
    ]

    # --- dataset info for compliance documentation  -------------------------
    dataset_info = {
        "name": HF_DATASET_NAME,
        "source": HF_DATASET_NAME,
        "ingestion_time": start_time.isoformat(),
        "sha256": sha256,
        **dataset_stats,
        "column_names": df.columns.tolist(),
        "potential_sensitive_attributes": sensitive_attrs,
    }

    # Get a timestamp string for the metadata key
    timestamp = start_time.strftime("%Y%m%d_%H%M%S")

    # --- WhyLogs profile for data quality documentation  ---------------------
    profile_view: DatasetProfileView | None = None
    if log_data_profile:
        profile_view = why.log(df).view()
        out_dir = Path("reports/data_profiles")
        out_dir.mkdir(parents=True, exist_ok=True)
        # save dataset profile for compliance documentation (Article 10)
        profile_view.write(out_dir / f"profile_{timestamp}.bin")

    # --- compliance metadata --------------------------------------
    metadata = to_native(
        {
            "timestamp": timestamp,
            "data_snapshot": dataset_info,
            "sensitive_attrs": sensitive_attrs,
            "whylogs_profile": str(profile_view) if profile_view else None,
        }
    )
    log_metadata(metadata=metadata)

    # --- log completion for traceability  -----------------------------------
    print(f"Ingestion completed at {datetime.now()}, SHA-256: {sha256}")

    # --- return for pipeline  -----------------------------------------------
    return df
