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
import json
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Dict

import pandas as pd
from zenml import log_metadata, step


@step
def generate_compliance_metadata(
    dataset_trn: pd.DataFrame,
    dataset_tst: pd.DataFrame,
    original_train_df: pd.DataFrame,
    original_test_df: pd.DataFrame,
    preprocessing_metadata: Dict[str, Any],
    target: str,
    random_state: int,
) -> Annotated[Dict[str, Any], "compliance_info"]:
    """Generate compliance documentation for EU AI Act requirements.

    This step handles documentation for Articles 10 (Data Governance),
    12 (Record-keeping), and 15 (Accuracy).

    Args:
        dataset_trn: Preprocessed training dataframe
        dataset_tst: Preprocessed test dataframe
        original_train_df: Original training data before preprocessing
        original_test_df: Original test data before preprocessing
        preprocessing_metadata: Metadata from the preprocessing step
        target: Name of target column
        random_state: Random state used
    """
    # Generate feature metadata for documentation (Article 10)
    feature_metadata = {"target": target, "features": []}
    sensitive_attributes = []

    for col in dataset_trn.columns:
        if col == target:
            continue

        col_meta = {"name": col, "dtype": str(dataset_trn[col].dtype)}

        # Categorize as numerical or categorical
        if pd.api.types.is_numeric_dtype(dataset_trn[col]):
            col_meta.update(
                {
                    "type": "numerical",
                    "min": float(dataset_trn[col].min()),
                    "max": float(dataset_trn[col].max()),
                    "mean": float(dataset_trn[col].mean()),
                    "std": float(dataset_trn[col].std()),
                }
            )
        else:
            col_meta.update(
                {
                    "type": "categorical",
                    "categories": dataset_trn[col].unique().tolist(),
                    "most_common": dataset_trn[col].value_counts().index[0]
                    if len(dataset_trn[col].value_counts()) > 0
                    else None,
                }
            )

        # Mark crypto-specific sensitive attributes that may have fairness implications
        if any(
            term in col.lower()
            for term in [
                "wallet_age",
                "total_balance",
                "incoming_tx_sum",
                "outgoing_tx_sum",
                "max_eth_ever",
                "first_tx",
                "balance",
                "net_",
                "risk_factor",
            ]
        ):
            col_meta["sensitive_attribute"] = True
            sensitive_attributes.append(col)

        feature_metadata["features"].append(col_meta)

    # Create comprehensive preprocessing documentation for compliance (Article 12)
    compliance_info = {
        "version": "1.0.0",  # Documentation version
        "dataset_type": "cryptocurrency_lending",  # Domain-specific metadata
        "preprocessing_details": preprocessing_metadata,
        "feature_count": len(dataset_trn.columns),
        "target_column": target,
        "sensitive_attributes": sensitive_attributes,
        "random_state": random_state,
        "data_quality": {
            "missing_values_original": {
                "train": int(original_train_df.isna().sum().sum()),
                "test": int(original_test_df.isna().sum().sum()),
            },
            "missing_values_processed": {
                "train": int(dataset_trn.isna().sum().sum()),
                "test": int(dataset_tst.isna().sum().sum()),
            },
            "shape_changes": {
                "train": {
                    "original": original_train_df.shape,
                    "processed": dataset_trn.shape,
                },
                "test": {
                    "original": original_test_df.shape,
                    "processed": dataset_tst.shape,
                },
            },
        },
        "eu_ai_act_articles": {
            "article_10": "Data governance documentation - records all transformations",
            "article_12": "Record-keeping - maintains detailed preprocessing history",
            "article_15": "Accuracy - documents data quality improvements",
        },
    }

    # Save compliance info to file for audit records
    compliance_dir = Path("compliance/records")
    compliance_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    with open(compliance_dir / f"compliance_record_{timestamp}.json", "w") as f:
        json.dump(compliance_info, f, indent=2)

    # log metadata for the pipeline
    log_metadata(
        metadata={
            "compliance_record": compliance_info,
            "feature_metadata": feature_metadata,
            "random_state": random_state,
            "target": target,
            "sensitive_attributes": sensitive_attributes,
            "timestamp": timestamp,
        }
    )

    # Log documentation for audit purposes
    print(f"Compliance documentation saved: compliance_record_{timestamp}.json")
    print("Articles covered: 10 (Data Governance), 12 (Record-keeping), 15 (Accuracy)")

    return compliance_info
