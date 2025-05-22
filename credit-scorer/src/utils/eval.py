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

from typing import Any, Dict, List, Tuple

import pandas as pd
from fairlearn.metrics import MetricFrame, selection_rate
from sklearn.metrics import accuracy_score
from zenml.logger import get_logger

from src.constants import Artifacts as A

logger = get_logger(__name__)


def get_sensitive_feature(
    test_df: pd.DataFrame, attr: str
) -> Tuple[pd.Series, str, bool]:
    """Get the sensitive feature for fairness analysis.

    Args:
        test_df: Test dataset
        attr: Protected attribute name

    Returns:
        Tuple of (sensitive_feature_series, feature_name, skip_flag)
    """
    # Special case: DAYS_BIRTH not in dataset
    if attr == "DAYS_BIRTH" and "DAYS_BIRTH" not in test_df.columns:
        logger.info(f"Skipping {attr} - dropped during preprocessing")
        return None, attr, True

    # Case 1: Direct match
    if attr in test_df.columns:
        logger.info(f"Using direct column {attr}")
        return test_df[attr], attr, False

    # Case 2: Numerical with prefix
    if any(col == f"num__{attr}" for col in test_df.columns):
        col_name = f"num__{attr}"
        logger.info(f"Using numerical column {col_name}")
        return test_df[col_name], col_name, False

    # Case 3: Categorical - reconstruct from one-hot encoding
    categorical_cols = [
        col for col in test_df.columns if col.startswith(f"cat__{attr}_")
    ]
    if categorical_cols:
        logger.info(f"Reconstructing {attr} from one-hot encoded columns")

        # Reconstruct categories
        cat_values = {}
        for col in categorical_cols:
            category = col.split(f"cat__{attr}_")[1]
            for idx, val in enumerate(test_df[col]):
                if val == 1.0:
                    cat_values[idx] = category

        # Create Series
        sensitive_features = pd.Series(
            [cat_values.get(i, "Unknown") for i in range(len(test_df))],
            name=attr,
        )
        return sensitive_features, attr, False

    # Case 4: Not found
    logger.warning(
        f"Protected attribute '{attr}' not found in transformed dataset"
    )
    return None, attr, True


def calculate_fairness_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    sensitive_features: pd.Series,
) -> Dict[str, Any]:
    """Calculate fairness metrics for a protected attribute.

    Args:
        y_true: True target values
        y_pred: Predicted target values
        sensitive_features: Protected attribute values
        feature_name: Name of the protected attribute

    Returns:
        Dictionary of fairness metrics
    """
    frame = MetricFrame(
        metrics={"selection_rate": selection_rate, "accuracy": accuracy_score},
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_features,
    )

    disparity = frame.difference(method="between_groups")["selection_rate"]

    return {
        "selection_rate_by_group": frame.by_group["selection_rate"].to_dict(),
        "accuracy_by_group": frame.by_group["accuracy"].to_dict(),
        "selection_rate_disparity": disparity,
    }


def analyze_fairness(
    y_test: pd.Series,
    y_pred: pd.Series,
    protected_attributes: List[str],
    test_df: pd.DataFrame,
    approval_thresholds: Dict[str, float],
) -> Tuple[Dict[str, Dict], bool]:
    """Analyze fairness across protected attributes.

    Args:
        y_test: True target values
        y_pred: Predicted target values
        protected_attributes: List of protected attribute names
        test_df: Test dataset with all columns
        approval_thresholds: Dictionary of approval thresholds for fairness metrics

    Returns:
        Tuple of (fairness_report, bias_flag)
    """
    fairness_report = {}
    bias_flag = False

    for attr in protected_attributes:
        # Get the sensitive feature
        sensitive_features, feature_name, skip = get_sensitive_feature(
            test_df, attr
        )
        if skip:
            continue

        # Calculate fairness metrics
        metrics = calculate_fairness_metrics(
            y_test, y_pred, sensitive_features
        )
        fairness_report[feature_name] = metrics

        # Check if bias threshold is exceeded
        disparity = metrics["selection_rate_disparity"]
        if abs(disparity) > approval_thresholds["bias_disparity"]:
            bias_flag = True

    return fairness_report, bias_flag
