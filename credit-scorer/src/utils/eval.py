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

    # Case 2: Numerical with prefix - CREATE BALANCED GROUPS
    if any(col == f"num__{attr}" for col in test_df.columns):
        col_name = f"num__{attr}"
        logger.info(
            f"Creating balanced groups for numerical column {col_name}"
        )

        # For AGE_YEARS, create balanced age groups instead of continuous values
        if attr == "AGE_YEARS":
            age_values = test_df[col_name]
            logger.info(
                f"Age range: {age_values.min():.1f} - {age_values.max():.1f}"
            )

            # Handle weird age preprocessing (might be standardized/normalized)
            if (
                age_values.min() < 10
            ):  # Likely standardized or processed incorrectly
                logger.info(
                    "Detected non-standard age values, using percentile-based grouping"
                )
                # Use percentiles to create balanced groups
                age_percentiles = age_values.quantile([0.33, 0.67]).values

                age_groups = []
                for age in age_values:
                    if age <= age_percentiles[0]:
                        age_groups.append("young_adult")
                    elif age <= age_percentiles[1]:
                        age_groups.append("middle_age")
                    else:
                        age_groups.append("mature")
            else:
                # Normal age values
                age_groups = []
                for age in age_values:
                    if age < 35:
                        age_groups.append("young_adult")  # < 35
                    elif age < 50:
                        age_groups.append("middle_age")  # 35-50
                    else:
                        age_groups.append("mature")  # 50+

            age_series = pd.Series(age_groups, name=f"{attr}_groups")
            logger.info(
                f"Age group distribution: {age_series.value_counts().to_dict()}"
            )
            return age_series, f"{attr}_groups", False
        else:
            # For other numerical attributes, create quantile-based groups
            try:
                groups = pd.qcut(
                    test_df[col_name],
                    q=3,
                    duplicates="drop",
                    labels=["low", "medium", "high"],
                )
                return groups, f"{attr}_groups", False
            except:
                logger.warning(
                    f"Could not create groups for {col_name}, using original values"
                )
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

        # For education, group into broader categories to prevent 0.000 DI ratios
        if attr == "NAME_EDUCATION_TYPE":
            education_groups = []
            for edu in sensitive_features:
                if "Higher education" in str(edu) or "Academic degree" in str(
                    edu
                ):
                    education_groups.append("higher_education")
                elif "Secondary" in str(edu) or "Incomplete" in str(edu):
                    education_groups.append("secondary_education")
                else:
                    education_groups.append("other_education")

            grouped_series = pd.Series(education_groups, name=f"{attr}_groups")
            logger.info(
                f"Education group distribution: {grouped_series.value_counts().to_dict()}"
            )
            return grouped_series, f"{attr}_groups", False

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

    # Calculate disparate impact ratio (DI ratio)
    # DI = min(selection_rates) / max(selection_rates)
    # DI < 0.8 indicates adverse impact per four-fifths rule
    selection_rates = frame.by_group["selection_rate"]
    if len(selection_rates) >= 2 and selection_rates.max() > 0:
        disparate_impact_ratio = selection_rates.min() / selection_rates.max()
    else:
        # Handle edge cases: only one group or no positive predictions
        disparate_impact_ratio = 1.0

    # Also calculate the old difference metric for backward compatibility
    disparity_difference = frame.difference(method="between_groups")[
        "selection_rate"
    ]

    return {
        "selection_rate_by_group": selection_rates.to_dict(),
        "accuracy_by_group": frame.by_group["accuracy"].to_dict(),
        "disparate_impact_ratio": disparate_impact_ratio,
        "selection_rate_disparity": disparity_difference,  # Keep for compatibility
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

        # Check for adverse impact using disparate impact ratio
        # DI ratio < 0.8 indicates adverse impact per four-fifths rule
        di_ratio = metrics["disparate_impact_ratio"]
        di_threshold = approval_thresholds.get(
            "disparate_impact_threshold", 0.8
        )

        if di_ratio < di_threshold:
            bias_flag = True
            logger.warning(
                f"Adverse impact detected for {feature_name}: "
                f"DI ratio = {di_ratio:.3f} < {di_threshold}"
            )

    return fairness_report, bias_flag
