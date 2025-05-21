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

from typing import Annotated, Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from zenml import get_step_context, log_metadata, step
from zenml.client import Client
from zenml.logger import get_logger
from zenml.types import HTMLString

from src.constants import (
    EVAL_VISUALIZATION_NAME,
    EVALUATION_RESULTS_NAME,
    MODEL_NAME,
    TEST_DATASET_NAME,
)
from src.utils import (
    analyze_fairness,
    generate_eval_visualization,
    report_bias_incident,
)

logger = get_logger(__name__)


@step()
def evaluate_model(
    protected_attributes: List[str],
    test_df: Annotated[pd.DataFrame, TEST_DATASET_NAME],
    target: str = "TARGET",
    model: Annotated[Any, MODEL_NAME] = None,
    approval_thresholds: Dict[str, float] = None,
    cost_matrix: Optional[Dict[str, float]] = None,
) -> Tuple[
    Annotated[Dict[str, Any], EVALUATION_RESULTS_NAME],
    Annotated[HTMLString, EVAL_VISUALIZATION_NAME],
]:
    """Compute performance + fairness metrics. Articles 9 & 15 compliant.

    Args:
        protected_attributes: List of protected attributes
        test_df: Test dataset
        target: Target column name
        model: The trained model
        approval_thresholds: Approval thresholds for fairness metrics
        cost_matrix: Optional cost matrix for financial impact evaluation
                    e.g. {"fp_cost": 1, "fn_cost": 5} means false negatives
                    are 5x more costly than false positives

    Returns:
        Dictionary containing evaluation and fairness results
    """
    # Get the run ID for artifacts
    run_id = str(get_step_context().pipeline_run.id)

    # Default cost matrix for credit scoring (false negatives more costly)
    if not cost_matrix:
        cost_matrix = {"fp_cost": 1, "fn_cost": 10}

    # Get model and identify target column
    if model is None:
        client = Client()
        model = client.get_artifact_version(name_id_or_prefix=MODEL_NAME)

    target_col = next(
        col for col in test_df.columns if col.endswith(f"__{target}") or col == target
    )

    # Prepare test data
    X_test, y_test = test_df.drop(columns=[target_col]), test_df[target_col]

    # Log class distribution info
    class_counts = y_test.value_counts()
    logger.info(f"Class distribution in test set: {class_counts}")

    # Get predictions and raw probabilities
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Log prediction distribution
    pred_counts = pd.Series(y_pred).value_counts()
    logger.info(f"Predictions distribution: {pred_counts}")

    # Import all necessary metrics

    # Calculate confusion matrix for default threshold (0.5)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    logger.info(f"Confusion Matrix:\n{cm}")

    # ===== 1. Standard metrics at default threshold =====
    performance_metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "auc_roc": roc_auc_score(y_test, y_prob),
        # Add specific components for easier interpretation
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }

    # ===== 2. Add metrics suited for imbalanced data =====
    # Average precision summarizes precision-recall curve
    performance_metrics["average_precision"] = average_precision_score(y_test, y_prob)

    # Balanced accuracy is mean of sensitivity and specificity
    performance_metrics["balanced_accuracy"] = balanced_accuracy_score(y_test, y_pred)

    # ===== 3. Financial impact metric =====
    # Calculate expected cost using the cost matrix
    total_cost = (fp * cost_matrix["fp_cost"]) + (fn * cost_matrix["fn_cost"])
    # Normalize by dataset size for comparison across datasets
    performance_metrics["normalized_cost"] = total_cost / len(y_test)

    # ===== 4. Threshold optimization =====
    # Try different thresholds to find optimal F1 score
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    threshold_metrics = {}

    # Find optimal threshold for F1
    best_f1 = 0
    best_threshold = 0.5  # Default

    for threshold in thresholds:
        y_pred_at_threshold = (y_prob >= threshold).astype(int)
        precision = precision_score(y_test, y_pred_at_threshold)
        recall = recall_score(y_test, y_pred_at_threshold)
        f1 = f1_score(y_test, y_pred_at_threshold)

        # Calculate confusion matrix at this threshold
        cm_t = confusion_matrix(y_test, y_pred_at_threshold)
        tn_t, fp_t, fn_t, tp_t = cm_t.ravel()

        # Calculate cost at this threshold
        threshold_cost = (fp_t * cost_matrix["fp_cost"]) + (fn_t * cost_matrix["fn_cost"])
        normalized_threshold_cost = threshold_cost / len(y_test)

        threshold_metrics[threshold] = {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "true_positives": int(tp_t),
            "false_positives": int(fp_t),
            "false_negatives": int(fn_t),
            "true_negatives": int(tn_t),
            "normalized_cost": normalized_threshold_cost,
        }

        logger.info(
            f"Threshold {threshold}: precision={precision:.4f}, recall={recall:.4f}, "
            f"f1={f1:.4f}, cost={normalized_threshold_cost:.4f}"
        )

        # Track best F1 score
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    # Find optimal threshold for minimizing cost
    cost_values = [metrics["normalized_cost"] for metrics in threshold_metrics.values()]
    min_cost_idx = np.argmin(cost_values)
    min_cost_threshold = list(threshold_metrics.keys())[min_cost_idx]

    # ===== 5. Add optimal threshold information =====
    performance_metrics["threshold_metrics"] = threshold_metrics
    performance_metrics["optimal_f1_threshold"] = best_threshold
    performance_metrics["optimal_f1_score"] = best_f1
    performance_metrics["optimal_cost_threshold"] = min_cost_threshold
    performance_metrics["optimal_cost"] = threshold_metrics[min_cost_threshold]["normalized_cost"]

    # ===== 6. Calculate final metrics using cost-optimal threshold =====
    optimal_y_pred = (y_prob >= min_cost_threshold).astype(int)
    performance_metrics["optimal_precision"] = precision_score(y_test, optimal_y_pred)
    performance_metrics["optimal_recall"] = recall_score(y_test, optimal_y_pred)
    performance_metrics["optimal_f1"] = f1_score(y_test, optimal_y_pred)
    performance_metrics["optimal_accuracy"] = accuracy_score(y_test, optimal_y_pred)

    logger.info(f"Performance metrics at default threshold (0.5): {performance_metrics}")
    logger.info(f"Optimal threshold for F1: {best_threshold}, F1: {best_f1:.4f}")
    logger.info(
        f"Optimal threshold for cost: {min_cost_threshold}, "
        f"Cost: {threshold_metrics[min_cost_threshold]['normalized_cost']:.4f}"
    )

    # ===== 7. Fairness analysis with updated metrics =====
    fairness_metrics, bias_flag = analyze_fairness(
        y_test,
        optimal_y_pred,  # Use predictions from optimal threshold
        protected_attributes,
        test_df,
        approval_thresholds,
    )

    # Create fairness report
    fairness_report = {
        "metrics": performance_metrics,
        "fairness_metrics": fairness_metrics,
        "bias_flag": bias_flag,
        "protected_attributes_checked": protected_attributes,
        "cost_matrix": cost_matrix,
    }

    # ===== 8. Generate visualizations for better understanding =====
    # Precision-Recall curve
    eval_visualization = generate_eval_visualization(
        performance_metrics=performance_metrics,
        threshold_metrics=threshold_metrics,
        min_cost_threshold=min_cost_threshold,
        y_test=y_test,
        y_prob=y_prob,
    )

    # Log metadata for the pipeline
    log_metadata(
        {
            "metrics": {
                # Include only non-complex metrics for dashboard display
                "accuracy": performance_metrics["accuracy"],
                "precision": performance_metrics["precision"],
                "recall": performance_metrics["recall"],
                "f1_score": performance_metrics["f1_score"],
                "auc_roc": performance_metrics["auc_roc"],
                "average_precision": performance_metrics["average_precision"],
                "optimal_threshold": min_cost_threshold,
                "optimal_precision": performance_metrics["optimal_precision"],
                "optimal_recall": performance_metrics["optimal_recall"],
                "optimal_f1": performance_metrics["optimal_f1"],
                "optimal_cost": performance_metrics["optimal_cost"],
            },
            "bias_flag": bias_flag,
        }
    )

    # Create incident report if bias detected
    if bias_flag:
        report_bias_incident(fairness_report, run_id)

    eval_results = {
        "metrics": performance_metrics,
        "fairness": fairness_report,
    }
    # Return the evaluation results
    return eval_results, eval_visualization
