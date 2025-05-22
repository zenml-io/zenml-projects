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

from src.constants import Artifacts as A
from src.utils import (
    analyze_fairness,
    generate_eval_visualization,
    report_bias_incident,
)

logger = get_logger(__name__)


@step()
def evaluate_model(
    protected_attributes: List[str],
    test_df: Annotated[pd.DataFrame, A.TEST_DATASET],
    optimal_threshold: Annotated[float, A.OPTIMAL_THRESHOLD],
    target: str = "TARGET",
    model: Annotated[Any, A.MODEL] = None,
    approval_thresholds: Dict[str, float] = None,
    cost_matrix: Optional[Dict[str, float]] = None,
) -> Tuple[
    Annotated[Dict[str, Any], A.EVALUATION_RESULTS],
    Annotated[HTMLString, A.EVAL_VISUALIZATION],
]:
    """Compute performance + fairness metrics. Articles 9 & 15 compliant.

    Args:
        protected_attributes: List of protected attributes
        test_df: Test dataset
        target: Target column name
        model: The trained model
        optimal_threshold: The optimal threshold for the model
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
        model = client.get_artifact_version(name_id_or_prefix=A.MODEL)

    target_col = next(
        col
        for col in test_df.columns
        if col.endswith(f"__{target}") or col == target
    )

    # Prepare test data
    X_test, y_test = test_df.drop(columns=[target_col]), test_df[target_col]

    # Log class distribution info
    class_counts = y_test.value_counts()
    logger.info(f"Class distribution in test set: {class_counts}")

    # Get raw probabilities first
    y_prob = model.predict_proba(X_test)[:, 1]

    # Use the optimal threshold from training
    logger.info(f"Using optimal threshold from training: {optimal_threshold}")
    y_pred = (y_prob >= optimal_threshold).astype(int)

    # Log prediction distribution
    pred_counts = pd.Series(y_pred).value_counts()
    logger.info(f"Predictions distribution: {pred_counts}")

    # Calculate confusion matrix at optimal threshold
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    logger.info(f"Confusion Matrix:\n{cm}")

    # ===== 1. Standard metrics at optimal threshold =====
    performance_metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "auc_roc": roc_auc_score(y_test, y_prob),
        # Add specific components for easier interpretation
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }

    # ===== 2. Add metrics suited for imbalanced data =====
    # Average precision summarizes precision-recall curve
    performance_metrics["average_precision"] = average_precision_score(
        y_test, y_prob
    )

    # Balanced accuracy is mean of sensitivity and specificity
    performance_metrics["balanced_accuracy"] = balanced_accuracy_score(
        y_test, y_pred
    )

    # ===== 3. Financial impact metric =====
    # Calculate expected cost using the cost matrix
    total_cost = (fp * cost_matrix["fp_cost"]) + (fn * cost_matrix["fn_cost"])
    # Normalize by dataset size for comparison across datasets
    performance_metrics["normalized_cost"] = total_cost / len(y_test)

    # ===== 4. Use the training-optimized threshold =====
    # Since we already have the optimal threshold from training, use it directly
    performance_metrics["optimal_threshold"] = optimal_threshold
    performance_metrics["optimal_f1_threshold"] = optimal_threshold
    performance_metrics["optimal_f1_score"] = performance_metrics["f1_score"]
    performance_metrics["optimal_cost_threshold"] = optimal_threshold
    performance_metrics["optimal_cost"] = performance_metrics[
        "normalized_cost"
    ]

    # The metrics calculated above are already at the optimal threshold
    performance_metrics["optimal_precision"] = performance_metrics["precision"]
    performance_metrics["optimal_recall"] = performance_metrics["recall"]
    performance_metrics["optimal_f1"] = performance_metrics["f1_score"]
    performance_metrics["optimal_accuracy"] = performance_metrics["accuracy"]

    # ===== 5. Optional: Test a few comparison thresholds for context =====
    # Generate some comparison metrics at different thresholds for visualization
    # Include the optimal threshold so it appears in the visualization
    comparison_thresholds = [
        0.05,
        0.1,
        optimal_threshold,
        0.15,
        0.2,
        0.25,
        0.3,
    ]
    # Remove duplicates and sort
    comparison_thresholds = sorted(list(set(comparison_thresholds)))
    threshold_metrics = {}

    for threshold in comparison_thresholds:
        y_pred_at_threshold = (y_prob >= threshold).astype(int)
        precision = precision_score(
            y_test, y_pred_at_threshold, zero_division=0
        )
        recall = recall_score(y_test, y_pred_at_threshold, zero_division=0)
        f1 = f1_score(y_test, y_pred_at_threshold, zero_division=0)

        # Calculate confusion matrix at this threshold
        cm_t = confusion_matrix(y_test, y_pred_at_threshold)
        tn_t, fp_t, fn_t, tp_t = cm_t.ravel()

        # Calculate cost at this threshold
        threshold_cost = (fp_t * cost_matrix["fp_cost"]) + (
            fn_t * cost_matrix["fn_cost"]
        )
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

    # Add threshold comparison metrics for visualization
    performance_metrics["threshold_metrics"] = threshold_metrics

    # Log performance summary
    logger.info(
        f"Performance metrics at optimal threshold ({optimal_threshold}):"
    )
    logger.info(f"  Accuracy: {performance_metrics['accuracy']:.4f}")
    logger.info(f"  Precision: {performance_metrics['precision']:.4f}")
    logger.info(f"  Recall: {performance_metrics['recall']:.4f}")
    logger.info(f"  F1 Score: {performance_metrics['f1_score']:.4f}")
    logger.info(f"  AUC-ROC: {performance_metrics['auc_roc']:.4f}")
    logger.info(
        f"  Normalized Cost: {performance_metrics['normalized_cost']:.4f}"
    )

    # ===== 6. Fairness analysis =====
    fairness_metrics, bias_flag = analyze_fairness(
        y_test,
        y_pred,  # Use predictions from optimal threshold
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

    # ===== 7. Generate visualizations =====
    eval_visualization = generate_eval_visualization(
        performance_metrics=performance_metrics,
        threshold_metrics=threshold_metrics,
        min_cost_threshold=optimal_threshold,  # Use our optimal threshold
        y_test=y_test,
        y_prob=y_prob,
    )

    # ===== 8. Log metadata for the pipeline =====
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
                "balanced_accuracy": performance_metrics["balanced_accuracy"],
                "optimal_threshold": optimal_threshold,
                "optimal_precision": performance_metrics["optimal_precision"],
                "optimal_recall": performance_metrics["optimal_recall"],
                "optimal_f1": performance_metrics["optimal_f1"],
                "optimal_cost": performance_metrics["optimal_cost"],
                "normalized_cost": performance_metrics["normalized_cost"],
                # Confusion matrix components
                "true_positives": performance_metrics["true_positives"],
                "false_positives": performance_metrics["false_positives"],
                "true_negatives": performance_metrics["true_negatives"],
                "false_negatives": performance_metrics["false_negatives"],
            },
            "bias_flag": bias_flag,
            "fairness_summary": {
                "protected_attributes": protected_attributes,
                "bias_detected": bias_flag,
            },
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
