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

"""
Step for saving classification results with automatic metrics calculation.
"""

import datetime
import json
import os
from collections import Counter
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from utils import logger


def save_json_results(json_dict: Dict, output_path: str) -> None:
    """
    Save results to JSON file.

    Args:
        json_dict: Dictionary to save
        output_path: Path to save JSON file
    """
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(json_dict, f, indent=2, ensure_ascii=False)
    logger.log_output_file(output_path)


def calculate_metrics(
    base_dataset_path: str,
    classifications: List[Dict],
) -> Tuple[Dict[str, Any], List[Dict]]:
    """
    Calculate agreement metrics between original classifications and predictions.
    Excludes articles where the reason field starts with "Error".

    Args:
        base_dataset_path: Path to the original JSONL file with ground truth labels
        classifications: List of classification results

    Returns:
        Tuple of (metrics_dict, error_entries)
    """
    original_labels = {}
    with open(base_dataset_path, "r") as f:
        for line in f:
            article = json.loads(line)
            url = article["meta"]["url"]
            # Convert 'accept'/'reject' to boolean
            is_accepted = article["answer"].lower() == "accept"
            original_labels[url] = is_accepted

    error_entries = []
    error_counts = Counter()

    y_true = []
    y_pred = []
    confidence_values = []

    for i, classification in enumerate(classifications):
        reason = classification.get("reason", "")
        if reason.startswith("Error"):
            error_entries.append(
                {
                    "index": i,
                    "error": reason,
                    "url": classification.get("meta", {}).get(
                        "url", "unknown"
                    ),
                }
            )
            error_counts[reason] += 1
            continue

        url = classification.get("meta", {}).get("url", "")
        if url in original_labels:
            y_true.append(original_labels[url])
            y_pred.append(classification.get("is_accepted", False))
            confidence_values.append(classification.get("confidence", 0.0))

    # skip metrics calculation if there's no valid data
    total_errors = len(error_entries)
    if not y_true or not y_pred:
        logger.log_warning("No valid data available for metrics calculation")
        return {
            "error_count": total_errors,
            "error_types": dict(error_counts),
        }, error_entries

    # main metrics
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = conf_matrix.ravel()

    # other metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0

    # Class balance analysis
    total_samples = len(y_true)
    valid_samples_percentage = (
        total_samples / (total_samples + total_errors) * 100
        if (total_samples + total_errors) > 0
        else 0
    )

    class_distribution = {
        "positive_ratio": sum(y_true) / total_samples,
        "negative_ratio": (total_samples - sum(y_true)) / total_samples,
    }

    confidence_by_type = {"TP": [], "TN": [], "FP": [], "FN": []}

    for i in range(len(y_true)):
        confidence = float(confidence_values[i])
        if y_true[i] == y_pred[i]:
            pred_type = "TP" if y_true[i] else "TN"
        else:
            pred_type = "FP" if y_pred[i] else "FN"
        confidence_by_type[pred_type].append(confidence)

    confidence_stats = {
        key: {
            "mean": np.mean(values) if values else 0,
            "std": np.std(values) if values else 0,
            "count": len(values),
        }
        for key, values in confidence_by_type.items()
    }

    log_metrics_to_console(
        total_samples,
        total_errors,
        valid_samples_percentage,
        accuracy,
        f1,
        precision,
        recall,
        specificity,
        npv,
        class_distribution,
        confidence_stats,
        error_counts,
    )

    return {
        "summary": {
            "valid_samples": total_samples,
            "valid_samples_percentage": valid_samples_percentage,
            "error_samples": total_errors,
            "error_samples_percentage": 100 - valid_samples_percentage,
        },
        "basic_metrics": {
            "accuracy": accuracy,
            "f1_score": f1,
            "precision": precision,
            "recall": recall,
        },
        "advanced_metrics": {"specificity": specificity, "npv": npv},
        "confusion_matrix": conf_matrix.tolist(),
        "class_distribution": class_distribution,
        "confidence_stats": confidence_stats,
        "error_count": total_errors,
        "error_types": dict(error_counts),
    }, error_entries


def log_metrics_to_console(
    total_samples,
    total_errors,
    valid_samples_percentage,
    accuracy,
    f1,
    precision,
    recall,
    specificity,
    npv,
    class_distribution,
    confidence_stats,
    error_counts,
):
    """Log metrics to console using the logger."""
    # error statistics
    logger.log_process(f"\nError Analysis ({total_errors} errors detected):")
    for error_msg, count in error_counts.most_common():
        logger.log_warning(f"{count} occurrences: {error_msg}")

    # summary
    logger.log_process("\nAnalysis Summary:")
    logger.log_success(
        f"Valid samples: {total_samples} ({valid_samples_percentage:.1f}%)"
    )
    logger.log_success(
        f"Error samples: {total_errors} ({100 - valid_samples_percentage:.1f}%)"
    )

    # metrics
    logger.log_process("\nDetailed Performance Metrics:")
    logger.log_success(f"Accuracy: {accuracy:.3f}")
    logger.log_success(f"F1 Score: {f1:.3f}")
    logger.log_success(f"Precision: {precision:.3f}")
    logger.log_success(f"Recall (Sensitivity): {recall:.3f}")
    logger.log_success(f"Specificity: {specificity:.3f}")
    logger.log_success(f"Negative Predictive Value: {npv:.3f}")

    # class distribution
    logger.log_process("\nClass Distribution:")
    logger.log_success(
        f"Positive class ratio: {class_distribution['positive_ratio']:.3f}"
    )
    logger.log_success(
        f"Negative class ratio: {class_distribution['negative_ratio']:.3f}"
    )

    # confidence analysis
    logger.log_process("\nConfidence Analysis by Prediction Type:")
    for pred_type, stats in confidence_stats.items():
        logger.log_success(
            f"{pred_type}: mean={stats['mean']:.3f}, std={stats['std']:.3f}, count={stats['count']}"
        )


def create_metrics_report(metrics: Dict, output_path: str):
    """
    Create a markdown report with the metrics results.

    Args:
        metrics: Dictionary containing all metrics
        output_path: Path to save the markdown report
    """
    summary = metrics["summary"]
    basic = metrics["basic_metrics"]
    advanced = metrics["advanced_metrics"]
    class_dist = metrics["class_distribution"]
    conf_stats = metrics["confidence_stats"]
    error_types = metrics["error_types"]

    # Create markdown content
    markdown = f"""# Classification Metrics Report
Generated: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Summary
- Valid samples: {summary["valid_samples"]} ({summary["valid_samples_percentage"]:.1f}%)
- Error samples: {summary["error_samples"]} ({summary["error_samples_percentage"]:.1f}%)

## Error Analysis
Total errors: {metrics["error_count"]}

| Error Type | Count |
|------------|-------|
"""

    # error types table
    for error, count in error_types.items():
        markdown += f"| {error} | {count} |\n"

    # performance metrics
    markdown += f"""
## Performance Metrics
| Metric | Value |
|--------|-------|
| Accuracy | {basic["accuracy"]:.3f} |
| F1 Score | {basic["f1_score"]:.3f} |
| Precision | {basic["precision"]:.3f} |
| Recall (Sensitivity) | {basic["recall"]:.3f} |
| Specificity | {advanced["specificity"]:.3f} |
| Negative Predictive Value | {advanced["npv"]:.3f} |

## Class Distribution
- Positive class ratio: {class_dist["positive_ratio"]:.3f}
- Negative class ratio: {class_dist["negative_ratio"]:.3f}

## Confidence Analysis by Prediction Type
| Type | Mean | Std | Count |
|------|------|-----|-------|
"""

    # confidence stats table
    for pred_type, stats in conf_stats.items():
        markdown += f"| {pred_type} | {stats['mean']:.3f} | {stats['std']:.3f} | {stats['count']} |\n"

    with open(output_path, "w") as f:
        f.write(markdown)


def calculate_and_save_metrics_from_json(
    results_path: str = "classification_results/evaluation/results.json",
    base_dataset_path: str = "data/composite_dataset.jsonl",
):
    """
    Calculate metrics from a results JSON file and save them in JSON and Markdown formats.

    Args:
        results_path: Path to the saved results JSON file
        base_dataset_path: Path to ground truth data
    """
    if not os.path.exists(base_dataset_path):
        logger.log_warning(
            f"Ground truth file not found at {base_dataset_path}. Metrics not calculated."
        )
        return

    if not os.path.exists(results_path):
        logger.log_warning(
            f"Results file not found at {results_path}. Metrics not calculated."
        )
        return

    try:
        with open(results_path, "r") as f:
            results_data = json.load(f)

        classifications = []
        results = results_data.get("results", {})

        indices = results.get("is_accepted", {}).keys()

        for idx in indices:
            classification = {
                "is_accepted": results.get("is_accepted", {}).get(idx, False),
                "confidence": results.get("confidence", {}).get(idx, 0.0),
                "reason": results.get("reason", {}).get(idx, ""),
                "meta": {"url": results.get("meta_url", {}).get(idx, "")},
            }
            classifications.append(classification)

    except Exception as e:
        logger.log_warning(f"Failed to load or parse results file: {str(e)}")
        return

    output_dir = os.path.dirname(results_path)

    metrics_json_path = os.path.join(output_dir, "metrics.json")
    metrics_report_path = os.path.join(output_dir, "report.md")

    metrics, error_entries = calculate_metrics(
        base_dataset_path=base_dataset_path,
        classifications=classifications,
    )

    # Add metrics to the original results file
    try:
        results_data["metrics"] = metrics
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.log_warning(
            f"Failed to update results file with metrics: {str(e)}"
        )

    # Save metrics as separate JSON
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    logger.log_output_file(metrics_json_path)

    # Create and save markdown report
    create_metrics_report(metrics, metrics_report_path)
    logger.log_output_file(metrics_report_path)

    return output_dir
