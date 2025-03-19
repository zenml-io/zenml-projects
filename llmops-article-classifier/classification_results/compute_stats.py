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

# classification_results/compute_stats.py
"""
This script computes statistics for classification results by comparing DeepSeek's
classifications against original labels from the composite dataset.

It aggregates per-article statistics across multiple runs (e.g. count of runs, counts
of accepted vs. rejected classifications, and summed confidences) and computes overall
run-level statistics (totals and averages). These metrics were used to refine the prompt
and improve model performance.
"""

import glob
import json
from typing import Any, Dict, List


def convert_column_oriented_to_list(data: dict) -> List[dict]:
    """
    Convert a column-oriented JSON format (each key maps to a dict of row index → value)
    into a list of row dictionaries.

    Args:
        data (dict): Column-oriented data.

    Returns:
        List[dict]: A list where each element is a dictionary representing a row.

    Raises:
        ValueError: If any column's value is not a dict.
    """
    columns = list(data.keys())
    if not all(isinstance(data[col], dict) for col in columns):
        raise ValueError("Data is not in a column-oriented format.")

    # Assume all columns share the same sorted row indices.
    row_indices = sorted(next(iter(data.values())).keys(), key=lambda x: int(x))
    rows = []
    for idx in row_indices:
        row = {col: data[col].get(idx) for col in columns}
        rows.append(row)
    return rows


def load_json_file(filename: str) -> List[Dict[str, Any]]:
    """
    Load JSON data from a file and return a list of dictionaries.

    The file can be in one of three formats:
      - A dict with a "results" key in column-oriented format.
      - A list of dictionaries.
      - A dict in column-oriented format.

    Args:
        filename (str): The path to the JSON file.

    Returns:
        List[Dict[str, Any]]: Parsed JSON data as a list of dictionaries.
    """
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "results" in data:
        return convert_column_oriented_to_list(data["results"])
    elif isinstance(data, list):
        return data
    else:
        return convert_column_oriented_to_list(data)


def aggregate_statistics(json_files: List[str]) -> Dict[int, Dict[str, Any]]:
    """
    Aggregate per-article statistics across multiple runs.

    For each article (by index), aggregates:
      - Number of runs (`num_runs`)
      - Count of True and False values for "is_accepted"
      - Total of confidence scores

    Args:
        json_files (List[str]): List of JSON file paths.

    Returns:
        Dict[int, Dict[str, Any]]: Mapping from article index to aggregated statistics.
    """
    stats: Dict[int, Dict[str, Any]] = {}
    for filename in json_files:
        data = load_json_file(filename)
        for i, record in enumerate(data):
            if i not in stats:
                stats[i] = {
                    "true_count": 0,
                    "false_count": 0,
                    "total_confidence": 0.0,
                    "num_runs": 0,
                }
            accepted = record.get("is_accepted")
            confidence = record.get("confidence", 0.0)
            if accepted:
                stats[i]["true_count"] += 1
            else:
                stats[i]["false_count"] += 1
            stats[i]["total_confidence"] += confidence
            stats[i]["num_runs"] += 1
    return stats


def compute_similarity(stats: Dict[int, Dict[str, Any]]) -> float:
    """
    Compute the average consistency across articles.

    For each article, the consistency is defined as the fraction of runs that agree with
    the majority classification. The function returns the average consistency over all articles.

    Args:
        stats (Dict[int, Dict[str, Any]]): Per-article aggregated statistics.

    Returns:
        float: The average consistency (0.0 to 1.0).
    """
    total_similarity = 0.0
    count = 0
    for rec in stats.values():
        if rec["num_runs"] > 0:
            majority = max(rec["true_count"], rec["false_count"])
            total_similarity += majority / rec["num_runs"]
            count += 1
    return total_similarity / count if count > 0 else 0.0


def aggregate_run_level_statistics(json_files: List[str]) -> Dict[str, Any]:
    """
    Aggregate overall run-level statistics from all JSON files.

    Computes:
      - Total number of runs.
      - Total articles processed (sum over runs).
      - Total counts for True and False classifications.
      - Average confidence across all articles.

    Args:
        json_files (List[str]): List of JSON file paths.

    Returns:
        Dict[str, Any]: A dictionary containing aggregated run-level statistics.
    """
    total_true = 0
    total_false = 0
    total_confidence = 0.0
    total_articles = 0
    num_runs = len(json_files)

    for filename in json_files:
        data = load_json_file(filename)
        run_true = sum(1 for record in data if record.get("is_accepted"))
        run_false = sum(1 for record in data if not record.get("is_accepted"))
        run_conf = sum(record.get("confidence", 0.0) for record in data)
        total_true += run_true
        total_false += run_false
        total_confidence += run_conf
        total_articles += len(data)

    avg_conf = total_confidence / total_articles if total_articles > 0 else 0.0
    return {
        "num_runs": num_runs,
        "total_articles": total_articles,
        "total_true": total_true,
        "total_false": total_false,
        "avg_confidence": avg_conf,
    }


def print_statistics(stats: Dict[int, Dict[str, Any]]) -> None:
    """
    Print aggregated per-article statistics in a tabular format.

    Args:
        stats (Dict[int, Dict[str, Any]]): Aggregated statistics for each article.
    """
    print("Article Statistics Across Pipeline Runs:")
    header = (
        f"{'Article Index':>13} | {'Num Runs':>8} | "
        f"{'True Count':>10} | {'False Count':>11} | {'Avg Confidence':>14}"
    )
    print(header)
    print("-" * len(header))
    for idx in sorted(stats.keys()):
        rec = stats[idx]
        num_runs = rec["num_runs"]
        avg_conf = rec["total_confidence"] / num_runs if num_runs else 0.0
        print(
            f"{idx:13} | {num_runs:8} | {rec['true_count']:10} | "
            f"{rec['false_count']:11} | {avg_conf:14.4f}"
        )


def print_run_summary(run_summary: Dict[str, Any], avg_consistency: float) -> None:
    """
    Print overall run-level summary statistics.

    Args:
        run_summary (Dict[str, Any]): Dictionary of aggregated run-level statistics.
        avg_consistency (float): Average consistency computed from per-article statistics.
    """
    print("\nOverall Run Summary:")
    header = f"{'Metric':>25} | {'Value':>10}"
    print(header)
    print("-" * len(header))
    print(f"{'Number of Runs':>25} | {run_summary['num_runs']:10d}")
    print(f"{'Total Articles':>25} | {run_summary['total_articles']:10d}")
    print(f"{'Total True Articles':>25} | {run_summary['total_true']:10d}")
    print(f"{'Total False Articles':>25} | {run_summary['total_false']:10d}")
    print(f"{'Avg Confidence':>25} | {run_summary['avg_confidence']:10.4f}")
    print(f"{'Avg Consistency':>25} | {avg_consistency:10.2f}")


def main() -> None:
    """
    Main function to aggregate and print classification statistics from JSON files.

    It searches for files matching the pattern 'results_*.json', aggregates per-article and
    run-level statistics, computes average consistency, and prints the results.
    """
    json_files = glob.glob("composite_results_*.json")
    if not json_files:
        print("No JSON files found matching pattern 'results_*.json'")
        return

    print(f"Found {len(json_files)} JSON file(s): {json_files}\n")
    stats = aggregate_statistics(json_files)
    print_statistics(stats)

    run_summary = aggregate_run_level_statistics(json_files)
    avg_consistency = compute_similarity(stats)
    print_run_summary(run_summary, avg_consistency)


if __name__ == "__main__":
    main()
