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
"""This module contains detailed error analysis and metrics for OCR results."""

import difflib
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple, Union

from jiwer import cer, wer


@dataclass
class ErrorAnalysis:
    """Detailed error analysis results."""

    total_errors: int
    insertions: int
    deletions: int
    substitutions: int
    common_substitutions: Dict[str, str]  # actual -> predicted pairs
    error_positions: Dict[str, int]  # position categories -> counts
    error_distribution: Dict[str, float]  # percentages of error types


def analyze_errors(ground_truth: str, predicted: str) -> ErrorAnalysis:
    """Perform detailed error analysis between ground truth and prediction.

    Args:
        ground_truth: The reference text (ground truth)
        predicted: The OCR extracted text to analyze

    Returns:
        ErrorAnalysis object with detailed error metrics
    """
    # Clean up texts for comparison
    ground_truth = re.sub(r"\s+", " ", ground_truth).strip()
    predicted = re.sub(r"\s+", " ", predicted).strip()

    # Get character-level diff
    d = difflib.SequenceMatcher(None, ground_truth, predicted)

    # Track error types
    insertions = 0
    deletions = 0
    substitutions = 0
    substitution_pairs = []
    error_positions = Counter()

    # Process diff blocks
    for tag, i1, i2, j1, j2 in d.get_opcodes():
        if tag == "replace":
            substitutions += max(i2 - i1, j2 - j1)
            # Track character pairs for substitution analysis
            if i2 - i1 == j2 - j1:  # 1:1 substitution
                for idx in range(i2 - i1):
                    substitution_pairs.append((ground_truth[i1 + idx], predicted[j1 + idx]))
            # Track position in text (beginning, middle, end)
            if i1 < len(ground_truth) * 0.2:
                error_positions["beginning"] += 1
            elif i1 > len(ground_truth) * 0.8:
                error_positions["end"] += 1
            else:
                error_positions["middle"] += 1
        elif tag == "delete":
            deletions += i2 - i1
        elif tag == "insert":
            insertions += j2 - j1

    # Analyze common substitutions
    common_subs = Counter(substitution_pairs).most_common(10)
    common_substitutions = {}
    for (gt, pred), count in common_subs:
        common_substitutions[gt] = pred

    # Calculate total errors and distribution
    total_errors = insertions + deletions + substitutions

    # Calculate error distribution (percentages)
    dist = {}
    if total_errors > 0:
        dist = {
            "insertions": insertions / total_errors * 100,
            "deletions": deletions / total_errors * 100,
            "substitutions": substitutions / total_errors * 100,
        }

    return ErrorAnalysis(
        total_errors=total_errors,
        insertions=insertions,
        deletions=deletions,
        substitutions=substitutions,
        common_substitutions=common_substitutions,
        error_positions=dict(error_positions),
        error_distribution=dist,
    )


def compare_multi_model(
    ground_truth: str,
    model_texts: Dict[str, str],
) -> Dict[str, Dict[str, Union[float, int, Dict]]]:
    """Compares OCR results from multiple models with the ground truth.

    Args:
        ground_truth (str): The ground truth text.
        model_texts (Dict[str, str]): Dictionary mapping model display names to extracted text.

    Returns:
        Dict[str, Dict[str, Union[float, int, Dict]]]: A dictionary of model names to metrics.
    """
    # Initialize results dictionary
    results = {}

    # Calculate metrics for each model
    for model_display, text in model_texts.items():
        model_metrics = {}

        # Basic metrics
        model_metrics["CER"] = cer(ground_truth, text)
        model_metrics["WER"] = wer(ground_truth, text)

        # Detailed error analysis
        model_analysis = analyze_errors(ground_truth, text)

        # Add detailed metrics
        model_metrics.update(
            {
                "Insertions": model_analysis.insertions,
                "Deletions": model_analysis.deletions,
                "Substitutions": model_analysis.substitutions,
                "Insertion Rate": model_analysis.error_distribution.get("insertions", 0),
                "Deletion Rate": model_analysis.error_distribution.get("deletions", 0),
                "Substitution Rate": model_analysis.error_distribution.get("substitutions", 0),
                "Error Positions": model_analysis.error_positions,
                "Common Substitutions": model_analysis.common_substitutions,
            }
        )

        # Store in results
        results[model_display] = model_metrics

    return results
