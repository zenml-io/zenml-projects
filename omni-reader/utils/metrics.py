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
from typing import Dict, List, Tuple

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


def compare_results(ground_truth: str, gemma_text: str, mistral_text: str) -> dict:
    """Compares Gemma 3 and Mistral OCR results with the ground truth.

    Args:
        ground_truth (str): The ground truth text.
        gemma_text (str): The text extracted by Gemma 3.
        mistral_text (str): The text extracted by Mistral OCR.

    Returns:
        dict: A dictionary containing metrics and error analysis for each model.
    """
    # Basic metrics
    metrics = {
        "Gemma CER": cer(ground_truth, gemma_text),
        "Gemma WER": wer(ground_truth, gemma_text),
        "Mistral CER": cer(ground_truth, mistral_text),
        "Mistral WER": wer(ground_truth, mistral_text),
    }

    # Detailed error analysis
    gemma_analysis = analyze_errors(ground_truth, gemma_text)
    mistral_analysis = analyze_errors(ground_truth, mistral_text)

    # Add detailed metrics to the results
    metrics.update(
        {
            "Gemma Insertions": gemma_analysis.insertions,
            "Gemma Deletions": gemma_analysis.deletions,
            "Gemma Substitutions": gemma_analysis.substitutions,
            "Gemma Insertion Rate": gemma_analysis.error_distribution.get("insertions", 0),
            "Gemma Deletion Rate": gemma_analysis.error_distribution.get("deletions", 0),
            "Gemma Substitution Rate": gemma_analysis.error_distribution.get("substitutions", 0),
            "Gemma Error Positions": gemma_analysis.error_positions,
            "Gemma Common Substitutions": gemma_analysis.common_substitutions,
            "Mistral Insertions": mistral_analysis.insertions,
            "Mistral Deletions": mistral_analysis.deletions,
            "Mistral Substitutions": mistral_analysis.substitutions,
            "Mistral Insertion Rate": mistral_analysis.error_distribution.get("insertions", 0),
            "Mistral Deletion Rate": mistral_analysis.error_distribution.get("deletions", 0),
            "Mistral Substitution Rate": mistral_analysis.error_distribution.get("substitutions", 0),
            "Mistral Error Positions": mistral_analysis.error_positions,
            "Mistral Common Substitutions": mistral_analysis.common_substitutions,
        }
    )

    return metrics
