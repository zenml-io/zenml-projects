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
from difflib import SequenceMatcher
from typing import Any, Dict, List, Union

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
                    substitution_pairs.append(
                        (ground_truth[i1 + idx], predicted[j1 + idx])
                    )
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


def levenshtein_ratio(s1: str, s2: str) -> float:
    """Calculate the Levenshtein ratio between two strings."""
    return SequenceMatcher(None, s1, s2).ratio()


def find_best_model(
    model_metrics: Dict[str, Dict[str, float]],
    metric: str,
    lower_is_better: bool = True,
) -> str:
    """Find the best performing model(s) for a given metric, showing ties when they occur."""
    best_models = []
    best_value = None

    for model, metrics in model_metrics.items():
        if metric in metrics:
            value = metrics[metric]
            if (
                best_value is None
                or (lower_is_better and value < best_value)
                or (not lower_is_better and value > best_value)
            ):
                best_value = value
    if best_value is not None:
        for model, metrics in model_metrics.items():
            if metric in metrics:
                value = metrics[metric]
                if (lower_is_better and abs(value - best_value) < 1e-6) or (
                    not lower_is_better and abs(value - best_value) < 1e-6
                ):
                    best_models.append(model)

    if not best_models:
        return "N/A"
    elif len(best_models) == 1:
        return best_models[0]
    else:
        # return ties as a comma-separated list
        return ", ".join(best_models)


def calculate_custom_metrics(
    ground_truth_text: str,
    model_texts: Dict[str, str],
    model_displays: List[str],
) -> Dict[str, Dict[str, float]]:
    """Calculate metrics for each model and between model pairs."""
    all_metrics = {}
    model_pairs = []
    for i, model1 in enumerate(model_displays):
        if model1 not in all_metrics:
            all_metrics[model1] = {}
        text1 = model_texts.get(model1, "")
        if ground_truth_text:
            all_metrics[model1]["CER"] = cer(ground_truth_text, text1)
            all_metrics[model1]["WER"] = wer(ground_truth_text, text1)
            all_metrics[model1]["GT Similarity"] = levenshtein_ratio(
                ground_truth_text, text1
            )
        for j, model2 in enumerate(model_displays):
            if i < j:
                model_pairs.append((model1, model2))
    for model1, model2 in model_pairs:
        text1 = model_texts.get(model1, "")
        text2 = model_texts.get(model2, "")
        similarity = levenshtein_ratio(text1, text2)
        pair_key = f"{model1}_{model2}"
        all_metrics[pair_key] = similarity
    return all_metrics


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
    results = {}

    for model_display, text in model_texts.items():
        model_metrics = {}

        model_metrics["CER"] = cer(ground_truth, text)
        model_metrics["WER"] = wer(ground_truth, text)

        model_analysis = analyze_errors(ground_truth, text)

        model_metrics.update(
            {
                "Insertions": model_analysis.insertions,
                "Deletions": model_analysis.deletions,
                "Substitutions": model_analysis.substitutions,
                "Insertion Rate": model_analysis.error_distribution.get(
                    "insertions", 0
                ),
                "Deletion Rate": model_analysis.error_distribution.get(
                    "deletions", 0
                ),
                "Substitution Rate": model_analysis.error_distribution.get(
                    "substitutions", 0
                ),
                "Error Positions": model_analysis.error_positions,
                "Common Substitutions": model_analysis.common_substitutions,
            }
        )

        results[model_display] = model_metrics

    return results


def normalize_text(s: str) -> str:
    """Normalize text for comparison."""
    s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()
    s = s.replace("\n", " ")
    # Normalize apostrophes and similar characters
    s = re.sub(r"[''′`]", "'", s)
    return s


def calculate_model_similarities(
    results: List[Dict[str, Any]], model_displays: List[str]
) -> Dict[str, float]:
    """Calculate the average pairwise Levenshtein ratio between model outputs.

    Expects each result to have keys formatted as:
        "raw_text_{model_display}"
    where model_display is converted to lowercase and spaces are replaced with underscores.

    Args:
        results (List[Dict[str, Any]]): List of dictionaries containing model outputs.
        model_displays (List[str]): List of model display names.

    Returns:
        Dict[str, float]: Dictionary mapping each model pair (formatted as "Model1_Model2")
                          to their average similarity score.
    """
    similarity_sums = {}
    similarity_counts = {}

    for result in results:
        # Map model display names to their corresponding text
        model_texts = {}
        for display in model_displays:
            key = f"raw_text_{display.lower().replace(' ', '_')}"
            text = result.get(key, "")
            if isinstance(text, str):
                text = normalize_text(text)
                if text:
                    model_texts[display] = text

        # Only proceed if at least two models have valid text
        if len(model_texts) < 2:
            continue

        # Compute pairwise similarity for each combination
        for i in range(len(model_displays)):
            for j in range(i + 1, len(model_displays)):
                model1 = model_displays[i]
                model2 = model_displays[j]
                if model1 not in model_texts or model2 not in model_texts:
                    continue
                text1 = model_texts[model1]
                text2 = model_texts[model2]
                similarity = levenshtein_ratio(text1, text2)
                pair_key = f"{model1}_{model2}"
                similarity_sums[pair_key] = (
                    similarity_sums.get(pair_key, 0) + similarity
                )
                similarity_counts[pair_key] = (
                    similarity_counts.get(pair_key, 0) + 1
                )

    # Average the similarities for each pair
    similarities = {
        pair: similarity_sums[pair] / similarity_counts[pair]
        for pair in similarity_sums
    }
    return similarities
