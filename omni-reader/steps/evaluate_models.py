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
"""This module contains the steps for evaluating the OCR models."""

from typing import Any, Dict, List, Optional, Tuple

import polars as pl
import textdistance
from typing_extensions import Annotated
from zenml import get_step_context, log_metadata, step
from zenml.types import HTMLString

from utils import compare_results


def create_metrics_table(metrics: Dict[str, float]) -> str:
    """Create an HTML table for metrics.

    Args:
        metrics: Dictionary of metrics to display

    Returns:
        str: HTML table string
    """
    rows = ""
    for key, value in metrics.items():
        rows += f"""
        <tr>
            <td class="p-2 border">{key}</td>
            <td class="p-2 border text-right">{value:.4f}</td>
        </tr>
        """

    table = f"""
    <table class="min-w-full bg-white border border-gray-300 shadow-sm">
        <thead>
            <tr class="bg-gray-100">
                <th class="p-2 border text-left">Metric</th>
                <th class="p-2 border text-right">Value</th>
            </tr>
        </thead>
        <tbody>
            {rows}
        </tbody>
    </table>
    """
    return table


def create_model_comparison_card(
    image_name: str, ground_truth: str, gemma_text: str, mistral_text: str, metrics: Dict[str, Any]
) -> str:
    """Create a card for comparing OCR results for a specific image.

    Args:
        image_name: Name of the analyzed image
        ground_truth: Ground truth text
        gemma_text: Text extracted by Gemma
        mistral_text: Text extracted by Mistral
        metrics: Metrics for this comparison

    Returns:
        str: HTML card string
    """
    # Basic metrics for display
    basic_metrics = {
        "Gemma CER": metrics["Gemma CER"],
        "Gemma WER": metrics["Gemma WER"],
        "Mistral CER": metrics["Mistral CER"],
        "Mistral WER": metrics["Mistral WER"],
        "Models Similarity": metrics["models_similarity"],
        "Gemma-GT Similarity": metrics["gemma_gt_similarity"],
        "Mistral-GT Similarity": metrics["mistral_gt_similarity"],
    }

    metrics_table = create_metrics_table(basic_metrics)

    card = f"""
    <div class="bg-white rounded-lg shadow-md p-6 mb-6 border border-gray-200">
        <h3 class="text-xl font-bold mb-4 text-blue-700">{image_name}</h3>
        
        <div class="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
            <div class="border rounded-lg p-4 bg-gray-50">
                <h4 class="font-bold mb-2 text-gray-700">Ground Truth</h4>
                <div class="whitespace-pre-wrap text-sm">{ground_truth}</div>
            </div>
            
            <div class="border rounded-lg p-4 bg-gray-50">
                <h4 class="font-bold mb-2 text-blue-600">Gemma Output</h4>
                <div class="whitespace-pre-wrap text-sm">{gemma_text}</div>
            </div>
            
            <div class="border rounded-lg p-4 bg-gray-50">
                <h4 class="font-bold mb-2 text-purple-600">Mistral Output</h4>
                <div class="whitespace-pre-wrap text-sm">{mistral_text}</div>
            </div>
        </div>

        <div class="mb-4">
            <h4 class="font-bold mb-2">Key Metrics</h4>
            {metrics_table}
        </div>

        <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
                <h4 class="font-bold mb-2">Gemma Errors</h4>
                <ul class="list-disc pl-5">
                    <li>Insertions: {metrics.get("Gemma Insertions", 0)} ({metrics.get("Gemma Insertion Rate", 0):.1f}%)</li>
                    <li>Deletions: {metrics.get("Gemma Deletions", 0)} ({metrics.get("Gemma Deletion Rate", 0):.1f}%)</li>
                    <li>Substitutions: {metrics.get("Gemma Substitutions", 0)} ({metrics.get("Gemma Substitution Rate", 0):.1f}%)</li>
                </ul>
            </div>

            <div>
                <h4 class="font-bold mb-2">Mistral Errors</h4>
                <ul class="list-disc pl-5">
                    <li>Insertions: {metrics.get("Mistral Insertions", 0)} ({metrics.get("Mistral Insertion Rate", 0):.1f}%)</li>
                    <li>Deletions: {metrics.get("Mistral Deletions", 0)} ({metrics.get("Mistral Deletion Rate", 0):.1f}%)</li>
                    <li>Substitutions: {metrics.get("Mistral Substitutions", 0)} ({metrics.get("Mistral Substitution Rate", 0):.1f}%)</li>
                </ul>
            </div>
        </div>
    </div>
    """
    return card


def create_summary_visualization(
    avg_metrics: Dict[str, float],
    time_comparison: Dict[str, Any],
    individual_metrics: List[Dict[str, Any]] = None,
    results: List[Dict[str, Any]] = None,
) -> HTMLString:
    """Create an HTML visualization of evaluation results.

    Args:
        avg_metrics: Average metrics for all images
        time_comparison: Processing time comparison between models
        individual_metrics: Metrics for individual images
        results: Raw results with text data

    Returns:
        HTMLString: HTML visualization
    """
    step_context = get_step_context()
    pipeline_run_name = step_context.pipeline_run.name

    # Summary metrics for display in cards
    has_metrics = individual_metrics is not None and len(individual_metrics) > 0

    # Prepare sample card if we have data
    sample_cards = ""
    if has_metrics and results:
        # Show up to 3 examples
        for i, (metrics, result) in enumerate(zip(individual_metrics[:3], results[:3])):
            image_name = result["image_name"]
            ground_truth_text = result.get("ground_truth_text", result.get("raw_text_gt", "No ground truth available"))
            gemma_text = result["raw_text"]
            mistral_text = result["raw_text_right"]

            sample_cards += create_model_comparison_card(
                image_name, ground_truth_text, gemma_text, mistral_text, metrics
            )

    # Summary metrics section
    metrics_section = ""
    if has_metrics:
        metrics_section = f"""
        <div class="mb-8">
            <h2 class="text-2xl font-bold mb-4">OCR Model Performance Metrics</h2>

            <div class="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
                <div class="bg-white shadow rounded-lg p-4 border border-gray-200">
                    <h3 class="text-lg font-bold mb-2 text-blue-600">Gemma Metrics</h3>
                    <div class="grid grid-cols-2 gap-2">
                        <div class="text-gray-600">CER:</div>
                        <div class="text-right font-medium">{avg_metrics["avg_gemma_cer"]:.4f}</div>
                        <div class="text-gray-600">WER:</div>
                        <div class="text-right font-medium">{avg_metrics["avg_gemma_wer"]:.4f}</div>
                        <div class="text-gray-600">GT Similarity:</div>
                        <div class="text-right font-medium">{avg_metrics["avg_gemma_gt_similarity"]:.4f}</div>
                        <div class="text-gray-600">Proc. Time:</div>
                        <div class="text-right font-medium">{time_comparison["avg_gemma_time"]:.2f}s</div>
                    </div>
                </div>

                <div class="bg-white shadow rounded-lg p-4 border border-gray-200">
                    <h3 class="text-lg font-bold mb-2 text-purple-600">Mistral Metrics</h3>
                    <div class="grid grid-cols-2 gap-2">
                        <div class="text-gray-600">CER:</div>
                        <div class="text-right font-medium">{avg_metrics["avg_mistral_cer"]:.4f}</div>
                        <div class="text-gray-600">WER:</div>
                        <div class="text-right font-medium">{avg_metrics["avg_mistral_wer"]:.4f}</div>
                        <div class="text-gray-600">GT Similarity:</div>
                        <div class="text-right font-medium">{avg_metrics["avg_mistral_gt_similarity"]:.4f}</div>
                        <div class="text-gray-600">Proc. Time:</div>
                        <div class="text-right font-medium">{time_comparison["avg_mistral_time"]:.2f}s</div>
                    </div>
                </div>
                
                <div class="bg-white shadow rounded-lg p-4 border border-gray-200">
                    <h3 class="text-lg font-bold mb-2 text-gray-700">Model Comparison</h3>
                    <div class="grid grid-cols-2 gap-2">
                        <div class="text-gray-600">Models Similarity:</div>
                        <div class="text-right font-medium">{avg_metrics["avg_models_similarity"]:.4f}</div>
                        <div class="text-gray-600">Time Diff:</div>
                        <div class="text-right font-medium">{time_comparison["time_difference"]:.2f}s</div>
                        <div class="text-gray-600">Faster Model:</div>
                        <div class="text-right font-medium">{time_comparison["faster_model"]}</div>
                        <div class="text-gray-600">Better CER:</div>
                        <div class="text-right font-medium">
                            {"Gemma" if avg_metrics["avg_gemma_cer"] < avg_metrics["avg_mistral_cer"] else "Mistral"}
                        </div>
                    </div>
                </div>
            </div>
        </div>
        """
    else:
        # If we only have time metrics
        metrics_section = f"""
        <div class="mb-8">
            <h2 class="text-2xl font-bold mb-4">Model Performance Metrics</h2>

            <div class="bg-white shadow rounded-lg p-4 border border-gray-200 mb-6">
                <h3 class="text-lg font-bold mb-2">Processing Time Comparison</h3>
                <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <div>
                        <div class="text-gray-600 mb-1">Gemma Time</div>
                        <div class="text-xl font-bold">{time_comparison["avg_gemma_time"]:.2f}s</div>
                    </div>
                    <div>
                        <div class="text-gray-600 mb-1">Mistral Time</div>
                        <div class="text-xl font-bold">{time_comparison["avg_mistral_time"]:.2f}s</div>
                    </div>
                    <div>
                        <div class="text-gray-600 mb-1">Time Difference</div>
                        <div class="text-xl font-bold">{time_comparison["time_difference"]:.2f}s</div>
                    </div>
                    <div>
                        <div class="text-gray-600 mb-1">Faster Model</div>
                        <div class="text-xl font-bold">{time_comparison["faster_model"]}</div>
                    </div>
                </div>
            </div>
        </div>
        """

    # Build the full HTML
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>OCR Model Evaluation Results</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <style>
            body {{
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                color: #333;
                line-height: 1.5;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
            }}
            .text-gradient {{
                background-clip: text;
                -webkit-background-clip: text;
                color: transparent;
                background-image: linear-gradient(90deg, #4f46e5, #7c3aed);
            }}
        </style>
    </head>
    <body class="bg-gray-50">
        <div class="container px-4 py-8 mx-auto">
            <header class="mb-8">
                <h1 class="text-3xl font-bold mb-2">OCR Model Evaluation Dashboard</h1>
                <p class="text-gray-600">Pipeline Run: {pipeline_run_name}</p>
            </header>

            {metrics_section}

            <div>
                <h2 class="text-2xl font-bold mb-4">Sample Results</h2>
                {sample_cards if sample_cards else '<p class="text-gray-600">No sample results available</p>'}
            </div>
        </div>
    </body>
    </html>
    """

    return HTMLString(html)


@step()
def evaluate_models(
    gemma_results: Dict[str, pl.DataFrame],
    mistral_results: Dict[str, pl.DataFrame],
    ground_truth: Optional[Dict[str, pl.DataFrame]] = None,
) -> Annotated[HTMLString, "ocr_visualization"]:
    """Compare the performance of Gemma 3 and MistralAI with visualization.

    Args:
        gemma_results: Dictionary containing gemma_results dataframe
        mistral_results: Dictionary containing mistral_results dataframe
        ground_truth: Optional dictionary containing ground_truth_results from OpenAI

    Returns:
        Tuple containing:
            - Dictionary with evaluation results
            - HTMLString visualization of the results
    """
    # Extract dataframes from dictionaries
    gemma_df = gemma_results["gemma_results"]
    mistral_df = mistral_results["mistral_results"]

    # Join results
    results = gemma_df.join(mistral_df, on=["id", "image_name"], how="inner")

    evaluation_metrics = []
    processed_results = []

    # Handle OpenAI ground truth or manually provided ground truth
    ground_truth_df = None
    if ground_truth is not None and isinstance(ground_truth, dict) and "ground_truth_results" in ground_truth:
        # We have ground truth from OpenAI
        ground_truth_df = ground_truth["ground_truth_results"]

    if ground_truth_df is not None:
        # calculate accuracy metrics
        results = results.join(
            ground_truth_df,
            on=["id", "image_name"],
            how="inner",
            suffix="_gt",  # Use a custom suffix to avoid naming conflicts
        )

        has_ground_truth_text = "ground_truth_text" in results.columns
        ground_truth_text_col = "ground_truth_text" if has_ground_truth_text else "raw_text_gt"

        for row in results.iter_rows(named=True):
            metrics = compare_results(
                row[ground_truth_text_col],  # ground truth text
                row["raw_text"],  # gemma text
                row["raw_text_right"],  # mistral text (after join)
            )

            # calculate text similarity between models
            models_similarity = textdistance.jaccard.normalized_similarity(row["raw_text"], row["raw_text_right"])

            # Calculate similarity of each model to ground truth
            gemma_gt_similarity = textdistance.jaccard.normalized_similarity(
                row["raw_text"], row[ground_truth_text_col]
            )
            mistral_gt_similarity = textdistance.jaccard.normalized_similarity(
                row["raw_text_right"], row[ground_truth_text_col]
            )

            metrics["id"] = row["id"]
            metrics["image_name"] = row["image_name"]
            metrics["models_similarity"] = models_similarity
            metrics["gemma_gt_similarity"] = gemma_gt_similarity
            metrics["mistral_gt_similarity"] = mistral_gt_similarity

            evaluation_metrics.append(metrics)
            processed_results.append(dict(row))

    # calculate summary metrics
    if evaluation_metrics:
        df_metrics = pl.DataFrame(evaluation_metrics)

        # averages
        avg_metrics = {
            "avg_gemma_cer": df_metrics["Gemma CER"].mean(),
            "avg_gemma_wer": df_metrics["Gemma WER"].mean(),
            "avg_mistral_cer": df_metrics["Mistral CER"].mean(),
            "avg_mistral_wer": df_metrics["Mistral WER"].mean(),
            "avg_models_similarity": df_metrics["models_similarity"].mean(),
            "avg_gemma_gt_similarity": df_metrics["gemma_gt_similarity"].mean(),
            "avg_mistral_gt_similarity": df_metrics["mistral_gt_similarity"].mean(),
        }

        # Calculate timing metrics
        gemma_times = gemma_df.select("processing_time").to_series().mean()
        mistral_times = mistral_df.select("processing_time").to_series().mean()

        time_comparison = {
            "avg_gemma_time": gemma_times,
            "avg_mistral_time": mistral_times,
            "time_difference": abs(gemma_times - mistral_times),
            "faster_model": "Gemma" if gemma_times < mistral_times else "Mistral",
        }

        # Create HTML visualization
        html_visualization = create_summary_visualization(
            avg_metrics=avg_metrics,
            time_comparison={
                "avg_gemma_time": float(time_comparison["avg_gemma_time"]),
                "avg_mistral_time": float(time_comparison["avg_mistral_time"]),
                "time_difference": float(time_comparison["time_difference"]),
                "faster_model": time_comparison["faster_model"],
            },
            individual_metrics=evaluation_metrics,
            results=processed_results,
        )

        # Log metadata for ZenML dashboard
        log_metadata(
            metadata={
                "avg_gemma_cer": float(avg_metrics["avg_gemma_cer"]),
                "avg_gemma_wer": float(avg_metrics["avg_gemma_wer"]),
                "avg_mistral_cer": float(avg_metrics["avg_mistral_cer"]),
                "avg_mistral_wer": float(avg_metrics["avg_mistral_wer"]),
                "avg_models_similarity": float(avg_metrics["avg_models_similarity"]),
                "avg_gemma_gt_similarity": float(avg_metrics["avg_gemma_gt_similarity"]),
                "avg_mistral_gt_similarity": float(avg_metrics["avg_mistral_gt_similarity"]),
                "avg_gemma_time": float(time_comparison["avg_gemma_time"]),
                "avg_mistral_time": float(time_comparison["avg_mistral_time"]),
                "time_difference": float(time_comparison["time_difference"]),
                "faster_model": time_comparison["faster_model"],
            }
        )

        # Return both the metrics dictionary and HTML visualization
        # only returns the visualization for now
        # return {
        #     "individual_metrics": evaluation_metrics,
        #     "average_metrics": avg_metrics,
        #     "time_comparison": {
        #         "avg_gemma_time": float(time_comparison["avg_gemma_time"]),
        #         "avg_mistral_time": float(time_comparison["avg_mistral_time"]),
        #         "time_difference": float(time_comparison["time_difference"]),
        #         "faster_model": time_comparison["faster_model"],
        #     },
        # },

        return html_visualization

    # if no ground truth, just return processing times comparison
    gemma_times = gemma_df.select("processing_time").to_series().mean()
    mistral_times = mistral_df.select("processing_time").to_series().mean()

    time_comparison = {
        "avg_gemma_time": gemma_times,
        "avg_mistral_time": mistral_times,
        "time_difference": abs(gemma_times - mistral_times),
        "faster_model": "Gemma" if gemma_times < mistral_times else "Mistral",
    }

    # Create HTML visualization for time comparison only
    html_visualization = create_summary_visualization(
        avg_metrics={},  # No metrics available
        time_comparison={
            "avg_gemma_time": float(time_comparison["avg_gemma_time"]),
            "avg_mistral_time": float(time_comparison["avg_mistral_time"]),
            "time_difference": float(time_comparison["time_difference"]),
            "faster_model": time_comparison["faster_model"],
        },
    )

    # Log metadata for ZenML dashboard
    log_metadata(
        metadata={
            "avg_gemma_time": float(time_comparison["avg_gemma_time"]),
            "avg_mistral_time": float(time_comparison["avg_mistral_time"]),
            "time_difference": float(time_comparison["time_difference"]),
            "faster_model": time_comparison["faster_model"],
        }
    )

    # Convert the results to a serializable format
    processed_results = [dict(row) for row in results.iter_rows(named=True)]

    # Return both the metrics dictionary and HTML visualization
    # return {
    #     "results": processed_results,
    #     "time_comparison": {
    #         "avg_gemma_time": float(time_comparison["avg_gemma_time"]),
    #         "avg_mistral_time": float(time_comparison["avg_mistral_time"]),
    #         "time_difference": float(time_comparison["time_difference"]),
    #         "faster_model": time_comparison["faster_model"],
    #     },
    # },

    return html_visualization
