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

from typing import Any, Dict, List, Optional

import polars as pl
import textdistance
from typing_extensions import Annotated
from zenml import get_step_context, log_metadata, step
from zenml.types import HTMLString

from utils import compare_results, get_model_info


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
    image_name: str,
    ground_truth: str,
    model1_text: str,
    model2_text: str,
    metrics: Dict[str, Any],
    model1_name: str,
    model2_name: str,
) -> str:
    """Create a card for comparing OCR results for a specific image.

    Args:
        image_name: Name of the analyzed image
        ground_truth: Ground truth text
        model1_text: Text extracted by first model
        model2_text: Text extracted by second model
        metrics: Metrics for this comparison
        model1_name: Name of the first model
        model2_name: Name of the second model

    Returns:
        str: HTML card string
    """
    model1_display, model1_prefix = get_model_info(model1_name)
    model2_display, model2_prefix = get_model_info(model2_name)

    basic_metrics = {
        f"{model1_display} CER": metrics[f"{model1_display} CER"],
        f"{model1_display} WER": metrics[f"{model1_display} WER"],
        f"{model2_display} CER": metrics[f"{model2_display} CER"],
        f"{model2_display} WER": metrics[f"{model2_display} WER"],
        "Models Similarity": metrics["models_similarity"],
        f"{model1_display}-GT Similarity": metrics[f"{model1_prefix}_gt_similarity"],
        f"{model2_display}-GT Similarity": metrics[f"{model2_prefix}_gt_similarity"],
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
                <h4 class="font-bold mb-2 text-blue-600">{model1_display} Output</h4>
                <div class="whitespace-pre-wrap text-sm">{model1_text}</div>
            </div>
            
            <div class="border rounded-lg p-4 bg-gray-50">
                <h4 class="font-bold mb-2 text-purple-600">{model2_display} Output</h4>
                <div class="whitespace-pre-wrap text-sm">{model2_text}</div>
            </div>
        </div>

        <div class="mb-4">
            <h4 class="font-bold mb-2">Key Metrics</h4>
            {metrics_table}
        </div>

        <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
                <h4 class="font-bold mb-2">{model1_display} Errors</h4>
                <ul class="list-disc pl-5">
                    <li>Insertions: {metrics.get(f"{model1_display} Insertions", 0)} ({metrics.get(f"{model1_display} Insertion Rate", 0):.1f}%)</li>
                    <li>Deletions: {metrics.get(f"{model1_display} Deletions", 0)} ({metrics.get(f"{model1_display} Deletion Rate", 0):.1f}%)</li>
                    <li>Substitutions: {metrics.get(f"{model1_display} Substitutions", 0)} ({metrics.get(f"{model1_display} Substitution Rate", 0):.1f}%)</li>
                </ul>
            </div>

            <div>
                <h4 class="font-bold mb-2">{model2_display} Errors</h4>
                <ul class="list-disc pl-5">
                    <li>Insertions: {metrics.get(f"{model2_display} Insertions", 0)} ({metrics.get(f"{model2_display} Insertion Rate", 0):.1f}%)</li>
                    <li>Deletions: {metrics.get(f"{model2_display} Deletions", 0)} ({metrics.get(f"{model2_display} Deletion Rate", 0):.1f}%)</li>
                    <li>Substitutions: {metrics.get(f"{model2_display} Substitutions", 0)} ({metrics.get(f"{model2_display} Substitution Rate", 0):.1f}%)</li>
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
    model1_name: str = "model1",
    model2_name: str = "model2",
) -> HTMLString:
    """Create an HTML visualization of evaluation results.

    Args:
        avg_metrics: Average metrics for all images
        time_comparison: Processing time comparison between models
        individual_metrics: Metrics for individual images
        results: Raw results with text data
        model1_name: Full name of the first model
        model2_name: Full name of the second model

    Returns:
        HTMLString: HTML visualization
    """
    model1_display, model1_prefix = get_model_info(model1_name)
    model2_display, model2_prefix = get_model_info(model2_name)

    step_context = get_step_context()
    pipeline_run_name = step_context.pipeline_run.name

    sample_cards = ""
    if individual_metrics and results:
        # Show up to 3 examples
        for i, (metrics, result) in enumerate(zip(individual_metrics[:3], results[:3])):
            image_name = result["image_name"]
            ground_truth_text = result.get(
                "ground_truth_text", result.get("raw_text_gt", "No ground truth available")
            )
            model1_text = result["raw_text"]
            model2_text = result["raw_text_right"]

            sample_cards += create_model_comparison_card(
                image_name,
                ground_truth_text,
                model1_text,
                model2_text,
                metrics,
                model1_name,
                model2_name,
            )

    # Summary metrics section
    if avg_metrics:
        metrics_section = f"""
        <div class="mb-8">
            <h2 class="text-2xl font-bold mb-4">OCR Model Performance Metrics</h2>
            <div class="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
                <div class="bg-white shadow rounded-lg p-4 border border-gray-200">
                    <h3 class="text-lg font-bold mb-2 text-blue-600">{model1_display} Metrics</h3>
                    <div class="grid grid-cols-2 gap-2">
                        <div class="text-gray-600">CER:</div>
                        <div class="text-right font-medium">{avg_metrics[f"avg_{model1_prefix}_cer"]:.4f}</div>
                        <div class="text-gray-600">WER:</div>
                        <div class="text-right font-medium">{avg_metrics[f"avg_{model1_prefix}_wer"]:.4f}</div>
                        <div class="text-gray-600">GT Similarity:</div>
                        <div class="text-right font-medium">{avg_metrics[f"avg_{model1_prefix}_gt_similarity"]:.4f}</div>
                        <div class="text-gray-600">Proc. Time:</div>
                        <div class="text-right font-medium">{time_comparison[f"avg_{model1_prefix}_time"]:.2f}s</div>
                    </div>
                </div>
                <div class="bg-white shadow rounded-lg p-4 border border-gray-200">
                    <h3 class="text-lg font-bold mb-2 text-purple-600">{model2_display} Metrics</h3>
                    <div class="grid grid-cols-2 gap-2">
                        <div class="text-gray-600">CER:</div>
                        <div class="text-right font-medium">{avg_metrics[f"avg_{model2_prefix}_cer"]:.4f}</div>
                        <div class="text-gray-600">WER:</div>
                        <div class="text-right font-medium">{avg_metrics[f"avg_{model2_prefix}_wer"]:.4f}</div>
                        <div class="text-gray-600">GT Similarity:</div>
                        <div class="text-right font-medium">{avg_metrics[f"avg_{model2_prefix}_gt_similarity"]:.4f}</div>
                        <div class="text-gray-600">Proc. Time:</div>
                        <div class="text-right font-medium">{time_comparison[f"avg_{model2_prefix}_time"]:.2f}s</div>
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
                            {model1_display if avg_metrics[f"avg_{model1_prefix}_cer"] < avg_metrics[f"avg_{model2_prefix}_cer"] else model2_display}
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
                        <div class="text-gray-600 mb-1">{model1_display} Time</div>
                        <div class="text-xl font-bold">{time_comparison[f"avg_{model1_prefix}_time"]:.2f}s</div>
                    </div>
                    <div>
                        <div class="text-gray-600 mb-1">{model2_display} Time</div>
                        <div class="text-xl font-bold">{time_comparison[f"avg_{model2_prefix}_time"]:.2f}s</div>
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
    model1_df: pl.DataFrame,
    model2_df: pl.DataFrame,
    ground_truth_df: Optional[pl.DataFrame] = None,
    model1_name: str = "ollama/gemma3:27b",
    model2_name: str = "pixtral-12b-2409",
) -> Annotated[HTMLString, "ocr_visualization"]:
    """Compare the performance of two configurable models with visualization.

    Args:
        model1_df: First model results DataFrame
        model2_df: Second model results DataFrame
        ground_truth_df: Optional ground truth results DataFrame
        model1_name: Name of the first model (default: ollama/gemma3:27b)
        model2_name: Name of the second model (default: pixtral-12b-2409)
        model1_display: Display name for the first model (default: Gemma)
        model2_display: Display name for the second model (default: Mistral)

    Returns:
        HTMLString visualization of the results
    """
    model1_display, model1_prefix = get_model_info(model1_name)
    model2_display, model2_prefix = get_model_info(model2_name)

    # Join results
    results = model1_df.join(model2_df, on=["id", "image_name"], how="inner")
    evaluation_metrics = []
    processed_results = []

    if ground_truth_df is not None:
        results = results.join(
            ground_truth_df,
            on=["id", "image_name"],
            how="inner",
            suffix="_gt",
        )

        has_ground_truth_text = "ground_truth_text" in results.columns
        ground_truth_text_col = "ground_truth_text" if has_ground_truth_text else "raw_text_gt"

        for row in results.iter_rows(named=True):
            metrics = compare_results(
                row[ground_truth_text_col],
                row["raw_text"],
                row["raw_text_right"],
                model1_display=model1_display,
                model2_display=model2_display,
            )

            models_similarity = textdistance.jaccard.normalized_similarity(
                row["raw_text"], row["raw_text_right"]
            )

            model1_similarity = textdistance.jaccard.normalized_similarity(
                row["raw_text"], row[ground_truth_text_col]
            )
            model2_similarity = textdistance.jaccard.normalized_similarity(
                row["raw_text_right"], row[ground_truth_text_col]
            )

            metrics["id"] = row["id"]
            metrics["image_name"] = row["image_name"]
            metrics["models_similarity"] = models_similarity
            metrics[f"{model1_prefix}_gt_similarity"] = model1_similarity
            metrics[f"{model2_prefix}_gt_similarity"] = model2_similarity

            evaluation_metrics.append(metrics)
            processed_results.append(dict(row))

        # Calculate summary metrics
        if evaluation_metrics:
            df_metrics = pl.DataFrame(evaluation_metrics)

            # Calculate average metrics
            avg_metrics = {
                f"avg_{model1_prefix}_cer": df_metrics[f"{model1_display} CER"].mean(),
                f"avg_{model1_prefix}_wer": df_metrics[f"{model1_display} WER"].mean(),
                f"avg_{model2_prefix}_cer": df_metrics[f"{model2_display} CER"].mean(),
                f"avg_{model2_prefix}_wer": df_metrics[f"{model2_display} WER"].mean(),
                "avg_models_similarity": df_metrics["models_similarity"].mean(),
                f"avg_{model1_prefix}_gt_similarity": df_metrics[
                    f"{model1_prefix}_gt_similarity"
                ].mean(),
                f"avg_{model2_prefix}_gt_similarity": df_metrics[
                    f"{model2_prefix}_gt_similarity"
                ].mean(),
            }

            model1_times = model1_df.select("processing_time").to_series().mean()
            model2_times = model2_df.select("processing_time").to_series().mean()
            model1_time_key = f"avg_{model1_prefix}_time"
            model2_time_key = f"avg_{model2_prefix}_time"
            time_comparison = {
                model1_time_key: model1_times,
                model2_time_key: model2_times,
                "time_difference": abs(model1_times - model2_times),
                "faster_model": model1_display if model1_times < model2_times else model2_display,
            }

            # Log metadata for ZenML dashboard
            log_metadata(
                metadata={
                    f"avg_{model1_prefix}_cer": float(avg_metrics[f"avg_{model1_prefix}_cer"]),
                    f"avg_{model1_prefix}_wer": float(avg_metrics[f"avg_{model1_prefix}_wer"]),
                    f"avg_{model2_prefix}_cer": float(avg_metrics[f"avg_{model2_prefix}_cer"]),
                    f"avg_{model2_prefix}_wer": float(avg_metrics[f"avg_{model2_prefix}_wer"]),
                    "avg_models_similarity": float(avg_metrics["avg_models_similarity"]),
                    f"avg_{model1_prefix}_gt_similarity": float(
                        avg_metrics[f"avg_{model1_prefix}_gt_similarity"]
                    ),
                    f"avg_{model2_prefix}_gt_similarity": float(
                        avg_metrics[f"avg_{model2_prefix}_gt_similarity"]
                    ),
                    model1_time_key: float(time_comparison[model1_time_key]),
                    model2_time_key: float(time_comparison[model2_time_key]),
                    "time_difference": float(time_comparison["time_difference"]),
                    "faster_model": time_comparison["faster_model"],
                }
            )

            html_visualization = create_summary_visualization(
                avg_metrics=avg_metrics,
                time_comparison=time_comparison,
                individual_metrics=evaluation_metrics,
                results=processed_results,
                model1_name=model1_name,
                model2_name=model2_name,
            )

            return html_visualization

        # FALLBACK: if no ground truth metrics, only use processing times.
        model1_times = model1_df.select("processing_time").to_series().mean()
        model2_times = model2_df.select("processing_time").to_series().mean()
        model1_time_key = f"avg_{model1_prefix}_time"
        model2_time_key = f"avg_{model2_prefix}_time"
        time_comparison = {
            model1_time_key: model1_times,
            model2_time_key: model2_times,
            "time_difference": abs(model1_times - model2_times),
            "faster_model": model1_display if model1_times < model2_times else model2_display,
        }
        html_visualization = create_summary_visualization(
            avg_metrics={},
            time_comparison=time_comparison,
            model1_name=model1_name,
            model2_name=model2_name,
        )

        log_metadata(
            metadata={
                model1_time_key: float(time_comparison[model1_time_key]),
                model2_time_key: float(time_comparison[model2_time_key]),
                "time_difference": float(time_comparison["time_difference"]),
                "faster_model": time_comparison["faster_model"],
            }
        )

        return html_visualization
