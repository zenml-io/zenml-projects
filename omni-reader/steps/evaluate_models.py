"""This module contains the steps for evaluating the OCR models."""

import base64
import os
import re
import unicodedata
from typing import Any, Dict, List, Optional

import polars as pl
import textdistance
from jiwer import cer, wer
from typing_extensions import Annotated
from zenml import get_step_context, log_metadata, step
from zenml.types import HTMLString

from utils import compare_multi_model, get_model_info
from utils.model_configs import MODEL_CONFIGS


def load_svg_logo(logo_name: str) -> str:
    """Load an SVG logo as base64 encoded string."""
    logo_path = os.path.join("./assets/logos", logo_name)
    try:
        if os.path.exists(logo_path):
            with open(logo_path, "rb") as f:
                return base64.b64encode(f.read()).decode()
    except Exception:
        pass
    return ""


def create_metrics_table(metrics: Dict[str, float]) -> str:
    """Create an HTML table for metrics."""
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
                        <div class="text-gray-600">Fastest Model:</div>
                        <div class="text-right font-medium">{time_comparison["fastest_model"]}</div>
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
                        <div class="text-gray-600 mb-1">Fastest Model</div>
                        <div class="text-xl font-bold">{time_comparison["fastest_model"]}</div>
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
    model_results: Dict[str, pl.DataFrame],
    ground_truth_df: Optional[pl.DataFrame] = None,
    primary_models: Optional[List[str]] = None,
) -> Annotated[HTMLString, "ocr_visualization"]:
    """Compare the performance of multiple configurable models with visualization.

    Args:
        model_results: Dictionary mapping model names to their results DataFrames
        ground_truth_df: Optional ground truth results DataFrame
        primary_models: Optional list of model names to highlight in comparison.
            If None or less than 2 models, uses the first two models from model_results.

    Returns:
        HTMLString visualization of the results
    """
    # Ensure we have at least two models for comparison
    if len(model_results) < 2:
        raise ValueError("At least two models are required for comparison")

    # If primary_models not specified or invalid, use the first two models
    if not primary_models or len(primary_models) < 2:
        primary_models = list(model_results.keys())[:2]

    # Extract the primary models for main comparison
    model1_name = primary_models[0]
    model2_name = primary_models[1]

    model1_df = model_results[model1_name]
    model2_df = model_results[model2_name]

    model1_display, model1_prefix = get_model_info(model1_name)
    model2_display, model2_prefix = get_model_info(model2_name)

    # Join results
    results = model1_df.join(model2_df, on=["id", "image_name"], how="inner", suffix="_right")
    evaluation_metrics = []
    processed_results = []

    # Calculate processing times for all models
    all_model_times = {}
    for model_name, df in model_results.items():
        display, prefix = get_model_info(model_name)
        time_key = f"avg_{prefix}_time"
        all_model_times[time_key] = df.select("processing_time").to_series().mean()
        all_model_times[f"{prefix}_display"] = display

    # Find fastest model
    fastest_model_time = min(
        [(time, model) for model, time in all_model_times.items() if not model.endswith("_display")]
    )
    fastest_model_key = fastest_model_time[1]
    fastest_model_prefix = fastest_model_key.replace("avg_", "").replace("_time", "")
    fastest_model_display = all_model_times.get(
        f"{fastest_model_prefix}_display", fastest_model_prefix
    )

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

            model1_time_key = f"avg_{model1_prefix}_time"
            model2_time_key = f"avg_{model2_prefix}_time"

            # Combine processing times with other metrics
            time_comparison = {
                **all_model_times,
                "time_difference": abs(
                    all_model_times[model1_time_key] - all_model_times[model2_time_key]
                ),
                "fastest_model": fastest_model_display,
            }

            # Prepare metadata for ZenML dashboard
            metadata_dict = {
                **{
                    f"avg_{model}_time": float(time)
                    for model, time in all_model_times.items()
                    if not model.endswith("_display")
                },
                "fastest_model": fastest_model_display,
                "model_count": len(model_results),
                "avg_models_similarity": float(avg_metrics["avg_models_similarity"]),
            }

            # Add accuracy metrics for primary models
            metadata_dict.update(
                {
                    f"avg_{model1_prefix}_cer": float(avg_metrics[f"avg_{model1_prefix}_cer"]),
                    f"avg_{model1_prefix}_wer": float(avg_metrics[f"avg_{model1_prefix}_wer"]),
                    f"avg_{model2_prefix}_cer": float(avg_metrics[f"avg_{model2_prefix}_cer"]),
                    f"avg_{model2_prefix}_wer": float(avg_metrics[f"avg_{model2_prefix}_wer"]),
                    f"avg_{model1_prefix}_gt_similarity": float(
                        avg_metrics[f"avg_{model1_prefix}_gt_similarity"]
                    ),
                    f"avg_{model2_prefix}_gt_similarity": float(
                        avg_metrics[f"avg_{model2_prefix}_gt_similarity"]
                    ),
                }
            )

            # Log metadata for ZenML dashboard
            log_metadata(metadata=metadata_dict)

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
        time_comparison = {
            **all_model_times,
            "time_difference": abs(
                all_model_times[f"avg_{model1_prefix}_time"]
                - all_model_times[f"avg_{model2_prefix}_time"]
            ),
            "fastest_model": fastest_model_display,
        }

        html_visualization = create_summary_visualization(
            avg_metrics={},
            time_comparison=time_comparison,
            model1_name=model1_name,
            model2_name=model2_name,
        )

        # Prepare metadata for ZenML dashboard
        metadata_dict = {
            **{
                f"avg_{model}_time": float(time)
                for model, time in all_model_times.items()
                if not model.endswith("_display")
            },
            "fastest_model": fastest_model_display,
            "model_count": len(model_results),
        }

        log_metadata(metadata=metadata_dict)

        return html_visualization
