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
"""This module contains the functions for creating the HTML visualizations."""

import base64
import os
from typing import Any, Dict, List

import polars as pl
from zenml import get_step_context
from zenml.logger import get_logger
from zenml.types import HTMLString

from utils.metrics import find_best_model
from utils.model_configs import MODEL_CONFIGS

logger = get_logger(__name__)


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


def create_comparison_table(
    models: List[str], metric_data: Dict[str, Dict[str, float]], metric_names: List[str]
) -> str:
    """Create an HTML table comparing metrics across models."""
    headers = "".join([f'<th class="p-2 border text-center">{model}</th>' for model in models])
    rows = ""
    for metric in metric_names:
        row_data = ""
        best_value = None
        best_index = -1
        for i, model in enumerate(models):
            value = metric_data[model].get(metric, 0)
            if "CER" in metric or "WER" in metric:  # Lower is better
                if best_value is None or value < best_value:
                    best_value = value
                    best_index = i
            else:  # Higher is better (e.g. similarity)
                if best_value is None or value > best_value:
                    best_value = value
                    best_index = i
        for i, model in enumerate(models):
            value = metric_data[model].get(metric, 0)
            highlight = "bg-green-100" if i == best_index else ""
            row_data += f'<td class="p-2 border text-right {highlight}">{value:.4f}</td>'
        rows += f"""
        <tr>
            <td class="p-2 border font-medium">{metric}</td>
            {row_data}
        </tr>
        """
    table = f"""
    <table class="min-w-full bg-white border border-gray-300 shadow-sm">
        <thead>
            <tr class="bg-gray-100">
                <th class="p-2 border text-left">Metric</th>
                {headers}
            </tr>
        </thead>
        <tbody>
            {rows}
        </tbody>
    </table>
    """
    return table


def create_model_card_with_logo(model_display: str, metrics: Dict[str, float]) -> str:
    """Create a card for a model with its logo and metrics."""
    logo_path = None
    for _, config in MODEL_CONFIGS.items():
        if config.display == model_display:
            logo_path = config.logo
            break
    if not logo_path or not os.path.exists(os.path.join("./assets/logos", logo_path)):
        logo_path = "default.svg"
    logo_b64 = load_svg_logo(logo_path)
    logo_html = f'<img src="data:image/svg+xml;base64,{logo_b64}" width="30" class="mr-2" alt="{model_display} logo">'
    metrics_rows = ""
    metric_keys = ["CER", "WER", "GT Similarity", "Proc. Time"]
    for key in metric_keys:
        if key in metrics:
            metrics_rows += f"""
            <div class="grid grid-cols-2 gap-2">
                <div class="text-gray-600">{key}:</div>
                <div class="text-right font-medium">{metrics[key]:.4f}</div>
            </div>
            """
    card = f"""
    <div class="bg-white shadow rounded-lg p-4 border border-gray-200">
        <h3 class="text-lg font-bold mb-2 flex items-center">
            {logo_html} {model_display} Metrics
        </h3>
        {metrics_rows}
    </div>
    """
    return card


def create_model_comparison_card(
    image_name: str,
    ground_truth: str,
    model_texts: Dict[str, str],
    model_metrics: Dict[str, Dict[str, Any]],
) -> str:
    """Create a card for comparing OCR results for a specific image across models.

    Args:
        image_name: Name of the image
        ground_truth: Ground truth text
        model_texts: Dictionary mapping model names to their extracted text
        model_metrics: Dictionary of metrics for each model

    Returns:
        HTML card as a string
    """
    model_sections = ""
    num_models = len(model_texts)
    cols_class = "grid-cols-1"
    if num_models <= 3:
        cols_class = f"grid-cols-1 md:grid-cols-{num_models + 1}"  # +1 for ground truth
    else:
        cols_per_row = min(3, (num_models + 1) // 2)
        cols_class = f"grid-cols-1 md:grid-cols-{cols_per_row}"
    for model_display, text in model_texts.items():
        logo_path = None
        for _, config in MODEL_CONFIGS.items():
            if config.display == model_display:
                logo_path = config.logo
                break
        if not logo_path:
            logo_path = "default.svg"
        logo_b64 = load_svg_logo(logo_path)
        logo_html = f'<img src="data:image/svg+xml;base64,{logo_b64}" width="20" class="inline mr-1" alt="{model_display} logo">'
        model_sections += f"""
        <div class="border rounded-lg p-4 bg-gray-50">
            <h4 class="font-bold mb-2 text-blue-600 flex items-center">
                {logo_html} {model_display} Output
            </h4>
            <div class="whitespace-pre-wrap text-sm">{text}</div>
        </div>
        """
    metrics_table = create_comparison_table(
        list(model_texts.keys()), model_metrics, ["CER", "WER", "GT Similarity"]
    )
    error_sections = ""
    error_cols = min(3, len(model_texts))
    for model_display, metrics in model_metrics.items():
        error_sections += f"""
        <div>
            <h4 class="font-bold mb-2">{model_display} Errors</h4>
            <ul class="list-disc pl-5">
                <li>Insertions: {metrics.get("Insertions", 0)} ({metrics.get("Insertion Rate", 0):.1f}%)</li>
                <li>Deletions: {metrics.get("Deletions", 0)} ({metrics.get("Deletion Rate", 0):.1f}%)</li>
                <li>Substitutions: {metrics.get("Substitutions", 0)} ({metrics.get("Substitution Rate", 0):.1f}%)</li>
            </ul>
        </div>
        """

    # Simple header for ground truth text files
    ground_truth_header = '<h4 class="font-bold mb-2 text-gray-700">📄 Ground Truth</h4>'

    card = f"""
    <div class="bg-white rounded-lg shadow-md p-6 mb-6 border border-gray-200">
        <h3 class="text-xl font-bold mb-4 text-blue-700">{image_name}</h3>
        
        <div class="{cols_class} gap-4 mb-6">
            <div class="border rounded-lg p-4 bg-gray-50">
                {ground_truth_header}
                <div class="whitespace-pre-wrap text-sm">{ground_truth}</div>
            </div>
            
            {model_sections}
        </div>

        <div class="mb-4">
            <h4 class="font-bold mb-2">📊 Key Metrics</h4>
            {metrics_table}
        </div>

        <div class="grid grid-cols-1 md:grid-cols-{error_cols} gap-4">
            {error_sections}
        </div>
    </div>
    """
    return card


def create_model_similarity_matrix(models: List[str], similarities: Dict[str, float]) -> str:
    """Create a matrix showing similarity between model outputs."""
    headers = "".join([f'<th class="p-2 border text-center">{model}</th>' for model in models])
    rows = ""
    for i, model1 in enumerate(models):
        row_cells = ""
        for j, model2 in enumerate(models):
            if i == j:
                row_cells += '<td class="p-2 border text-center bg-gray-200">1.0000</td>'
                continue
            key1 = f"{model1}_{model2}"
            key2 = f"{model2}_{model1}"
            similarity = similarities.get(key1, similarities.get(key2, 0))
            row_cells += f'<td class="p-2 border text-center">{similarity:.4f}</td>'
        rows += f"""
        <tr>
            <td class="p-2 border font-medium">{model1}</td>
            {row_cells}
        </tr>
        """
    table = f"""
    <div class="mt-4">
        <h3 class="text-lg font-bold mb-2">📈 Model Similarity Matrix</h3>
        <p class="text-sm text-gray-600 mb-2">Higher values indicate more similar outputs between models</p>
        <table class="min-w-full bg-white border border-gray-300 shadow-sm">
            <thead>
                <tr class="bg-gray-100">
                    <th class="p-2 border"></th>
                    {headers}
                </tr>
            </thead>
            <tbody>
                {rows}
            </tbody>
        </table>
    </div>
    """
    return table


def create_summary_visualization(
    model_metrics: Dict[str, Dict[str, float]],
    time_comparison: Dict[str, Any],
    similarities: Dict[str, float] = None,
) -> HTMLString:
    """Create an HTML visualization of evaluation results for multiple models."""
    step_context = get_step_context()
    pipeline_run_name = step_context.pipeline_run.name
    models = list(model_metrics.keys())

    model_cards = ""
    cols_per_row = min(3, len(models))
    for model_display, metrics in model_metrics.items():
        prefix = None
        for _, config in MODEL_CONFIGS.items():
            if config.display == model_display:
                prefix = config.prefix
                break
        if not prefix:
            prefix = model_display.lower().replace(" ", "_")
        time_key = f"avg_{prefix}_time"
        if time_key in time_comparison:
            metrics["Proc. Time"] = time_comparison[time_key]
        model_cards += create_model_card_with_logo(model_display, metrics)

    fastest_model = time_comparison["fastest_model"]

    best_cer = find_best_model(model_metrics, "CER", lower_is_better=True)
    best_wer = find_best_model(model_metrics, "WER", lower_is_better=True)
    best_similarity = find_best_model(model_metrics, "GT Similarity", lower_is_better=False)

    metrics_grid = f"""
    <div class="grid grid-cols-1 md:grid-cols-{cols_per_row} gap-6 mb-6">
        {model_cards}
        <div class="bg-white shadow rounded-lg p-4 border border-gray-200">
            <h3 class="text-lg font-bold mb-2 flex items-center">
                🏆 Overall Best
            </h3>
            <div class="grid grid-cols-2 gap-2">
                <div class="text-gray-600">Fastest Model:</div>
                <div class="text-right font-medium">{fastest_model}</div>
                <div class="text-gray-600">Best CER:</div>
                <div class="text-right font-medium">{best_cer}</div>
                <div class="text-gray-600">Best WER:</div>
                <div class="text-right font-medium">{best_wer}</div>
                <div class="text-gray-600">Best Similarity:</div>
                <div class="text-right font-medium">{best_similarity}</div>
            </div>
        </div>
    </div>
    """

    comparison_metrics = ["CER", "WER", "GT Similarity"]
    comparison_table = create_comparison_table(models, model_metrics, comparison_metrics)
    similarity_matrix = ""
    if similarities and len(models) > 1:
        similarity_matrix = create_model_similarity_matrix(models, similarities)
    metrics_section = f"""
    <div class="mb-8">
        <h2 class="text-2xl font-bold mb-4">📊 OCR Model Performance Metrics</h2>
        {metrics_grid}
        
        <div class="bg-white shadow rounded-lg p-4 border border-gray-200 mb-6">
            <h3 class="text-lg font-bold mb-2">📈 Model Comparison</h3>
            {comparison_table}
            {similarity_matrix}
        </div>
    </div>
    """
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
                <h1 class="text-3xl font-bold mb-2">🔍 OCR Model Evaluation Dashboard</h1>
                <p class="text-gray-600">Pipeline Run: {pipeline_run_name}</p>
            </header>
            {metrics_section}
        </div>
    </body>
    </html>
    """
    return HTMLString(html)


def create_ocr_batch_visualization(df: pl.DataFrame) -> HTMLString:
    """Create an HTML visualization of batch OCR processing results."""
    # Extract metrics
    total_results = len(df)
    # Ensure all raw_text values are strings
    raw_texts = []
    for txt in df["raw_text"].to_list():
        if isinstance(txt, list):
            raw_texts.append("\n".join(txt))
        else:
            raw_texts.append(str(txt))
    total_chars = sum(len(txt) for txt in raw_texts)
    avg_conf = df["confidence"].mean() if "confidence" in df.columns else 0
    total_proc_time = df["processing_time"].sum() if "processing_time" in df.columns else 0
    avg_proc_time = df["processing_time"].mean() if "processing_time" in df.columns else 0

    # Get model-specific metrics
    model_metrics = {}
    model_displays = []

    if "model_name" in df.columns:
        for model in df["model_name"].unique().to_list():
            mdf = df.filter(pl.col("model_name") == model)
            # Ensure all model-specific raw_text values are strings
            m_raw_texts = []
            for txt in mdf["raw_text"].to_list():
                if isinstance(txt, list):
                    m_raw_texts.append("\n".join(txt))
                else:
                    m_raw_texts.append(str(txt))
            m_chars = sum(len(txt) for txt in m_raw_texts)
            m_conf = mdf["confidence"].mean() if "confidence" in mdf.columns else 0
            m_total_time = mdf["processing_time"].sum() if "processing_time" in mdf.columns else 0
            m_avg_time = mdf["processing_time"].mean() if "processing_time" in mdf.columns else 0

            model_metrics[model] = {
                "total_images": len(mdf),
                "total_chars": m_chars,
                "avg_confidence": m_conf,
                "total_time": m_total_time,
                "avg_time": m_avg_time,
                "char_per_second": m_chars / m_total_time if m_total_time > 0 else 0,
            }
            model_displays.append(model)

    # Create model cards HTML
    model_cards = ""
    if model_displays:
        cols_per_row = min(3, len(model_displays))
        for model in model_displays:
            metrics = model_metrics[model]

            # Try to get the model logo if available
            logo_path = None
            logo_html = ""
            try:
                for _, config in MODEL_CONFIGS.items():
                    if config.display == model:
                        logo_path = config.logo
                        break
                if logo_path and os.path.exists(os.path.join("./assets/logos", logo_path)):
                    logo_b64 = load_svg_logo(logo_path)
                    logo_html = f'<img src="data:image/svg+xml;base64,{logo_b64}" width="30" class="mr-2" alt="{model} logo">'
            except Exception as e:
                logger.warning(f"Error loading logo for {model}: {e}")
                # Default to just the model name without logo if there's an issue
                pass

            model_cards += f"""
            <div class="bg-white shadow rounded-lg p-4 border border-gray-200">
                <h3 class="text-lg font-bold mb-2 flex items-center">
                    {logo_html} {model}
                </h3>
                <div class="grid grid-cols-2 gap-2">
                    <div class="text-gray-600">Images:</div>
                    <div class="text-right font-medium">{metrics["total_images"]}</div>
                    <div class="text-gray-600">Characters:</div>
                    <div class="text-right font-medium">{metrics["total_chars"]}</div>
                    <div class="text-gray-600">Avg Confidence:</div>
                    <div class="text-right font-medium">{metrics["avg_confidence"]:.2f}</div>
                    <div class="text-gray-600">Total Time:</div>
                    <div class="text-right font-medium">{metrics["total_time"]:.2f}s</div>
                    <div class="text-gray-600">Avg Time/Image:</div>
                    <div class="text-right font-medium">{metrics["avg_time"]:.2f}s</div>
                    <div class="text-gray-600">Chars/Second:</div>
                    <div class="text-right font-medium">{metrics["char_per_second"]:.1f}</div>
                </div>
            </div>
            """

        model_grid = f"""
        <div class="grid grid-cols-1 md:grid-cols-{cols_per_row} gap-6 mb-6">
            {model_cards}
        </div>
        """
    else:
        # Single model view
        model_grid = f"""
        <div class="grid grid-cols-1 md:grid-cols-1 gap-6 mb-6">
            <div class="bg-white shadow rounded-lg p-4 border border-gray-200">
                <h3 class="text-lg font-bold mb-2 flex items-center">
                    OCR Processing Summary
                </h3>
                <div class="grid grid-cols-2 gap-2">
                    <div class="text-gray-600">Total Images:</div>
                    <div class="text-right font-medium">{total_results}</div>
                    <div class="text-gray-600">Total Characters:</div>
                    <div class="text-right font-medium">{total_chars}</div>
                    <div class="text-gray-600">Avg Confidence:</div>
                    <div class="text-right font-medium">{avg_conf:.2f}</div>
                    <div class="text-gray-600">Total Process Time:</div>
                    <div class="text-right font-medium">{total_proc_time:.2f}s</div>
                    <div class="text-gray-600">Avg Time/Image:</div>
                    <div class="text-right font-medium">{avg_proc_time:.2f}s</div>
                    <div class="text-gray-600">Chars/Second:</div>
                    <div class="text-right font-medium">{total_chars / total_proc_time if total_proc_time > 0 else 0:.1f}</div>
                </div>
            </div>
        </div>
        """

    # Create results table with sample data
    sample_size = min(10, total_results)  # Show up to 10 samples

    # Create table HTML
    table_rows = ""
    sample_df = df.head(sample_size)

    for i in range(sample_df.height):
        row = sample_df.row(i, named=True)
        model_col = (
            f'<td class="p-2 border">{row["model_name"]}</td>' if "model_name" in df.columns else ""
        )

        # Ensure raw_text is a string and limit displayed text length
        raw_text = row["raw_text"]
        if isinstance(raw_text, list):
            raw_text = "\n".join(raw_text)
        text_preview = str(raw_text)[:100] + ("..." if len(str(raw_text)) > 100 else "")

        # Calculate the length properly
        text_length = len(str(raw_text)) if raw_text is not None else 0

        table_rows += f"""
        <tr>
            <td class="p-2 border">{row["image_name"]}</td>
            {model_col}
            <td class="p-2 border">{text_length}</td>
            <td class="p-2 border">{row.get("confidence", 0):.2f}</td>
            <td class="p-2 border">{row.get("processing_time", 0):.2f}s</td>
            <td class="p-2 border text-xs max-w-xs overflow-hidden">{text_preview}</td>
        </tr>
        """

    model_header = '<th class="p-2 border">Model</th>' if "model_name" in df.columns else ""

    results_table = f"""
    <div class="bg-white shadow rounded-lg p-4 border border-gray-200 mb-6 overflow-x-auto">
        <h3 class="text-lg font-bold mb-2">Sample Results ({sample_size} of {total_results})</h3>
        <table class="min-w-full bg-white border border-gray-300">
            <thead>
                <tr class="bg-gray-100">
                    <th class="p-2 border">Image</th>
                    {model_header}
                    <th class="p-2 border">Chars</th>
                    <th class="p-2 border">Confidence</th>
                    <th class="p-2 border">Time (s)</th>
                    <th class="p-2 border">Text Preview</th>
                </tr>
            </thead>
            <tbody>
                {table_rows}
            </tbody>
        </table>
    </div>
    """

    # Final HTML
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>OCR Batch Processing Results</title>
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
                <h1 class="text-3xl font-bold mb-2">📝 OCR Batch Processing Results</h1>
                <p class="text-gray-600">Processed {total_results} images with {total_chars} total characters in {total_proc_time:.2f}s</p>
            </header>

            <div class="mb-8">
                <h2 class="text-2xl font-bold mb-4">Processing Summary</h2>
                {model_grid}
                {results_table}
            </div>
        </div>
    </body>
    </html>
    """

    return HTMLString(html)
