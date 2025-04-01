"""This module contains the steps for evaluating the OCR models."""

import base64
import os
import re
from typing import Any, Dict, List, Optional

import polars as pl
from jiwer import cer, wer
from textdistance import jaccard
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
    ground_truth_model: Optional[str] = None,
) -> str:
    """Create a card for comparing OCR results for a specific image across models."""
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

    # Add OpenAI logo to Ground Truth header if applicable
    ground_truth_header = '<h4 class="font-bold mb-2 text-gray-700">Ground Truth</h4>'
    if ground_truth_model and ground_truth_model in MODEL_CONFIGS:
        gt_config = MODEL_CONFIGS[ground_truth_model]
        if gt_config.logo:
            logo_b64 = load_svg_logo(gt_config.logo)
            ground_truth_header = f"""
            <h4 class="font-bold mb-2 text-gray-700 flex items-center">
                <img src="data:image/svg+xml;base64,{logo_b64}" width="20" class="inline mr-1" alt="{gt_config.display} logo">
                Ground Truth
            </h4>
            """

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
    ground_truth_model: str = None,
) -> HTMLString:
    """Create an HTML visualization of evaluation results for multiple models."""
    step_context = get_step_context()
    pipeline_run_name = step_context.pipeline_run.name
    models = list(model_metrics.keys())

    # Exclude ground truth model from best model calculations
    exclude_from_best = []
    if ground_truth_model:
        for display_name in models:
            for model_id, config in MODEL_CONFIGS.items():
                if config.display == display_name and model_id == ground_truth_model:
                    exclude_from_best.append(display_name)
                    break

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

    best_cer = find_best_model(
        model_metrics, "CER", lower_is_better=True, exclude_model_names=exclude_from_best
    )
    best_wer = find_best_model(
        model_metrics, "WER", lower_is_better=True, exclude_model_names=exclude_from_best
    )
    best_similarity = find_best_model(
        model_metrics, "GT Similarity", lower_is_better=False, exclude_model_names=exclude_from_best
    )

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


def normalize_text(s: str) -> str:
    """Normalize text for comparison."""
    s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()
    # Normalize apostrophes and similar characters
    s = re.sub(r"[''′`]", "'", s)
    return s


def calculate_model_similarities(
    results: List[Dict[str, Any]], model_displays: List[str]
) -> Dict[str, float]:
    """Calculate the average pairwise Jaccard similarity between model outputs.

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
        # Build a mapping from model display names to their corresponding text.
        model_texts = {}
        for display in model_displays:
            key = f"raw_text_{display.lower().replace(' ', '_')}"
            text = result.get(key, "")
            if isinstance(text, str):
                text = normalize_text(text)
                if text:
                    model_texts[display] = text

        # Only proceed if at least two models have valid text.
        if len(model_texts) < 2:
            continue

        # Compute pairwise similarity for each combination.
        for i in range(len(model_displays)):
            for j in range(i + 1, len(model_displays)):
                model1 = model_displays[i]
                model2 = model_displays[j]
                if model1 not in model_texts or model2 not in model_texts:
                    continue
                text1 = model_texts[model1]
                text2 = model_texts[model2]
                similarity = jaccard.normalized_similarity(text1, text2)
                pair_key = f"{model1}_{model2}"
                similarity_sums[pair_key] = similarity_sums.get(pair_key, 0) + similarity
                similarity_counts[pair_key] = similarity_counts.get(pair_key, 0) + 1

    # Average the similarities for each pair.
    similarities = {
        pair: similarity_sums[pair] / similarity_counts[pair] for pair in similarity_sums
    }
    return similarities


def find_best_model(
    model_metrics: Dict[str, Dict[str, float]],
    metric: str,
    lower_is_better: bool = True,
    exclude_model_names: List[str] = None,
) -> str:
    """Find the best performing model(s) for a given metric, showing ties when they occur."""
    best_models = []
    best_value = None
    exclude_model_names = exclude_model_names or []

    for model, metrics in model_metrics.items():
        if model in exclude_model_names:
            continue
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
            if model in exclude_model_names:
                continue
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
    ground_truth_text: str, model_texts: Dict[str, str], model_displays: List[str]
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
            all_metrics[model1]["GT Similarity"] = jaccard.normalized_similarity(
                ground_truth_text, text1
            )
        for j, model2 in enumerate(model_displays):
            if i < j:
                model_pairs.append((model1, model2))
    for model1, model2 in model_pairs:
        text1 = model_texts.get(model1, "")
        text2 = model_texts.get(model2, "")
        similarity = jaccard.normalized_similarity(text1, text2)
        pair_key = f"{model1}_{model2}"
        all_metrics[pair_key] = similarity
    return all_metrics


@step(enable_cache=False)
def evaluate_models(
    model_results: Dict[str, pl.DataFrame],
    ground_truth_df: Optional[pl.DataFrame] = None,
    ground_truth_model: Optional[str] = None,
) -> Annotated[HTMLString, "ocr_visualization"]:
    """Compare the performance of multiple configurable models with visualization.

    The ground truth model is separated from evaluation models so that it is used only
    for displaying the reference text. All metric calculations, similarity computations,
    and best model indicators are performed solely on evaluation models.
    """
    if not model_results:
        raise ValueError("At least one model is required for evaluation")

    # --- 1. Separate the ground truth model from evaluation models ---
    if ground_truth_model and ground_truth_model in model_results:
        gt_df = model_results[ground_truth_model].clone()
        del model_results[ground_truth_model]
    else:
        gt_df = None

    # If a separate ground_truth_df is provided, that overrides any GT model data.
    if ground_truth_df is not None:
        gt_df = ground_truth_df

    # --- 2. Build model info for evaluation models ---
    model_keys = list(model_results.keys())
    model_info = {}
    model_displays = []
    model_prefixes = {}
    for model_name in model_keys:
        display, prefix = get_model_info(model_name)
        model_info[model_name] = (display, prefix)
        model_displays.append(display)
        model_prefixes[display] = prefix

    # --- 3. Merge evaluation models' results ---
    base_model = model_keys[0]
    base_display, base_prefix = model_info[base_model]
    merged_results = model_results[base_model].clone()
    for i, model_name in enumerate(model_keys[1:], start=1):
        disp, pref = model_info[model_name]
        suffix = f"_{pref}" if i > 1 else "_right"
        merged_results = merged_results.join(
            model_results[model_name],
            on=["id", "image_name"],
            how="inner",
            suffix=suffix,
        )

    # --- 4. Join ground truth data if available ---
    if gt_df is not None:
        merged_results = merged_results.join(
            gt_df, on=["id", "image_name"], how="inner", suffix="_gt"
        )

    # --- 5. Calculate processing times for evaluation models ---
    all_model_times = {}
    for model_name, df in model_results.items():
        disp, pref = model_info[model_name]
        time_key = f"avg_{pref}_time"
        all_model_times[time_key] = df.select("processing_time").to_series().mean()
        all_model_times[f"{pref}_display"] = disp

    fastest_model_time, fastest_key = min(
        [(time, key) for key, time in all_model_times.items() if not key.endswith("_display")],
        key=lambda x: x[0],
    )
    fastest_prefix = fastest_key.replace("avg_", "").replace("_time", "")
    fastest_display = all_model_times.get(f"{fastest_prefix}_display", fastest_prefix)

    # --- 6. Per-image evaluation: compute metrics, error analysis, and build per-image cards ---
    evaluation_metrics = []
    image_cards_html = ""
    gt_text_col = "ground_truth_text"
    if gt_text_col not in merged_results.columns:
        if "raw_text_gt" in merged_results.columns:
            gt_text_col = "raw_text_gt"

    for row in merged_results.iter_rows(named=True):
        if gt_text_col not in row:
            continue
        ground_truth = row[gt_text_col]
        model_texts = {}
        model_texts[base_display] = row["raw_text"]
        for i, mkey in enumerate(model_keys[1:], start=1):
            disp, pref = model_info[mkey]
            col = "raw_text_right" if i == 1 else f"raw_text_{pref}"
            if col in row:
                model_texts[disp] = row[col]
        row_metrics = calculate_custom_metrics(ground_truth, model_texts, list(model_texts.keys()))
        error_analysis = compare_multi_model(ground_truth, model_texts)
        result_metrics = {"id": row["id"], "image_name": row["image_name"]}
        for disp in model_texts.keys():
            if disp in row_metrics:
                for met_name, val in row_metrics[disp].items():
                    result_metrics[f"{disp} {met_name}"] = val
        for disp, errs in error_analysis.items():
            for met_name, val in errs.items():
                result_metrics[f"{disp} {met_name}"] = val
        evaluation_metrics.append(result_metrics)

        # Merge GT Similarity values from row_metrics into error_analysis
        for disp in model_texts.keys():
            if disp in row_metrics and "GT Similarity" in row_metrics[disp]:
                if disp not in error_analysis:
                    error_analysis[disp] = {}
                error_analysis[disp]["GT Similarity"] = row_metrics[disp]["GT Similarity"]

        comparison_card = create_model_comparison_card(
            image_name=row["image_name"],
            ground_truth=ground_truth,
            model_texts=model_texts,
            model_metrics=error_analysis,
            ground_truth_model=ground_truth_model,
        )
        image_cards_html += comparison_card

    # --- 7. Compute average metrics for evaluation models ---
    model_metric_averages = {d: {} for d in model_displays}
    if evaluation_metrics:
        df_eval = pl.DataFrame(evaluation_metrics)
        for disp in model_displays:
            cer_col = f"{disp} CER"
            wer_col = f"{disp} WER"
            sim_col = f"{disp} GT Similarity"
            if cer_col in df_eval.columns:
                model_metric_averages[disp]["CER"] = df_eval[cer_col].mean()
            if wer_col in df_eval.columns:
                model_metric_averages[disp]["WER"] = df_eval[wer_col].mean()
            if sim_col in df_eval.columns:
                model_metric_averages[disp]["GT Similarity"] = df_eval[sim_col].mean()
    for disp in model_displays:
        pref = model_prefixes[disp]
        tkey = f"avg_{pref}_time"
        if tkey in all_model_times:
            model_metric_averages[disp]["Proc. Time"] = all_model_times[tkey]

    # --- 8. Calculate similarity matrix for evaluation models only ---
    sim_results = []
    for row in merged_results.iter_rows(named=True):
        texts_map = {}
        texts_map[base_display] = row.get("raw_text", "")
        for i, mkey in enumerate(model_keys[1:], start=1):
            disp, pref = model_info[mkey]
            col = "raw_text_right" if i == 1 else f"raw_text_{pref}"
            texts_map[disp] = row.get(col, "")
        sim_results.append(texts_map)
    similarities = {}
    if len(model_displays) > 1:
        similarities = calculate_model_similarities(
            results=[
                {
                    f"raw_text_{disp.lower().replace(' ', '_')}": texts_map[disp]
                    for disp in model_displays
                }
                for texts_map in sim_results
            ],
            model_displays=model_displays,
        )

    # --- 9. Build time comparison info ---
    time_comparison = {
        **all_model_times,
        "fastest_model": fastest_display,
        "model_count": len(model_keys),
    }
    if len(model_keys) >= 2:
        d1, p1 = model_info[model_keys[0]]
        d2, p2 = model_info[model_keys[1]]
        tk1, tk2 = f"avg_{p1}_time", f"avg_{p2}_time"
        if tk1 in all_model_times and tk2 in all_model_times:
            time_comparison["time_difference"] = abs(all_model_times[tk1] - all_model_times[tk2])

    # --- 10. Exclude GT model from final summary metrics ---
    if ground_truth_model:
        for model_id, cfg in MODEL_CONFIGS.items():
            if model_id == ground_truth_model and cfg.display in model_metric_averages:
                del model_metric_averages[cfg.display]
                break

    # Log metadata (customize the metadata_dict as needed)
    log_metadata(metadata={"fastest_model": fastest_display, "model_count": len(model_keys)})

    summary_html = create_summary_visualization(
        model_metrics=model_metric_averages,
        time_comparison=time_comparison,
        similarities=similarities,
        ground_truth_model=ground_truth_model,
    )

    # --- 11. Combine summary and per-image details ---
    final_html = f"""
    {summary_html}
    <div class="container mx-auto px-4">
      <h2 class="text-2xl font-bold my-4">Sample Results</h2>
      {image_cards_html}
    </div>
    """
    return HTMLString(final_html)
