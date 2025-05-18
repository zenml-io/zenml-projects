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

from typing import Dict

import polars as pl
from typing_extensions import Annotated
from zenml import log_metadata, step
from zenml.types import HTMLString

from utils import (
    calculate_custom_metrics,
    calculate_model_similarities,
    compare_multi_model,
    create_model_comparison_card,
    create_summary_visualization,
    get_model_info,
)


@step(enable_cache=False)
def evaluate_models(
    model_results: pl.DataFrame,
    ground_truth_df: pl.DataFrame,
) -> Annotated[HTMLString, "ocr_visualization"]:
    """Compare the performance of multiple configurable models with visualization.

    Args:
        model_results: Dictionary containing single or multiple model ocr results
        ground_truth_df: DataFrame containing ground truth texts

    Returns:
        HTML visualization of the evaluation results
    """
    if model_results is None or model_results.is_empty():
        raise ValueError("Model results are required for evaluation")

    if ground_truth_df is None or ground_truth_df.is_empty():
        raise ValueError("Ground truth data is required for evaluation")

    gt_df = ground_truth_df

    # --- 1. Extract unique model names from the flat DataFrame structure ---
    model_keys = model_results["model_name"].unique().to_list()
    if not model_keys:
        raise ValueError("No model names found in model_results")

    # --- 2. Build model info for evaluation models ---
    model_info = {}
    model_displays = []
    model_prefixes = {}
    for model_name in model_keys:
        display, prefix = get_model_info(model_name)
        model_info[model_name] = (display, prefix)
        model_displays.append(display)
        model_prefixes[display] = prefix

    # --- 3. Split model results by model ---
    model_results_dict = {}
    for model_name in model_keys:
        model_data = model_results.filter(pl.col("model_name") == model_name)
        model_results_dict[model_name] = model_data

    # --- 4. Merge evaluation models' results ---
    base_model = model_keys[0]
    base_display, base_prefix = model_info[base_model]
    merged_results = model_results_dict[base_model].clone()
    for i, model_name in enumerate(model_keys[1:], start=1):
        disp, pref = model_info[model_name]
        suffix = f"_{pref}" if i > 1 else "_right"
        merged_results = merged_results.join(
            model_results_dict[model_name],
            on=["id", "image_name"],
            how="inner",
            suffix=suffix,
        )

    # --- 5. Join ground truth data if available ---
    if gt_df is not None:
        merged_results = merged_results.join(
            gt_df, on=["id", "image_name"], how="inner", suffix="_gt"
        )

    # --- 6. Calculate processing times for evaluation models ---
    all_model_times = {}
    for model_name in model_keys:
        model_df = model_results.filter(pl.col("model_name") == model_name)
        disp, pref = model_info[model_name]
        time_key = f"avg_{pref}_time"
        all_model_times[time_key] = (
            model_df.select("processing_time").to_series().mean()
        )
        all_model_times[f"{pref}_display"] = disp

    fastest_model_time, fastest_key = min(
        [
            (time, key)
            for key, time in all_model_times.items()
            if not key.endswith("_display")
        ],
        key=lambda x: x[0],
    )
    fastest_prefix = fastest_key.replace("avg_", "").replace("_time", "")
    fastest_display = all_model_times.get(
        f"{fastest_prefix}_display", fastest_prefix
    )

    # --- 7. Per-image evaluation: compute metrics, error analysis, and build per-image cards ---
    evaluation_metrics = []
    image_cards_html = ""
    gt_text_col = "ground_truth_text"

    # Check if we have ground truth data in our joined dataset
    if (
        gt_text_col not in merged_results.columns
        and "raw_text_gt" in merged_results.columns
    ):
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
        row_metrics = calculate_custom_metrics(
            ground_truth, model_texts, list(model_texts.keys())
        )
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
                error_analysis[disp]["GT Similarity"] = row_metrics[disp][
                    "GT Similarity"
                ]

        comparison_card = create_model_comparison_card(
            image_name=row["image_name"],
            ground_truth=ground_truth,
            model_texts=model_texts,
            model_metrics=error_analysis,
        )
        image_cards_html += comparison_card

    # --- 8. Compute average metrics for evaluation models ---
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
                model_metric_averages[disp]["GT Similarity"] = df_eval[
                    sim_col
                ].mean()
    for disp in model_displays:
        pref = model_prefixes[disp]
        tkey = f"avg_{pref}_time"
        if tkey in all_model_times:
            model_metric_averages[disp]["Proc. Time"] = all_model_times[tkey]

    # --- 9. Calculate similarity matrix for evaluation models only ---
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
                    f"raw_text_{disp.lower().replace(' ', '_')}": texts_map[
                        disp
                    ]
                    for disp in model_displays
                }
                for texts_map in sim_results
            ],
            model_displays=model_displays,
        )

    # --- 10. Build time comparison info ---
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
            time_comparison["time_difference"] = abs(
                all_model_times[tk1] - all_model_times[tk2]
            )

    # Log metadata (customize the metadata_dict as needed)
    log_metadata(
        metadata={
            "fastest_model": fastest_display,
            "model_count": len(model_keys),
        }
    )

    summary_html = create_summary_visualization(
        model_metrics=model_metric_averages,
        time_comparison=time_comparison,
        similarities=similarities,
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
