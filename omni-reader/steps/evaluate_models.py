"""This module contains the steps for evaluating the OCR models."""

import tempfile
from typing import Any, Dict

import mlflow
import polars as pl
import textdistance
from zenml import step
from zenml.integrations.mlflow.flavors.mlflow_experiment_tracker_flavor import (
    MLFlowExperimentTrackerSettings,
)

from utils import (
    MLFLOW_SETTINGS,
    compare_results,
)


@step(experiment_tracker=MLFlowExperimentTrackerSettings(MLFLOW_SETTINGS))
def evaluate_models(
    gemma_results: pl.DataFrame,
    mistral_results: pl.DataFrame,
    ground_truth: pl.DataFrame = None,
) -> Dict[str, Any]:
    """Compare the performance of Gemma 3 and MistralAI."""
    results = gemma_results.join(mistral_results, on="id", how="inner")

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False
    ) as f:
        results_path = f.name
        results.to_pandas().to_csv(results_path, index=False)
    mlflow.log_artifact(results_path, "result_data")

    evaluation_metrics = []

    if ground_truth is not None:
        # calculate accuracy metrics
        results = results.join(ground_truth, on="id", how="inner")

        for row in results.iter_rows(named=True):
            metrics = compare_results(
                row["ground_truth_text"],
                row["gemma_text"],
                row["mistral_text"],
            )

            # calculate text similarity between models
            similarity = textdistance.jaccard.normalized_similarity(
                row["gemma_text"], row["mistral_text"]
            )

            metrics["id"] = row["id"]
            metrics["text_similarity"] = similarity
            evaluation_metrics.append(metrics)

            # log to mlflow
            for key, value in metrics.items():
                if key != "id":  # skip non-metric fields
                    mlflow.log_metric(f"{key}_{row['id']}", value)

            comparison_text = (
                f"# Document {row['id']} Comparison\n\n"
                f"## Ground Truth\n{row['ground_truth_text']}\n\n"
                f"## Gemma 3 Output\n{row['gemma_text']}\n\n"
                f"## MistralAI Output\n{row['mistral_text']}\n\n"
                f"## Metrics\n"
                f"* Gemma CER: {metrics['Gemma CER']:.4f}\n"
                f"* Gemma WER: {metrics['Gemma WER']:.4f}\n"
                f"* Mistral CER: {metrics['Mistral CER']:.4f}\n"
                f"* Mistral WER: {metrics['Mistral WER']:.4f}\n"
                f"* Text Similarity: {similarity:.4f}\n"
            )
            mlflow.log_text(comparison_text, f"comparison_{row['id']}.md")

    # calculate summary metrics
    if evaluation_metrics:
        df_metrics = pl.DataFrame(evaluation_metrics)

        # averages
        avg_metrics = {
            "avg_gemma_cer": df_metrics["Gemma CER"].mean(),
            "avg_gemma_wer": df_metrics["Gemma WER"].mean(),
            "avg_mistral_cer": df_metrics["Mistral CER"].mean(),
            "avg_mistral_wer": df_metrics["Mistral WER"].mean(),
            "avg_text_similarity": df_metrics["text_similarity"].mean(),
        }

        # average metrics
        for key, value in avg_metrics.items():
            mlflow.log_metric(key, value)

        # create summary report
        summary_text = (
            f"# OCR Model Comparison Summary\n\n"
            f"## Overall Metrics\n"
            f"* Average Gemma CER: {avg_metrics['avg_gemma_cer']:.4f}\n"
            f"* Average Gemma WER: {avg_metrics['avg_gemma_wer']:.4f}\n"
            f"* Average Mistral CER: {avg_metrics['avg_mistral_cer']:.4f}\n"
            f"* Average Mistral WER: {avg_metrics['avg_mistral_wer']:.4f}\n"
            f"* Average Text Similarity Between Models: {avg_metrics['avg_text_similarity']:.4f}\n"
        )
        mlflow.log_text(summary_text, "summary_report.md")

        return {
            "individual_metrics": evaluation_metrics,
            "average_metrics": avg_metrics,
        }

    # if no ground truth, just return processing times comparison
    gemma_times = (
        gemma_results.select("gemma_processing_time").to_series().mean()
    )
    mistral_times = (
        mistral_results.select("mistral_processing_time").to_series().mean()
    )

    time_comparison = {
        "avg_gemma_time": gemma_times,
        "avg_mistral_time": mistral_times,
        "time_difference": abs(gemma_times - mistral_times),
        "faster_model": "Gemma" if gemma_times < mistral_times else "Mistral",
    }

    # time comparison metrics
    for key, value in time_comparison.items():
        if isinstance(value, (int, float)):
            mlflow.log_metric(key, value)

    # time comparison report
    time_report = (
        f"# Model Processing Time Comparison\n\n"
        f"* Average Gemma Processing Time: {time_comparison['avg_gemma_time']:.4f} seconds\n"
        f"* Average Mistral Processing Time: {time_comparison['avg_mistral_time']:.4f} seconds\n"
        f"* Time Difference: {time_comparison['time_difference']:.4f} seconds\n"
        f"* Faster Model: {time_comparison['faster_model']}\n"
    )
    mlflow.log_text(time_report, "time_comparison.md")

    return {"results": results.to_dicts(), "time_comparison": time_comparison}
