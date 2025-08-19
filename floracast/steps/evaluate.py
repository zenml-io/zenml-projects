"""
Model evaluation step for FloraCast.
"""

from typing import Annotated, Tuple
import pandas as pd
import tempfile
import base64
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from darts import TimeSeries
from zenml import step, log_metadata
from zenml.logger import get_logger
from zenml.types import HTMLString

from utils.metrics import smape

logger = get_logger(__name__)


def create_evaluation_visualization(
    train_series: TimeSeries,
    val_series: TimeSeries,
    predictions: TimeSeries,
    actual: TimeSeries,
    score: float,
    metric: str,
) -> HTMLString:
    """
    Create an HTML visualization of the evaluation results.

    Args:
        train_series: Training time series data
        val_series: Validation time series data
        predictions: Model predictions
        actual: Actual validation values used for evaluation
        score: Evaluation score
        metric: Metric name

    Returns:
        HTMLString with embedded plot visualization
    """
    try:
        # Create the plot with modern styling
        plt.style.use("default")  # Reset to clean style
        fig, ax = plt.subplots(figsize=(14, 8))
        fig.patch.set_facecolor("white")

        # Convert to pandas for easier plotting
        train_df = train_series.pd_dataframe()
        val_df = val_series.pd_dataframe()
        pred_df = predictions.pd_dataframe()

        # Define modern color palette
        colors = {
            "train": "#E8F4FD",  # Very light blue
            "val": "#2E86AB",  # Modern blue
            "pred": "#F18F01",  # Vibrant orange
            "highlight": "#FFE66D",  # Soft yellow
        }

        # Focus zoom: show last 3 months of training + all validation + prediction period
        zoom_start = pred_df.index.min() - pd.Timedelta(days=90)

        # Filter data for zoom
        train_zoom = (
            train_df[train_df.index >= zoom_start]
            if len(train_df[train_df.index >= zoom_start]) > 0
            else train_df.tail(90)
        )

        # Plot training data (minimal context)
        ax.plot(
            train_zoom.index,
            train_zoom.iloc[:, 0],
            label="Training Data (Last 90 days)",
            color="#7FB3D3",  # More solid blue instead of very light
            alpha=0.8,  # More opaque
            linewidth=2,  # Slightly thicker
        )

        # Plot validation data
        ax.plot(
            val_df.index,
            val_df.iloc[:, 0],
            label="Ground Truth",
            color=colors["val"],
            alpha=0.9,
            linewidth=3,
            zorder=3,
        )

        # Plot predictions with modern style
        ax.plot(
            pred_df.index,
            pred_df.iloc[:, 0],
            label="AI Predictions",
            color=colors["pred"],
            alpha=0.95,
            linewidth=4,
            linestyle="-",
            zorder=4,
            marker="o",
            markersize=3,
            markeredgewidth=0,
        )

        # Add subtle shaded region
        ax.axvspan(
            pred_df.index.min(),
            pred_df.index.max(),
            alpha=0.08,
            color=colors["highlight"],
            zorder=1,
        )

        # Focus the x-axis on the interesting period
        ax.set_xlim(zoom_start, val_df.index.max())

        # Modern title styling
        performance_text = (
            "Excellent"
            if score < 20
            else "Good"
            if score < 40
            else "Needs Improvement"
        )

        ax.set_title(
            f"FloraCast AI Forecasting Model\\n{metric.upper()}: {score:.2f} ({performance_text})",
            fontsize=18,
            fontweight="600",
            color="#2C3E50",
            pad=20,
        )

        # Modern axis styling
        ax.set_xlabel("Date", fontsize=14, color="#34495E", fontweight="500")
        ax.set_ylabel(
            "Normalized Value", fontsize=14, color="#34495E", fontweight="500"
        )

        # Clean legend
        legend = ax.legend(
            fontsize=12,
            frameon=True,
            fancybox=True,
            shadow=True,
            framealpha=0.95,
            edgecolor="none",
        )
        legend.get_frame().set_facecolor("#FAFAFA")

        # Modern grid
        ax.grid(True, alpha=0.2, linestyle="-", linewidth=0.5, color="#BDC3C7")
        ax.set_facecolor("#FEFEFE")

        # Clean up axes
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#BDC3C7")
        ax.spines["bottom"].set_color("#BDC3C7")

        # Better tick formatting
        plt.xticks(rotation=45, ha="right", fontsize=11, color="#34495E")
        plt.yticks(fontsize=11, color="#34495E")

        plt.tight_layout(pad=2.0)

        # Save plot to base64 string
        with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
            fig.savefig(tmp.name, dpi=150, bbox_inches="tight")
            plt.close(fig)

            with open(tmp.name, "rb") as f:
                img_data = f.read()
                img_base64 = base64.b64encode(img_data).decode("utf-8")

        # Create HTML with embedded image
        html_content = f"""
        <html>
        <head>
            <title>Model Evaluation Results</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ text-align: center; margin-bottom: 20px; }}
                .metrics {{ background-color: #f5f5f5; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
                .plot {{ text-align: center; }}
                img {{ max-width: 100%; height: auto; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>FloraCast Model Evaluation</h1>
            </div>
            
            <div class="metrics">
                <h2>Evaluation Metrics</h2>
                <p><strong>Metric:</strong> {metric.upper()}</p>
                <p><strong>Score:</strong> {score:.4f}</p>
                <p><strong>Prediction Horizon:</strong> {len(predictions)} time steps</p>
                <p><strong>Training Data Points:</strong> {len(train_series)}</p>
                <p><strong>Validation Data Points:</strong> {len(val_series)}</p>
            </div>
            
            <div class="plot">
                <h2>Time Series Visualization</h2>
                <img src="data:image/png;base64,{img_base64}" alt="Evaluation Plot">
            </div>
            
            <div class="footer">
                <p><em>Generated by FloraCast evaluation step</em></p>
            </div>
        </body>
        </html>
        """

        return HTMLString(html_content)

    except Exception as e:
        logger.error(f"Failed to create visualization: {str(e)}")
        # Return simple HTML with error message
        error_html = f"""
        <html>
        <body>
            <h1>Evaluation Results</h1>
            <p><strong>Metric:</strong> {metric.upper()}</p>
            <p><strong>Score:</strong> {score:.4f}</p>
            <p><em>Visualization failed to generate: {str(e)}</em></p>
        </body>
        </html>
        """
        return HTMLString(error_html)


@step
def evaluate(
    model: object,
    train_series: TimeSeries,
    val_series: TimeSeries,
    horizon: int = 7,
    metric: str = "smape",
) -> Tuple[
    Annotated[float, "evaluation_score"],
    Annotated[HTMLString, "evaluation_visualization"],
]:
    """
    Evaluate the trained model on validation data.

    Args:
        model: Trained forecasting model
        train_series: Training time series
        val_series: Validation time series
        horizon: Forecasting horizon
        metric: Evaluation metric name

    Returns:
        Evaluation metric score (lower is better for SMAPE)
    """

    logger.info(f"Evaluating with horizon: {horizon}, metric: {metric}")

    try:
        # Generate predictions using TFT model
        # TFT requires the series parameter to generate predictions
        logger.info(f"Generating predictions for horizon: {horizon}")

        # For TFT models, we need to provide the series parameter
        if hasattr(model, "predict"):
            # Generate predictions using iterative multi-step approach for longer horizons
            # This is better than single-shot prediction for long horizons
            n_predict = min(
                len(val_series), 42
            )  # Cap at 6 weeks for reasonable evaluation

            # Use multiple prediction steps for better long-term accuracy
            predictions_list = []
            context_series = train_series

            # Predict in chunks of output_chunk_length (14 days)
            remaining_steps = n_predict
            while remaining_steps > 0:
                chunk_size = min(
                    14, remaining_steps
                )  # Model's output_chunk_length
                chunk_pred = model.predict(n=chunk_size, series=context_series)
                predictions_list.append(chunk_pred)

                # Extend context with the prediction for next iteration
                context_series = context_series.concatenate(chunk_pred)
                remaining_steps -= chunk_size

            # Combine all predictions
            if len(predictions_list) == 1:
                predictions = predictions_list[0]
            else:
                predictions = predictions_list[0]
                for pred_chunk in predictions_list[1:]:
                    predictions = predictions.concatenate(pred_chunk)
            logger.info(f"Generated {len(predictions)} predictions")

            # Truncate validation series to match prediction length
            actual = val_series[: len(predictions)]

            # Use original predictions for evaluation (no artificial perturbation)
            predictions_for_eval = predictions

            # Calculate metric on predictions
            if metric == "smape":
                score = smape(actual, predictions_for_eval)
            else:
                raise ValueError(f"Unknown metric: {metric}")

            logger.info(
                f"Evaluation {metric}: {score:.4f} over {len(predictions)} days"
            )

            # Log metadata to ZenML for observability
            log_metadata(
                {
                    "evaluation_metric": metric,
                    "score": float(score),
                    "horizon": horizon,
                    "num_predictions": len(predictions),
                    "actual_length": len(actual),
                    "model_type": type(model).__name__,
                }
            )

            # Create visualization
            visualization_html = create_evaluation_visualization(
                train_series, val_series, predictions, actual, score, metric
            )

            return float(score), visualization_html
        else:
            logger.error("Model does not have predict method")
            empty_viz = HTMLString(
                "<div><p>Model evaluation failed - no predict method available</p></div>"
            )
            return 9999.0, empty_viz

    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        logger.info("This might be due to TFT model prediction requirements")
        # Return a high penalty score for failed evaluation
        empty_viz = HTMLString(
            "<div><p>Evaluation failed due to exception</p></div>"
        )
        return 9999.0, empty_viz
