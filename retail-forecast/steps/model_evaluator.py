import base64
from io import BytesIO
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from prophet import Prophet
from zenml import log_metadata, step


@step
def evaluate_models(
    models: Dict[str, Prophet],
    test_data_dict: Dict[str, pd.DataFrame],
    series_ids: List[str],
    forecast_horizon: int = 7,
) -> Dict[str, float]:
    """
    Evaluate Prophet models on test data and log metrics.

    Args:
        models: Dictionary of trained Prophet models
        test_data_dict: Dictionary of test data for each series
        series_ids: List of series identifiers
        forecast_horizon: Number of future time periods to forecast

    Returns:
        Dictionary of average metrics across all series
    """
    # Initialize metrics storage
    all_metrics = {"mae": [], "rmse": [], "mape": []}

    series_metrics = {}

    # Create a figure for plotting forecasts
    plt.figure(figsize=(12, len(series_ids) * 4))

    for i, series_id in enumerate(series_ids):
        print(f"Evaluating model for {series_id}...")
        model = models[series_id]
        test_data = test_data_dict[series_id]

        # Create future dataframe for the test period
        future = model.make_future_dataframe(periods=forecast_horizon)

        # Generate forecast
        forecast = model.predict(future)

        # Extract actual and predicted values for the test period
        # Merge forecasts with actuals based on date
        merged_data = pd.merge(
            test_data,
            forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]],
            on="ds",
            how="left",
        )

        # Calculate metrics
        actuals = merged_data["y"].values
        predictions = merged_data["yhat"].values

        mae = np.mean(np.abs(actuals - predictions))
        rmse = np.sqrt(np.mean((actuals - predictions) ** 2))

        # Handle zeros in actuals for MAPE calculation
        mask = actuals != 0
        if np.any(mask):
            mape = (
                np.mean(
                    np.abs((actuals[mask] - predictions[mask]) / actuals[mask])
                )
                * 100
            )
        else:
            mape = np.nan

        # Store metrics
        series_metrics[series_id] = {"mae": mae, "rmse": rmse, "mape": mape}

        all_metrics["mae"].append(mae)
        all_metrics["rmse"].append(rmse)
        if not np.isnan(mape):
            all_metrics["mape"].append(mape)

        # Plot the forecast vs actual for this series
        plt.subplot(len(series_ids), 1, i + 1)
        plt.plot(merged_data["ds"], merged_data["y"], "b.", label="Actual")
        plt.plot(
            merged_data["ds"], merged_data["yhat"], "r-", label="Forecast"
        )
        plt.fill_between(
            merged_data["ds"],
            merged_data["yhat_lower"],
            merged_data["yhat_upper"],
            color="gray",
            alpha=0.2,
        )
        plt.title(f"Forecast vs Actual for {series_id}")
        plt.legend()

    # Calculate average metrics across all series
    average_metrics = {
        "avg_mae": np.mean(all_metrics["mae"]),
        "avg_rmse": np.mean(all_metrics["rmse"]),
        "avg_mape": np.mean(all_metrics["mape"])
        if all_metrics["mape"]
        else np.nan,
    }

    # Save plot to buffer
    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plot_data = base64.b64encode(buf.read()).decode("utf-8")
    plt.close()

    # Log metrics to ZenML
    log_metadata(
        metadata={
            "avg_mae": average_metrics["avg_mae"],
            "avg_rmse": average_metrics["avg_rmse"],
            "avg_mape": average_metrics["avg_mape"]
            if not np.isnan(average_metrics["avg_mape"])
            else 0.0,
            "forecast_plot": plot_data,
        }
    )

    print(f"Average MAE: {average_metrics['avg_mae']:.2f}")
    print(f"Average RMSE: {average_metrics['avg_rmse']:.2f}")
    print(
        f"Average MAPE: {average_metrics['avg_mape']:.2f}%"
        if not np.isnan(average_metrics["avg_mape"])
        else "Average MAPE: N/A"
    )

    return average_metrics
