from zenml import step, log_metadata
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from io import BytesIO
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from zenml.types import HTMLString
from typing_extensions import Annotated


@step
def make_predictions(
    model: Optional[TemporalFusionTransformer],
    training_dataset: Optional[TimeSeriesDataSet],
    test_data: pd.DataFrame,
    forecast_horizon: int = 14,
) -> Tuple[
    Annotated[Dict[str, Any], "forecast_data"],
    Annotated[bytes, "forecast_plot"],
    Annotated[Dict[str, Any], "sample_forecast"],
    Annotated[int, "forecast_horizon"],
    Annotated[str, "method"],
    Annotated[HTMLString, "forecast_visualization"],
]:
    """
    Generate predictions for future periods using the trained model.
    
    Args:
        model: Trained TFT model or None for naive forecast
        training_dataset: Training dataset used for the model or None for naive forecast
        test_data: Test dataframe with historical data
        forecast_horizon: Number of days to forecast into the future
    
    Returns:
        forecast_data: Dictionary containing forecast data
        forecast_plot: Bytes of the forecast plot image
        sample_forecast: Dictionary with sample forecasts
        forecast_horizon: Number of days in the forecast
        method: Name of the forecasting method used
        forecast_visualization: HTML visualization of forecast results
    """
    # Handle case where no model or training dataset are passed (predict-only mode)
    if model is None or training_dataset is None:
        print("Using naive forecasting method (last value)")
        
        # Create a naive model that predicts the last known value for each series
        forecast_df = naive_forecast(test_data, forecast_horizon)

        # Create a simple plot
        plt.figure(figsize=(15, 10))
        sample_series = np.random.choice(
            forecast_df["series_id"].unique(),
            size=min(3, len(forecast_df["series_id"].unique())),
            replace=False,
        )

        for i, series_id in enumerate(sample_series):
            historical = test_data[test_data["series_id"] == series_id].sort_values("date")
            forecast = forecast_df[forecast_df["series_id"] == series_id].sort_values("date")

            plt.subplot(3, 1, i + 1)
            plt.plot(
                historical["date"],
                historical["sales"],
                "b-",
                label="Historical",
            )
            plt.plot(
                forecast["date"],
                forecast["sales_prediction"],
                "r-",
                label="Naive Forecast",
            )
            plt.title(f"Series: {series_id} (Naive Forecast)")
            plt.legend()
            plt.grid(True)

        plt.tight_layout()
        forecast_plot_buffer = BytesIO()
        plt.savefig(forecast_plot_buffer, format="png")
        plt.close()
        forecast_plot_bytes = forecast_plot_buffer.getvalue()

        print(
            f"Generated naive forecasts for {len(test_data['series_id'].unique())} series, {forecast_horizon} days ahead"
        )

        # Create HTML visualization
        html_visualization = create_forecast_visualization(
            forecast_df,
            test_data,
            sample_series,
            forecast_horizon,
            method="naive",
        )

        # Log metadata about artifacts
        log_metadata(
            metadata={
                "forecast_data_artifact_name": "forecast_data",
                "forecast_data_artifact_type": "Dict[str, Any]",
                "visualization_artifact_name": "forecast_visualization",
                "visualization_artifact_type": "zenml.types.HTMLString",
                "forecast_method": "naive",
                "forecast_horizon": forecast_horizon,
            },
        )

        # Get sample forecasts
        sample_records = get_sample_forecasts(forecast_df)

        # Return naive forecast results
        return (
            forecast_df.to_dict(),
            forecast_plot_bytes,
            sample_records,
            forecast_horizon,
            "naive",
            html_visualization,
        )

    # Select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Generate future dates for forecasting
    last_date = pd.to_datetime(test_data["date"].max())
    future_dates = pd.date_range(
        start=last_date + timedelta(days=1), periods=forecast_horizon, freq="D"
    )

    # Create empty list for results
    forecasts = []

    # Get unique store-item combinations
    series_ids = test_data["series_id"].unique()

    # For each series (store-item combination), make a prediction
    for series_id in series_ids:
        # Get the series' data
        series_data = test_data[test_data["series_id"] == series_id]

        # Get the most recent data for this series
        max_date = series_data["date"].max()
        recent_data = series_data[series_data["date"] == max_date]

        # Extract store and item
        store = recent_data["store"].iloc[0]
        item = recent_data["item"].iloc[0]

        # Create the future dataframe with known features
        future_df = pd.DataFrame({"date": future_dates})
        future_df["store"] = store
        future_df["item"] = item
        future_df["series_id"] = series_id

        # Add calendar features
        future_df["day_of_week"] = future_df["date"].dt.dayofweek
        future_df["day_of_month"] = future_df["date"].dt.day
        future_df["month"] = future_df["date"].dt.month
        future_df["year"] = future_df["date"].dt.year
        future_df["week_of_year"] = future_df["date"].dt.isocalendar().week

        # Simulate known features
        future_df["is_weekend"] = (future_df["day_of_week"] >= 5).astype(int)
        future_df["is_holiday"] = (
            (future_df["day_of_month"] == 1)
            | (future_df["day_of_month"] == 15)
        ).astype(int)
        future_df["is_promo"] = (
            (future_df["day_of_month"] >= 10)
            & (future_df["day_of_month"] <= 20)
        ).astype(int)

        # Add encoded columns
        for col in ["store", "item", "series_id", "day_of_week", "month"]:
            # Get mapping from the test data
            mapping = {
                val: idx for idx, val in enumerate(test_data[col].unique())
            }
            future_df[f"{col}_encoded"] = future_df[col].map(mapping)

        # Add time index
        max_time_idx = test_data["time_idx"].max()
        future_df["time_idx"] = range(
            max_time_idx + 1, max_time_idx + 1 + len(future_df)
        )

        # Ensure all needed columns exist
        for col in test_data.columns:
            if col not in future_df.columns and col != "sales":
                # Try to get same value as the most recent data
                if col in recent_data.columns:
                    future_df[col] = recent_data[col].iloc[0]
                else:
                    # Default to 0 for missing features
                    future_df[col] = 0

        # Prepare dataset for prediction
        future_dataset = training_dataset.from_dataset(
            training_dataset, future_df, predict=True
        )
        future_dataloader = future_dataset.to_dataloader(
            train=False, batch_size=128
        )

        # Generate predictions
        predictions, _ = model.predict(
            future_dataloader,
            return_x=True,
            trainer_kwargs={"accelerator": device},
        )

        # Add predictions to the future dataframe
        future_df["sales_prediction"] = predictions.flatten().cpu().numpy()

        # Append to results
        forecasts.append(future_df)

    # Combine all forecasts
    forecast_df = pd.concat(forecasts, ignore_index=True)

    # Create plots for visualization
    plt.figure(figsize=(15, 10))

    # Get a few random series to plot
    sample_series = np.random.choice(
        forecast_df["series_id"].unique(),
        size=min(3, len(forecast_df["series_id"].unique())),
        replace=False,
    )

    for i, series_id in enumerate(sample_series):
        # Get historical data
        historical = test_data[test_data["series_id"] == series_id].sort_values("date")
        # Get forecast data
        forecast = forecast_df[forecast_df["series_id"] == series_id].sort_values("date")

        plt.subplot(3, 1, i + 1)
        # Plot historical
        plt.plot(historical["date"], historical["sales"], "b-", label="Historical")
        # Plot forecast
        plt.plot(
            forecast["date"],
            forecast["sales_prediction"],
            "r-",
            label="Forecast",
        )
        plt.title(f"Series: {series_id}")
        plt.legend()
        plt.grid(True)

    plt.tight_layout()

    # Capture the plot as bytes
    forecast_plot_buffer = BytesIO()
    plt.savefig(forecast_plot_buffer, format="png")
    plt.close()
    forecast_plot_bytes = forecast_plot_buffer.getvalue()

    print(
        f"Generated forecasts for {len(series_ids)} series, {forecast_horizon} days ahead"
    )

    # Create sample forecasts
    sample_records = get_sample_forecasts(forecast_df)

    # Create HTML visualization
    html_visualization = create_forecast_visualization(
        forecast_df, test_data, sample_series, forecast_horizon, method="tft"
    )

    # Log metadata about artifacts
    log_metadata(
        metadata={
            "forecast_data_artifact_name": "forecast_data",
            "forecast_data_artifact_type": "Dict[str, Any]",
            "visualization_artifact_name": "forecast_visualization",
            "visualization_artifact_type": "zenml.types.HTMLString",
            "forecast_method": "tft",
            "forecast_horizon": forecast_horizon,
        },
    )

    # Return forecasts as artifacts
    return (
        forecast_df.to_dict(),
        forecast_plot_bytes,
        sample_records,
        forecast_horizon,
        "tft",
        html_visualization,
    )


def get_sample_forecasts(forecast_df: pd.DataFrame) -> dict:
    """Extract sample forecasts for each series."""
    sample_records = {}
    series_ids_list = []
    dates = []
    predictions = []
    
    # Group by series_id and get first record from each group
    for series_id in forecast_df["series_id"].unique():
        series_data = forecast_df[forecast_df["series_id"] == series_id]
        first_row = series_data.iloc[0]
        series_ids_list.append(series_id)
        dates.append(first_row["date"])
        predictions.append(first_row["sales_prediction"])
    
    sample_records["series_id"] = series_ids_list
    sample_records["date"] = dates
    sample_records["sales_prediction"] = predictions
    
    return sample_records


def naive_forecast(
    test_df: pd.DataFrame, forecast_horizon: int
) -> pd.DataFrame:
    """
    Generate a naive forecast that uses the last known value for each series.
    This is used as a fallback when no model is available.
    """
    forecasts = []

    # Get unique store-item combinations
    series = test_df["series_id"].unique()

    # Get the last date in the test data
    last_date = pd.to_datetime(test_df["date"].max())

    # Generate future dates
    future_dates = pd.date_range(
        start=last_date + timedelta(days=1), periods=forecast_horizon, freq="D"
    )

    for series_id in series:
        # Get the series' data
        series_data = test_df[test_df["series_id"] == series_id]

        # Get the last sales value for this series
        last_sales = series_data.iloc[-1]["sales"]

        # Get store and item
        store = series_data.iloc[0]["store"]
        item = series_data.iloc[0]["item"]

        # Create future data with the last sales value
        future_df = pd.DataFrame({"date": future_dates})
        future_df["store"] = store
        future_df["item"] = item
        future_df["series_id"] = series_id
        future_df["sales_prediction"] = last_sales

        # Add to forecasts
        forecasts.append(future_df)

    # Combine all forecasts
    forecast_df = pd.concat(forecasts, ignore_index=True)
    return forecast_df


def create_forecast_visualization(
    forecast_df: pd.DataFrame,
    historical_df: pd.DataFrame,
    sample_series: list,
    forecast_horizon: int,
    method: str = "tft",
) -> HTMLString:
    """Create an HTML visualization of forecasting results."""
    # Create a simpler visualization with just the key information
    method_name = "Temporal Fusion Transformer" if method == "tft" else "Naive Forecast"
    
    # Get forecast start and end dates
    forecast_start = forecast_df["date"].min()
    forecast_end = forecast_df["date"].max()

    # Calculate total forecasted sales
    total_forecast = forecast_df["sales_prediction"].sum()
    
    # Create HTML
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Retail Sales Forecast</title>
        <script src="https://cdn.tailwindcss.com"></script>
    </head>
    <body class="bg-gray-50 p-4">
        <div class="container mx-auto">
            <h1 class="text-2xl font-bold mb-2">Retail Sales Forecast</h1>
            <p class="mb-4">Method: {method_name} | Period: {forecast_start} to {forecast_end}</p>
            
            <div class="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                <div class="bg-blue-50 p-4 rounded shadow border border-blue-200">
                    <div class="text-blue-700 text-sm font-medium">Forecast Horizon</div>
                    <div class="text-2xl font-bold">{forecast_horizon} days</div>
                </div>
                <div class="bg-green-50 p-4 rounded shadow border border-green-200">
                    <div class="text-green-700 text-sm font-medium">Series Count</div>
                    <div class="text-2xl font-bold">{len(forecast_df["series_id"].unique())}</div>
                </div>
                <div class="bg-purple-50 p-4 rounded shadow border border-purple-200">
                    <div class="text-purple-700 text-sm font-medium">Total Sales Forecast</div>
                    <div class="text-2xl font-bold">{total_forecast:.0f}</div>
                </div>
            </div>
        </div>
    </body>
    </html>
    """

    return HTMLString(html)
