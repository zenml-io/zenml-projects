import base64
import logging
from datetime import timedelta
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from prophet import Prophet
from typing_extensions import Annotated
from zenml import log_metadata, step
from zenml.types import HTMLString

logger = logging.getLogger(__name__)


@step
def make_predictions(
    model: Optional[Prophet],
    training_dataset: Optional[pd.DataFrame],
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
    """Generate predictions for future periods using the trained model.

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
        logger.info("Using naive forecasting method (last value)")

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
            historical = test_data[
                test_data["series_id"] == series_id
            ].sort_values("date")
            forecast = forecast_df[
                forecast_df["series_id"] == series_id
            ].sort_values("date")

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

        logger.info(
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
        historical = test_data[
            test_data["series_id"] == series_id
        ].sort_values("date")
        # Get forecast data
        forecast = forecast_df[
            forecast_df["series_id"] == series_id
        ].sort_values("date")

        plt.subplot(3, 1, i + 1)
        # Plot historical
        plt.plot(
            historical["date"], historical["sales"], "b-", label="Historical"
        )
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

    logger.info(
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
    """Generate a naive forecast that uses the last known value for each series.
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
    method_name = (
        "Temporal Fusion Transformer" if method == "tft" else "Naive Forecast"
    )

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


@step
def generate_forecasts(
    models: Dict[str, Prophet],
    train_data_dict: Dict[str, pd.DataFrame],
    series_ids: List[str],
    forecast_periods: int = 30,
) -> Tuple[
    Annotated[Dict[str, pd.DataFrame], "forecasts_by_series"],
    Annotated[pd.DataFrame, "combined_forecast"],
    Annotated[HTMLString, "forecast_dashboard"],
]:
    """Generate future forecasts using trained Prophet models.

    Args:
        models: Dictionary of trained Prophet models
        train_data_dict: Dictionary of training data for each series
        series_ids: List of series identifiers
        forecast_periods: Number of periods to forecast into the future

    Returns:
        forecasts_by_series: Dictionary of forecast dataframes for each series
        combined_forecast: Combined dataframe with all series forecasts
        forecast_dashboard: HTML dashboard with forecast visualizations
    """
    forecasts = {}

    # Create a plot to visualize all forecasts
    plt.figure(figsize=(12, len(series_ids) * 4))

    for i, series_id in enumerate(series_ids):
        logger.info(f"Generating forecast for {series_id}...")
        model = models[series_id]

        # Get last date from training data
        last_date = train_data_dict[series_id]["ds"].max()

        # Create future dataframe
        future = model.make_future_dataframe(periods=forecast_periods)

        # Generate forecast
        forecast = model.predict(future)

        # Store forecast
        forecasts[series_id] = forecast

        # Plot the forecast
        plt.subplot(len(series_ids), 1, i + 1)

        # Plot training data
        train_data = train_data_dict[series_id]
        plt.plot(train_data["ds"], train_data["y"], "b.", label="Historical")

        # Plot forecast
        plt.plot(forecast["ds"], forecast["yhat"], "r-", label="Forecast")
        plt.fill_between(
            forecast["ds"],
            forecast["yhat_lower"],
            forecast["yhat_upper"],
            color="gray",
            alpha=0.2,
        )

        # Add a vertical line at the forecast start
        plt.axvline(x=last_date, color="k", linestyle="--")

        plt.title(f"Forecast for {series_id}")
        plt.legend()

    # Save plot to buffer
    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plot_data = base64.b64encode(buf.read()).decode("utf-8")
    plt.close()

    # Create a combined forecast dataframe for all series
    combined_forecast = []
    for series_id, forecast in forecasts.items():
        # Add series_id column
        forecast_with_id = forecast.copy()
        forecast_with_id["series_id"] = series_id

        # Extract store and item from series_id
        store, item = series_id.split("-")
        forecast_with_id["store"] = store
        forecast_with_id["item"] = item

        combined_forecast.append(forecast_with_id)

    combined_df = pd.concat(combined_forecast)

    # Log basic metadata (not the large plot)
    log_metadata(
        metadata={
            "forecast_horizon": forecast_periods,
            "num_series": len(series_ids),
        }
    )

    # Create HTML dashboard
    forecast_dashboard = create_forecast_dashboard(
        forecasts, series_ids, train_data_dict, plot_data, forecast_periods
    )

    logger.info(
        f"Generated forecasts for {len(forecasts)} series, {forecast_periods} periods ahead"
    )

    return forecasts, combined_df, forecast_dashboard


def create_forecast_dashboard(
    forecasts, series_ids, train_data_dict, plot_image_data, forecast_horizon
):
    """Create an HTML dashboard for forecast visualization."""
    # Generate forecast metrics
    series_stats = []
    for series_id in series_ids:
        forecast = forecasts[series_id]
        future_period = forecast.iloc[-forecast_horizon:]

        # Extract store and item
        store, item = series_id.split("-")

        # Get statistics
        avg_forecast = future_period["yhat"].mean()
        min_forecast = future_period["yhat"].min()
        max_forecast = future_period["yhat"].max()

        # Get growth rate compared to historical
        historical = train_data_dict[series_id]["y"].mean()
        growth = (
            ((avg_forecast / historical) - 1) * 100 if historical > 0 else 0
        )

        series_stats.append(
            {
                "series_id": series_id,
                "store": store,
                "item": item,
                "avg_forecast": avg_forecast,
                "min_forecast": min_forecast,
                "max_forecast": max_forecast,
                "growth": growth,
            }
        )

    # Create table rows for series statistics
    series_rows = ""
    for stat in series_stats:
        growth_class = (
            "text-green-600 font-bold"
            if stat["growth"] >= 0
            else "text-red-600 font-bold"
        )
        growth_sign = "+" if stat["growth"] >= 0 else ""

        series_rows += f"""
        <tr class="border-b hover:bg-gray-50">
            <td class="py-3 px-4">{stat["series_id"]}</td>
            <td class="py-3 px-4">{stat["store"]}</td>
            <td class="py-3 px-4">{stat["item"]}</td>
            <td class="py-3 px-4 text-right">{stat["avg_forecast"]:.1f}</td>
            <td class="py-3 px-4 text-right">{stat["min_forecast"]:.1f}</td>
            <td class="py-3 px-4 text-right">{stat["max_forecast"]:.1f}</td>
            <td class="py-3 px-4 text-right {growth_class}">{growth_sign}{stat["growth"]:.1f}%</td>
        </tr>
        """

    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Retail Sales Forecast Dashboard</title>
        <style>
            body {{ font-family: 'Segoe UI', Arial, sans-serif; line-height: 1.6; color: #333; max-width: 1200px; margin: 0 auto; padding: 20px; }}
            h1, h2 {{ color: #2c3e50; }}
            .header {{ background-color: #f8f9fa; padding: 20px; border-radius: 8px; margin-bottom: 30px; border-left: 5px solid #3498db; }}
            .summary {{ display: flex; gap: 20px; margin-bottom: 30px; }}
            .summary-card {{ background: white; border-radius: 8px; padding: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); flex: 1; text-align: center; }}
            .summary-value {{ font-size: 24px; font-weight: bold; color: #3498db; margin: 10px 0; }}
            .table-container {{ margin: 30px 0; box-shadow: 0 4px 6px rgba(0,0,0,0.1); border-radius: 8px; overflow: hidden; }}
            table {{ width: 100%; border-collapse: collapse; }}
            th {{ background: #eef1f5; text-align: left; padding: 12px 15px; font-weight: 600; }}
            td {{ padding: 10px 15px; }}
            tr:nth-child(even) {{ background-color: #f8f9fa; }}
            .forecast-viz {{ margin-top: 40px; text-align: center; }}
            .forecast-viz img {{ max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 4px 10px rgba(0,0,0,0.1); }}
            .text-green-600 {{ color: #0d9488; }}
            .text-red-600 {{ color: #dc2626; }}
            .font-bold {{ font-weight: bold; }}
            .border-b {{ border-bottom: 1px solid #e5e7eb; }}
            .hover\\:bg-gray-50:hover {{ background-color: #f9fafb; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Retail Sales Forecast Dashboard</h1>
            <p>Forecast horizon: {forecast_horizon} periods | Total series: {len(series_ids)}</p>
        </div>
        
        <div class="summary">
            <div class="summary-card">
                <h3>Average Forecast</h3>
                <div class="summary-value">{sum([s["avg_forecast"] for s in series_stats]) / len(series_stats):.1f}</div>
                <p>Average predicted sales across all series</p>
            </div>
            
            <div class="summary-card">
                <h3>Total Growth</h3>
                <div class="summary-value">{sum([s["growth"] for s in series_stats]) / len(series_stats):.1f}%</div>
                <p>Average growth compared to historical</p>
            </div>
            
            <div class="summary-card">
                <h3>Top Performer</h3>
                <div class="summary-value">{max(series_stats, key=lambda x: x["growth"])["series_id"]}</div>
                <p>Series with highest growth rate</p>
            </div>
        </div>
        
        <h2>Forecast by Series</h2>
        <div class="table-container">
            <table>
                <thead>
                    <tr>
                        <th>Series ID</th>
                        <th>Store</th>
                        <th>Item</th>
                        <th>Avg Forecast</th>
                        <th>Min Forecast</th>
                        <th>Max Forecast</th>
                        <th>Growth</th>
                    </tr>
                </thead>
                <tbody>
                    {series_rows}
                </tbody>
            </table>
        </div>
        
        <div class="forecast-viz">
            <h2>Forecast Visualization</h2>
            <img src="data:image/png;base64,{plot_image_data}" alt="Forecast Visualization">
        </div>
    </body>
    </html>
    """

    return HTMLString(html)
