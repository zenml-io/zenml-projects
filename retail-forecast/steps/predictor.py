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
    model_artifacts: Optional[Dict[str, Any]],
    processed_data: dict,
    forecast_horizon: int = 14,  # Default to 14 days future forecast
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
    This step will:
    1. Load the TFT model
    2. Create future dates and features
    3. Forecast sales for each store-item combination
    4. Return and visualize the predictions

    The step also generates an HTML visualization for the ZenML dashboard
    
    Returns:
        Tuple containing:
            - forecast_data: Dictionary containing forecast data
            - forecast_plot: Bytes of the forecast plot image
            - sample_forecast: Dictionary with sample forecasts
            - forecast_horizon: Number of days in the forecast
            - method: Name of the forecasting method used
            - forecast_visualization: HTML visualization of forecast results
    """
    # Handle case where no model artifacts are passed (predict-only mode)
    # In a real application, you would fetch from a model registry
    if model_artifacts is None:
        print(
            "No model artifacts provided. Using a simple baseline model instead."
        )
        # Create a simple baseline model (e.g., last value or average)
        # For demonstration purposes we'll just use a naive forecast (last value)

        test_df = processed_data["test"]

        # Create a naive model that predicts the last known value for each series
        forecast_df = naive_forecast(test_df, forecast_horizon)

        # Create a simple plot
        plt.figure(figsize=(15, 10))
        sample_series = np.random.choice(
            forecast_df["series_id"].unique(),
            size=min(3, len(forecast_df["series_id"].unique())),
            replace=False,
        )

        for i, series_id in enumerate(sample_series):
            historical = test_df[
                test_df["series_id"] == series_id
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

        print(
            f"Generated naive forecasts for {len(test_df['series_id'].unique())} series, {forecast_horizon} days ahead"
        )

        # Create HTML visualization
        html_visualization = create_forecast_visualization(
            forecast_df,
            test_df,
            sample_series,
            forecast_horizon,
            method="naive",
        )

        # Log metadata about artifacts
        log_metadata(
            metadata={
                "forecast_data_artifact_name": "forecast_data",
                "forecast_data_artifact_type": "Dict",
                "visualization_artifact_name": "forecast_visualization",
                "visualization_artifact_type": "zenml.types.HTMLString",
                "forecast_method": "naive",
                "forecast_horizon": forecast_horizon,
            },
        )

        # Get sample forecasts without using pd.DataFrame methods
        sample_records = {}
        series_ids = []
        dates = []
        predictions = []
        
        # Group by series_id and get first record from each group
        for series_id in forecast_df["series_id"].unique():
            series_data = forecast_df[forecast_df["series_id"] == series_id]
            first_row = series_data.iloc[0]
            series_ids.append(series_id)
            dates.append(first_row["date"])
            predictions.append(first_row["sales_prediction"])
        
        sample_records["series_id"] = series_ids
        sample_records["date"] = dates
        sample_records["sales_prediction"] = predictions

        # Return naive forecast results
        return (
            forecast_df.to_dict(),
            forecast_plot_bytes,
            sample_records,
            forecast_horizon,
            "naive",
            html_visualization,
        )

    # Get model and training dataset from artifacts
    tft_model = model_artifacts["model"]
    training = model_artifacts["training_dataset"]

    # Get test data (most recent data)
    test_df = processed_data["test"]

    # Select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tft_model.to(device)

    # Generate future dates for forecasting
    last_date = pd.to_datetime(test_df["date"].max())
    future_dates = pd.date_range(
        start=last_date + timedelta(days=1), periods=forecast_horizon, freq="D"
    )

    # Create empty dataframe for results
    forecasts = []

    # Get unique store-item combinations
    series = test_df["series_id"].unique()

    # For each series (store-item combination), make a prediction
    for series_id in series:
        # Get the series' data
        series_data = test_df[test_df["series_id"] == series_id]

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
        # In a real application, you would have real promotion/holiday information
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
        # We need to use the same encoding as the training data
        for col in ["store", "item", "series_id", "day_of_week", "month"]:
            # Get mapping from the test data
            mapping = {
                val: idx for idx, val in enumerate(test_df[col].unique())
            }
            future_df[f"{col}_encoded"] = future_df[col].map(mapping)

        # Add time index
        # Get the highest time index from test data
        max_time_idx = test_df["time_idx"].max()
        future_df["time_idx"] = range(
            max_time_idx + 1, max_time_idx + 1 + len(future_df)
        )

        # Ensure all needed columns exist
        # This is a bit hacky, but makes sure we have all columns needed for prediction
        # In a real application, you'd have a more robust feature pipeline
        for col in test_df.columns:
            if col not in future_df.columns and col != "sales":
                # Try to get same value as the most recent data
                if col in recent_data.columns:
                    future_df[col] = recent_data[col].iloc[0]
                else:
                    # Default to 0 for missing features
                    future_df[col] = 0

        # Prepare dataset for prediction
        future_dataset = training.from_dataset(
            training, future_df, predict=True
        )
        future_dataloader = future_dataset.to_dataloader(
            train=False, batch_size=128
        )

        # Generate predictions
        predictions, x = tft_model.predict(
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

    # Create plots for visualization as ZenML artifact
    plt.figure(figsize=(15, 10))

    # Get a few random series to plot
    sample_series = np.random.choice(
        forecast_df["series_id"].unique(),
        size=min(3, len(forecast_df["series_id"].unique())),
        replace=False,
    )

    for i, series_id in enumerate(sample_series):
        # Get historical data
        historical = test_df[test_df["series_id"] == series_id].sort_values(
            "date"
        )
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

    # Instead of saving to disk, capture the plot as bytes to return as artifact
    forecast_plot_buffer = BytesIO()
    plt.savefig(forecast_plot_buffer, format="png")
    plt.close()
    forecast_plot_bytes = forecast_plot_buffer.getvalue()

    print(
        f"Generated forecasts for {len(series)} series, {forecast_horizon} days ahead"
    )

    # Create sample forecasts directly without relying on DataFrame operations
    sample_records = {}
    series_ids = []
    dates = []
    predictions = []
    
    # Group by series_id and get first record from each group
    for series_id in forecast_df["series_id"].unique():
        series_data = forecast_df[forecast_df["series_id"] == series_id]
        first_row = series_data.iloc[0]
        series_ids.append(series_id)
        dates.append(first_row["date"])
        predictions.append(first_row["sales_prediction"])
    
    sample_records["series_id"] = series_ids
    sample_records["date"] = dates
    sample_records["sales_prediction"] = predictions

    # Create HTML visualization
    html_visualization = create_forecast_visualization(
        forecast_df, test_df, sample_series, forecast_horizon, method="tft"
    )

    # Log metadata about artifacts
    log_metadata(
        metadata={
            "forecast_data_artifact_name": "forecast_data",
            "forecast_data_artifact_type": "Dict",
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
    """Create an HTML visualization of forecasting results.

    Args:
        forecast_df: DataFrame containing the forecasted values
        historical_df: DataFrame containing the historical values
        sample_series: List of series IDs to visualize
        forecast_horizon: Number of days in the forecast horizon
        method: Forecasting method used ('tft' or 'naive')

    Returns:
        HTMLString: HTML visualization of forecasting results
    """
    # Calculate summary statistics
    total_series = len(forecast_df["series_id"].unique())
    total_stores = len(forecast_df["store"].unique())
    total_items = len(forecast_df["item"].unique())

    # Get forecast start and end dates
    forecast_start = forecast_df["date"].min()
    forecast_end = forecast_df["date"].max()

    # Get average forecasted sales
    avg_forecast = forecast_df["sales_prediction"].mean()
    total_forecast = forecast_df["sales_prediction"].sum()

    # Create forecast summary stats HTML
    stats_html = f"""
    <div class="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div class="bg-blue-50 p-4 rounded-lg border border-blue-200">
            <div class="text-blue-500 text-sm font-medium">Forecast Horizon</div>
            <div class="text-2xl font-bold">{forecast_horizon} days</div>
        </div>
        <div class="bg-green-50 p-4 rounded-lg border border-green-200">
            <div class="text-green-500 text-sm font-medium">Total Series</div>
            <div class="text-2xl font-bold">{total_series}</div>
            <div class="text-xs text-gray-500">{total_stores} stores × {total_items} items</div>
        </div>
        <div class="bg-purple-50 p-4 rounded-lg border border-purple-200">
            <div class="text-purple-500 text-sm font-medium">Avg Sales Forecast</div>
            <div class="text-2xl font-bold">{avg_forecast:.1f}</div>
            <div class="text-xs text-gray-500">per store-item-day</div>
        </div>
        <div class="bg-orange-50 p-4 rounded-lg border border-orange-200">
            <div class="text-orange-500 text-sm font-medium">Total Sales Forecast</div>
            <div class="text-2xl font-bold">{total_forecast:.0f}</div>
            <div class="text-xs text-gray-500">across all series</div>
        </div>
    </div>
    """

    # Create sample series data for charts
    chart_data = []
    for i, series_id in enumerate(sample_series):
        historical = historical_df[
            historical_df["series_id"] == series_id
        ].sort_values("date")
        forecast = forecast_df[
            forecast_df["series_id"] == series_id
        ].sort_values("date")

        # Get store and item for display
        store = forecast["store"].iloc[0]
        item = forecast["item"].iloc[0]

        # Prepare data for plotting
        hist_dates = [
            d.strftime("%Y-%m-%d") for d in pd.to_datetime(historical["date"])
        ]
        forecast_dates = [
            d.strftime("%Y-%m-%d") for d in pd.to_datetime(forecast["date"])
        ]
        hist_values = historical["sales"].tolist()
        forecast_values = forecast["sales_prediction"].tolist()

        chart_data.append(
            {
                "id": i,
                "series_id": series_id,
                "store": store,
                "item": item,
                "hist_dates": hist_dates,
                "hist_values": hist_values,
                "forecast_dates": forecast_dates,
                "forecast_values": forecast_values,
            }
        )

    # Create charts HTML
    charts_html = ""
    for data in chart_data:
        charts_html += f"""
        <div class="bg-white shadow rounded-lg p-4 border border-gray-200 mb-4">
            <h3 class="text-lg font-bold mb-2">Series: {data["series_id"]} (Store: {data["store"]}, Item: {data["item"]})</h3>
            <div id="chart-{data["id"]}" style="height: 300px;"></div>
        </div>
        """

    # Create forecast by store summary
    store_summary = (
        forecast_df.groupby("store")["sales_prediction"].sum().reset_index()
    )
    store_summary = store_summary.sort_values(
        "sales_prediction", ascending=False
    )

    # Prepare store data for chart
    store_data = {
        "stores": store_summary["store"].tolist(),
        "values": store_summary["sales_prediction"].tolist(),
    }

    # Create complete HTML
    method_name = (
        "Temporal Fusion Transformer" if method == "tft" else "Naive Forecast"
    )

    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Retail Forecasting Results</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
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
        </style>
    </head>
    <body class="bg-gray-50">
        <div class="container px-4 py-8 mx-auto">
            <header class="mb-8">
                <h1 class="text-3xl font-bold mb-2">🔮 Retail Sales Forecast Dashboard</h1>
                <p class="text-gray-600">Method: {method_name} | Forecast Period: {forecast_start} to {forecast_end}</p>
            </header>
            
            {stats_html}
            
            <div class="mt-8 mb-6">
                <h2 class="text-xl font-bold mb-4">📊 Forecast by Store</h2>
                <div class="bg-white shadow rounded-lg p-4 border border-gray-200">
                    <div id="store-chart" style="height: 400px;"></div>
                </div>
            </div>
            
            <div class="mt-8">
                <h2 class="text-xl font-bold mb-4">📈 Sample Series Forecasts</h2>
                {charts_html}
            </div>
        </div>
        
        <script>
            // Store summary chart
            const storeData = {{
                x: {str(store_data["values"])},
                y: {str(store_data["stores"])},
                type: 'bar',
                orientation: 'h',
                marker: {{
                    color: 'rgba(55, 83, 109, 0.7)'
                }}
            }};
            
            Plotly.newPlot('store-chart', [storeData], {{
                margin: {{ l: 100, r: 20, t: 20, b: 30 }},
                title: 'Total Forecasted Sales by Store',
                xaxis: {{ title: 'Total Sales' }},
                yaxis: {{ automargin: true }}
            }});
            
            // Series charts
            {render_series_charts(chart_data)}
        </script>
    </body>
    </html>
    """

    return HTMLString(html)


def render_series_charts(chart_data):
    """Generate JavaScript code to render multiple series charts."""
    js_code = ""

    for data in chart_data:
        js_code += f"""
        (function() {{
            const historicalTrace = {{
                x: {str(data["hist_dates"])},
                y: {str(data["hist_values"])},
                type: 'scatter',
                mode: 'lines',
                name: 'Historical',
                line: {{ color: 'rgba(31, 119, 180, 1)' }}
            }};
            
            const forecastTrace = {{
                x: {str(data["forecast_dates"])},
                y: {str(data["forecast_values"])},
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Forecast',
                line: {{ color: 'rgba(255, 99, 132, 1)', dash: 'dot' }},
                marker: {{ size: 6 }}
            }};
            
            Plotly.newPlot('chart-{data["id"]}', [historicalTrace, forecastTrace], {{
                margin: {{ l: 50, r: 20, t: 30, b: 50 }},
                legend: {{ orientation: 'h', y: 1.1 }},
                xaxis: {{ title: 'Date' }},
                yaxis: {{ title: 'Sales' }}
            }});
        }})();
        """

    return js_code
