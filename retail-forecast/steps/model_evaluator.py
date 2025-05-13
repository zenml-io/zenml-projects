from zenml import step, log_metadata
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from io import BytesIO, StringIO
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.metrics import SMAPE, MAE, RMSE, QuantileLoss
from typing import Dict, Any, Tuple, List
from zenml.types import HTMLString
from typing_extensions import Annotated


@step
def evaluate_model(model_artifacts: Dict[str, Any], processed_data: dict) -> Tuple[
    Annotated[float, "mae"],
    Annotated[float, "rmse"],
    Annotated[float, "smape"],
    Annotated[float, "mape"],
    Annotated[bytes, "error_plot"],
    Annotated[List[str], "worst_series"],
    Annotated[Dict[str, Any], "store_errors"],
    Annotated[Dict[str, Any], "item_errors"],
    Annotated[Dict[str, Any], "date_errors"],
    Annotated[Any, "model"],
    Annotated[HTMLString, "evaluation_visualization"]
]:
    """
    Evaluate TFT model on the test set, calculating key retail metrics:
    - MAE (Mean Absolute Error)
    - RMSE (Root Mean Squared Error)
    - SMAPE (Symmetric Mean Absolute Percentage Error)
    - MAPE (Mean Absolute Percentage Error)
    
    Also generates forecast plots for visualization and HTML visualizations for the ZenML dashboard.
    
    Returns:
        Tuple containing:
            - mae: Mean Absolute Error
            - rmse: Root Mean Squared Error
            - smape: Symmetric Mean Absolute Percentage Error
            - mape: Mean Absolute Percentage Error
            - error_plot: Visualization of largest errors
            - worst_series: List of worst performing series
            - store_errors: Error statistics by store
            - item_errors: Error statistics by item
            - date_errors: Error statistics by date
            - model: The trained model for future predictions
            - evaluation_visualization: HTML visualization of evaluation results
    """
    test_df = processed_data["test"]
    
    # Get model and training dataset directly from previous step artifacts
    tft_model = model_artifacts["model"]
    training = model_artifacts["training_dataset"]
    
    # Create test dataset using the same parameters as training
    test_dataset = training.from_dataset(training, test_df, predict=True)
    test_dataloader = test_dataset.to_dataloader(train=False, batch_size=128)
    
    # Select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tft_model.to(device)
    
    # Make predictions
    predictions, x = tft_model.predict(test_dataloader, return_x=True, trainer_kwargs={"accelerator": device})
    
    # Get actuals and calculate errors
    actuals = torch.cat([y[0] for y in iter(test_dataloader)])
    
    # Calculate metrics
    mae = MAE()(predictions, actuals)
    rmse = RMSE()(predictions, actuals)
    smape = SMAPE()(predictions, actuals)
    
    # Calculate MAPE manually (avoid division by zero)
    actuals_np = actuals.numpy()
    preds_np = predictions.numpy()
    mask = actuals_np != 0  # Avoid division by zero
    mape = np.mean(np.abs((actuals_np[mask] - preds_np[mask]) / actuals_np[mask])) * 100
    
    # Print metrics
    print(f"Test MAE: {mae:.4f}")
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test SMAPE: {smape:.4f}%")
    print(f"Test MAPE: {mape:.4f}%")
    
    # Prepare visualizations
    # Convert predictions to pandas for easier analysis
    prediction_df = test_df[["date", "store", "item", "series_id", "sales"]].copy()
    prediction_df["prediction"] = preds_np
    
    # Calculate percentage error for each prediction
    prediction_df["abs_error"] = np.abs(prediction_df["sales"] - prediction_df["prediction"])
    prediction_df["percentage_error"] = np.where(
        prediction_df["sales"] > 0,
        prediction_df["abs_error"] / prediction_df["sales"] * 100,
        np.nan
    )
    
    # Aggregate errors by various dimensions
    store_errors = prediction_df.groupby("store")["percentage_error"].mean().reset_index()
    item_errors = prediction_df.groupby("item")["percentage_error"].mean().reset_index()
    date_errors = prediction_df.groupby("date")["percentage_error"].mean().reset_index()
    
    # Find top 3 series with highest errors
    series_errors = prediction_df.groupby("series_id")["percentage_error"].mean().reset_index()
    worst_series = series_errors.nlargest(3, "percentage_error")["series_id"].tolist()
    
    # Create error plot bytes for ZenML artifact
    plt.figure(figsize=(15, 10))
    
    for i, series_id in enumerate(worst_series):
        series_data = prediction_df[prediction_df["series_id"] == series_id]
        plt.subplot(3, 1, i+1)
        plt.plot(series_data["date"], series_data["sales"], 'b-', label='Actual')
        plt.plot(series_data["date"], series_data["prediction"], 'r-', label='Prediction')
        plt.title(f"Series: {series_id} - Mean Error: {series_data['percentage_error'].mean():.2f}%")
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    
    # Instead of saving to disk, capture the plot as bytes to return as artifact
    error_plot_buffer = BytesIO()
    plt.savefig(error_plot_buffer, format='png')
    plt.close()
    error_plot_bytes = error_plot_buffer.getvalue()
    
    # Create HTML visualization for ZenML dashboard
    html_visualization = create_evaluation_visualization({
        "mae": float(mae),
        "rmse": float(rmse),
        "smape": float(smape),
        "mape": float(mape),
    }, prediction_df, worst_series)
    
    # Log metadata about the artifacts
    log_metadata(
        metadata={
            "evaluation_metrics_artifact_name": "metrics",
            "evaluation_metrics_artifact_type": "Dict[str, Any]",
            "visualization_artifact_name": "evaluation_visualization",
            "visualization_artifact_type": "zenml.types.HTMLString",
        },
    )
    
    # Prepare dictionaries with primitive types
    store_errors_dict = store_errors.to_dict()
    item_errors_dict = item_errors.to_dict()
    date_errors_dict = date_errors.to_dict() 
    
    # Return metrics and plot data as a tuple with annotated types
    return (
        float(mae),
        float(rmse),
        float(smape),
        float(mape),
        error_plot_bytes,
        worst_series,
        store_errors_dict,
        item_errors_dict,
        date_errors_dict,
        tft_model,
        html_visualization
    )


def create_evaluation_visualization(metrics: Dict[str, float], 
                                   prediction_df: pd.DataFrame,
                                   worst_series: list) -> HTMLString:
    """Create an HTML visualization of model evaluation results.
    
    Args:
        metrics: Dictionary of evaluation metrics
        prediction_df: DataFrame containing actual and predicted values
        worst_series: List of series IDs with highest prediction errors
        
    Returns:
        HTMLString: HTML visualization of evaluation results
    """
    # Create metrics table HTML
    metrics_rows = ""
    for key, value in metrics.items():
        metrics_rows += f"""
        <tr>
            <td class="p-2 border">{key.upper()}</td>
            <td class="p-2 border text-right">{value:.4f}</td>
        </tr>
        """
    
    metrics_table = f"""
    <table class="min-w-full bg-white border border-gray-300 shadow-sm">
        <thead>
            <tr class="bg-gray-100">
                <th class="p-2 border text-left">Metric</th>
                <th class="p-2 border text-right">Value</th>
            </tr>
        </thead>
        <tbody>
            {metrics_rows}
        </tbody>
    </table>
    """
    
    # Create store error chart data
    store_errors = prediction_df.groupby("store")["percentage_error"].mean().reset_index()
    store_errors = store_errors.sort_values("percentage_error", ascending=False)
    
    store_data = []
    for _, row in store_errors.iterrows():
        store_data.append({
            "label": f"Store {row['store']}",
            "value": row["percentage_error"]
        })
    
    # Create item error chart data
    item_errors = prediction_df.groupby("item")["percentage_error"].mean().reset_index()
    item_errors = item_errors.sort_values("percentage_error", ascending=False).head(10)
    
    item_data = []
    for _, row in item_errors.iterrows():
        item_data.append({
            "label": f"Item {row['item']}",
            "value": row["percentage_error"]
        })
    
    # Create HTML visualization
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Retail Forecasting Model Evaluation</title>
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
                <h1 class="text-3xl font-bold mb-2">📊 Retail Forecasting Evaluation Dashboard</h1>
            </header>
            
            <div class="bg-white shadow rounded-lg p-6 mb-6 border border-gray-200">
                <h2 class="text-xl font-bold mb-4">📈 Model Performance Metrics</h2>
                {metrics_table}
            </div>
            
            <div class="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
                <div class="bg-white shadow rounded-lg p-6 border border-gray-200">
                    <h2 class="text-xl font-bold mb-4">🏪 Error by Store</h2>
                    <div id="store-chart" style="height: 400px;"></div>
                </div>
                
                <div class="bg-white shadow rounded-lg p-6 border border-gray-200">
                    <h2 class="text-xl font-bold mb-4">📦 Error by Item (Top 10)</h2>
                    <div id="item-chart" style="height: 400px;"></div>
                </div>
            </div>
        </div>
        
        <script>
            // Store error chart
            const storeData = {{
                x: {str([d["value"] for d in store_data])},
                y: {str([d["label"] for d in store_data])},
                type: 'bar',
                orientation: 'h',
                marker: {{
                    color: 'rgba(55, 83, 109, 0.7)'
                }}
            }};
            
            Plotly.newPlot('store-chart', [storeData], {{
                margin: {{ l: 100, r: 20, t: 20, b: 30 }},
                xaxis: {{ title: 'MAPE (%)' }},
                yaxis: {{ automargin: true }}
            }});
            
            // Item error chart
            const itemData = {{
                x: {str([d["value"] for d in item_data])},
                y: {str([d["label"] for d in item_data])},
                type: 'bar',
                orientation: 'h',
                marker: {{
                    color: 'rgba(26, 118, 255, 0.7)'
                }}
            }};
            
            Plotly.newPlot('item-chart', [itemData], {{
                margin: {{ l: 100, r: 20, t: 20, b: 30 }},
                xaxis: {{ title: 'MAPE (%)' }},
                yaxis: {{ automargin: true }}
            }});
        </script>
    </body>
    </html>
    """

    return HTMLString(html)
