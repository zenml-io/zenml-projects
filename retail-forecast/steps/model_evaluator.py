from zenml import step, log_metadata
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from io import BytesIO
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import SMAPE, MAE, RMSE
from typing import Tuple
from zenml.types import HTMLString
from typing_extensions import Annotated


@step
def evaluate_model(
    model: TemporalFusionTransformer, 
    training_dataset: TimeSeriesDataSet, 
    test_data: pd.DataFrame
) -> Tuple[
    Annotated[float, "mae"],
    Annotated[float, "rmse"],
    Annotated[float, "smape"],
    Annotated[float, "mape"],
    Annotated[HTMLString, "evaluation_visualization"]
]:
    """
    Evaluate TFT model on the test set, calculating key retail metrics
    
    Args:
        model: Trained TFT model
        training_dataset: Training dataset used for the model
        test_data: Test dataframe for evaluation

    Returns:
        mae: Mean Absolute Error
        rmse: Root Mean Squared Error
        smape: Symmetric Mean Absolute Percentage Error
        mape: Mean Absolute Percentage Error
        evaluation_visualization: HTML visualization of results
    """
    # Create test dataset using the same parameters as training
    test_dataset = training_dataset.from_dataset(training_dataset, test_data, predict=True)
    test_dataloader = test_dataset.to_dataloader(train=False, batch_size=128)
    
    # Select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Make predictions
    predictions, _ = model.predict(test_dataloader, return_x=True, trainer_kwargs={"accelerator": device})
    
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
    prediction_df = test_data[["date", "store", "item", "series_id", "sales"]].copy()
    prediction_df["prediction"] = preds_np
    
    # Calculate percentage error for each prediction
    prediction_df["abs_error"] = np.abs(prediction_df["sales"] - prediction_df["prediction"])
    prediction_df["percentage_error"] = np.where(
        prediction_df["sales"] > 0,
        prediction_df["abs_error"] / prediction_df["sales"] * 100,
        np.nan
    )
    
    # Find series with highest errors
    series_errors = prediction_df.groupby("series_id")["percentage_error"].mean().reset_index()
    worst_series = series_errors.nlargest(3, "percentage_error")["series_id"].tolist()
    
    # Prepare metrics for logging
    evaluation_metrics = {
        "mae": float(mae),
        "rmse": float(rmse),
        "smape": float(smape),
        "mape": float(mape)
    }
    
    # Log detailed metrics about the evaluation
    log_metadata(metadata=evaluation_metrics, infer_model=True)
    
    # Log store-level and item-level performance
    store_errors = prediction_df.groupby("store")["percentage_error"].mean().to_dict()
    item_errors = prediction_df.groupby("item")["percentage_error"].mean().to_dict()
    
    # Log detailed metrics
    log_metadata(metadata={
        "evaluation_metrics": evaluation_metrics,
        "store_level_errors": store_errors,
        "item_level_errors": item_errors,
        "worst_performing_series": worst_series
    })
    
    # Create HTML visualization for ZenML dashboard
    html_visualization = create_evaluation_visualization(
        evaluation_metrics, prediction_df, worst_series
    )
    
    # Return the key metrics and visualization
    return float(mae), float(rmse), float(smape), float(mape), html_visualization


def create_evaluation_visualization(metrics: dict, prediction_df: pd.DataFrame, worst_series: list) -> HTMLString:
    """Create an HTML visualization of model evaluation results."""
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
    </head>
    <body class="bg-gray-50 p-4">
        <div class="container mx-auto">
            <h1 class="text-2xl font-bold mb-4">Retail Forecasting Evaluation</h1>
            
            <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div class="bg-white p-4 rounded shadow">
                    <h2 class="text-lg font-semibold mb-2">Performance Metrics</h2>
                    {metrics_table}
                </div>
                
                <div class="bg-white p-4 rounded shadow">
                    <h2 class="text-lg font-semibold mb-2">Store Error Analysis</h2>
                    <div id="store-chart" style="height: 300px;"></div>
                </div>
            </div>
        </div>
        
        <script>
            // Store chart
            const storeData = {{
                x: {[d["value"] for d in store_data]},
                y: {[d["label"] for d in store_data]},
                type: 'bar',
                orientation: 'h',
                marker: {{ color: 'rgba(55, 83, 109, 0.7)' }}
            }};
            
            Plotly.newPlot('store-chart', [storeData], {{
                margin: {{ l: 100, r: 20, t: 20, b: 30 }},
                xaxis: {{ title: 'MAPE (%)' }}
            }});
        </script>
    </body>
    </html>
    """

    return HTMLString(html)
