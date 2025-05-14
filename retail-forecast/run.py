import argparse
from typing import Optional, Dict, Any, List, Tuple
import click
from pipelines.training_pipeline import training_pipeline
from pipelines.inference_pipeline import inference_pipeline
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from zenml.client import Client


@click.command(
    help="""
RetailForecast - Retail Demand Forecasting with ZenML

Run retail demand forecasting pipelines using state-of-the-art time series models.

Examples:

  \b
  # Run the training pipeline
  python run.py --train

  \b
  # Run the inference pipeline
  python run.py --predict

  \b
  # Run both training and inference
  python run.py --train --predict

  \b
  # Run with custom parameters
  python run.py --train --forecast-horizon 28 --epochs 20
"""
)
@click.option(
    "--train",
    is_flag=True,
    default=False,
    help="Run the training pipeline",
)
@click.option(
    "--predict",
    is_flag=True,
    default=False,
    help="Run the inference pipeline",
)
@click.option(
    "--forecast-horizon",
    type=int,
    default=14,
    help="Number of days to forecast",
)
@click.option(
    "--epochs",
    type=int,
    default=10,
    help="Number of training epochs",
)
@click.option(
    "--batch-size",
    type=int,
    default=64,
    help="Batch size for training",
)
@click.option(
    "--hidden-size",
    type=int,
    default=64,
    help="Hidden size for TFT model",
)
@click.option(
    "--learning-rate",
    type=float,
    default=0.001,
    help="Learning rate for training",
)
@click.option(
    "--dropout",
    type=float,
    default=0.1,
    help="Dropout rate for the model",
)
@click.option(
    "--encoder-length",
    type=int,
    default=30,
    help="Look-back window in days",
)
@click.option(
    "--no-cache",
    is_flag=True,
    default=False,
    help="Disable caching for the pipeline run",
)
def main(
    train: bool = False,
    predict: bool = False,
    forecast_horizon: int = 14,
    epochs: int = 10,
    batch_size: int = 64,
    hidden_size: int = 64,
    learning_rate: float = 0.001,
    dropout: float = 0.1,
    encoder_length: int = 30,
    no_cache: bool = False,
):
    """Run retail forecasting pipelines with ZenML.

    Args:
        train: Whether to run the training pipeline
        predict: Whether to run the inference pipeline
        forecast_horizon: Number of days to forecast into the future
        epochs: Number of training epochs
        batch_size: Batch size for training
        hidden_size: Hidden size for TFT model
        learning_rate: Learning rate for training
        dropout: Dropout rate for the model
        encoder_length: Look-back window in days
        no_cache: Disable caching for the pipeline run
    """
    # If neither pipeline is specified, run both
    if not train and not predict:
        train = True
        predict = True

    # Set training parameters
    training_params = {
        "forecast_horizon": forecast_horizon,
        "max_epochs": epochs,
        "batch_size": batch_size,
        "hidden_size": hidden_size,
        "learning_rate": learning_rate,
        "dropout": dropout,
        "max_encoder_length": encoder_length,
    }

    # Pipeline execution options
    pipeline_options = {}
    if no_cache:
        pipeline_options["enable_cache"] = False

    model = None
    training_dataset = None

    if train:
        print("\n" + "=" * 80)
        print("Running training pipeline...")
        print("=" * 80 + "\n")

        # Run the training pipeline
        training_results = training_pipeline(**training_params)

        try:
            # Get model and training dataset from pipeline results
            model, training_dataset = training_results
            
            # Get metrics from metadata
            client = Client()
            runs = client.list_pipeline_runs(pipeline_name="retail_forecasting_training_pipeline")
            latest_run = runs[0]  # Most recent run
            
            # Extract metrics from run metadata
            if "evaluation_metrics" in latest_run.run_metadata:
                metrics = latest_run.run_metadata["evaluation_metrics"]
                print(f"\nTraining complete! Model accuracy:")
                print(f"MAE: {metrics.get('mae', 'N/A')}")
                print(f"RMSE: {metrics.get('rmse', 'N/A')}")
                print(f"SMAPE: {metrics.get('smape', 'N/A')}%")
                print(f"MAPE: {metrics.get('mape', 'N/A')}%")
            else:
                print("\nTraining complete! Model trained successfully.")
                
        except Exception as e:
            print(f"Warning: Could not extract training results: {e}")
            import traceback
            traceback.print_exc()

    if predict:
        print("\n" + "=" * 80)
        print("Running inference pipeline...")
        print("=" * 80 + "\n")

        # Configure inference parameters
        inference_params = {
            "forecast_horizon": forecast_horizon,
        }

        # Pass model and training_dataset if available
        if model is not None and training_dataset is not None:
            inference_params["model"] = model
            inference_params["training_dataset"] = training_dataset

        # Run inference pipeline
        inference_results = inference_pipeline(**inference_params)

        try:
            # Get forecast results
            forecast_data, _, _ = inference_results
            print(f"\nForecasting complete!")
            print(f"Generated forecasts for {forecast_horizon} days ahead")
        except Exception as e:
            print(f"Warning: Could not extract forecast results: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
