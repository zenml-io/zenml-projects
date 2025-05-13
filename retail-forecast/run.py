import argparse
from typing import Optional
import click
from pipelines.training_pipeline import training_pipeline
from pipelines.inference_pipeline import inference_pipeline


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
  python run.py --train --forecast-horizon 28 --epochs 20 --batch-size 128 --hidden-size 128
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
    }

    # Pipeline execution options
    pipeline_options = {}
    if no_cache:
        pipeline_options["enable_cache"] = False

    model_eval_results = None

    if train:
        print("\n" + "=" * 80)
        print("Running training pipeline...")
        print("=" * 80 + "\n")

        # Configure and run the training pipeline
        configured_training_pipeline = training_pipeline.with_options(
            **pipeline_options
        )
        training_results = configured_training_pipeline(**training_params)

        # Extract evaluation results
        try:
            model_eval_results = training_results.steps[
                "evaluate_model"
            ].output
            metrics = model_eval_results.read()
            print(f"\nTraining complete! Model accuracy:")
            print(f"MAE: {metrics.get('mae', 'N/A')}")
            print(f"RMSE: {metrics.get('rmse', 'N/A')}")
            print(f"MAPE: {metrics.get('mape', 'N/A')}%")
            print("Evaluation plot saved in ZenML artifacts")
        except Exception as e:
            print(f"Warning: Could not extract training metrics: {e}")

    if predict:
        print("\n" + "=" * 80)
        print("Running inference pipeline...")
        print("=" * 80 + "\n")

        # Configure and run the inference pipeline
        configured_inference_pipeline = inference_pipeline.with_options(
            **pipeline_options
        )

        inference_params = {
            "forecast_horizon": forecast_horizon,
        }

        # Only pass model_artifacts if we have them from training
        if model_eval_results is not None:
            inference_params["model_artifacts"] = model_eval_results

        inference_results = configured_inference_pipeline(**inference_params)

        print("\nForecasting complete!")
        print("Forecast results and visualizations stored in ZenML artifacts")


if __name__ == "__main__":
    main()
