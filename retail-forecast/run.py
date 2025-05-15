import click
from pipelines.inference_pipeline import inference_pipeline
from pipelines.training_pipeline import training_pipeline
from zenml import Model


@click.command(
    help="""
RetailForecast - Simple Retail Demand Forecasting with ZenML and Prophet

Run a simplified retail demand forecasting pipeline using Facebook Prophet.

Examples:

  \b
  # Run the training pipeline with default parameters
  python run.py

  \b
  # Run with custom parameters
  python run.py --forecast-periods 60 --test-size 0.3
  
  \b
  # Run the inference pipeline
  python run.py --inference
"""
)
@click.option(
    "--forecast-periods",
    type=int,
    default=30,
    help="Number of days to forecast into the future",
)
@click.option(
    "--test-size",
    type=float,
    default=0.2,
    help="Proportion of data to use for testing",
)
@click.option(
    "--weekly-seasonality",
    type=bool,
    default=True,
    help="Whether to include weekly seasonality in the model",
)
@click.option(
    "--no-cache",
    is_flag=True,
    default=False,
    help="Disable caching for the pipeline run",
)
@click.option(
    "--inference",
    is_flag=True,
    default=False,
    help="Run the inference pipeline instead of the training pipeline",
)
def main(
    forecast_periods: int = 30,
    test_size: float = 0.2,
    weekly_seasonality: bool = True,
    no_cache: bool = False,
    inference: bool = False,
):
    """Run a simplified retail forecasting pipeline with ZenML.

    Args:
        forecast_periods: Number of days to forecast into the future
        test_size: Proportion of data to use for testing
        weekly_seasonality: Whether to include weekly seasonality in the model
        no_cache: Disable caching for the pipeline run
        inference: Run the inference pipeline instead of the training pipeline
    """
    pipeline_options = {}
    if no_cache:
        pipeline_options["enable_cache"] = False

    print("\n" + "=" * 80)
    # Run the appropriate pipeline
    if inference:
        print("Running retail forecasting inference pipeline...")

        # Create a new version of the model
        model = Model(
            name="retail_forecast_model",
            description="A retail forecast model trained on the sales data",
            version="production",
        )
        inference_pipeline.with_options(model=model, **pipeline_options)()
    else:
        # Create a new version of the model
        model = Model(
            name="retail_forecast_model",
            description="A retail forecast model trained on the sales data",
        )

        print("Running retail forecasting training pipeline...")
        training_pipeline.with_options(model=model, **pipeline_options)()
    print("=" * 80 + "\n")

    print("\n" + "=" * 80)
    print("Pipeline completed successfully!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
