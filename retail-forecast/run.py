import click
from pipelines.training_pipeline import training_pipeline


@click.command(
    help="""
RetailForecast - Simple Retail Demand Forecasting with ZenML and Prophet

Run a simplified retail demand forecasting pipeline using Facebook Prophet.

Examples:

  \b
  # Run the pipeline with default parameters
  python run.py

  \b
  # Run with custom parameters
  python run.py --forecast-periods 60 --test-size 0.3
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
def main(
    forecast_periods: int = 30,
    test_size: float = 0.2,
    weekly_seasonality: bool = True,
    no_cache: bool = False,
):
    """Run a simplified retail forecasting pipeline with ZenML.

    Args:
        forecast_periods: Number of days to forecast into the future
        test_size: Proportion of data to use for testing
        weekly_seasonality: Whether to include weekly seasonality in the model
        no_cache: Disable caching for the pipeline run
    """
    # Set pipeline parameters
    pipeline_params = {
        "forecast_periods": forecast_periods,
        "test_size": test_size,
        "weekly_seasonality": weekly_seasonality,
    }

    # Pipeline execution options
    pipeline_options = {}
    if no_cache:
        pipeline_options["enable_cache"] = False

    print("\n" + "=" * 80)
    print("Running retail forecasting pipeline...")
    print("=" * 80 + "\n")

    # Run the pipeline
    training_pipeline(**pipeline_params)

    print("\n" + "=" * 80)
    print("Pipeline completed successfully!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
