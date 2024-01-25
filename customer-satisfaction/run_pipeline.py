from pipelines.training_pipeline import customer_satisfaction_training_pipeline
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri

import click


@click.command()
@click.option(
    "--model_type",
    "-m",
    type=click.Choice(["lightgbm", "randomforest", "xgboost"]),
    default="xgboost",
    help="Here you can choose what type of model should be trained."
)
def main(model_type: str):
    (
        customer_satisfaction_training_pipeline
        .with_options(config_path="config.yaml")(model_type)
    )

    print(
        "Now run \n "
        f"    mlflow ui --backend-store-uri '{get_tracking_uri()}'\n"
        "To inspect your experiment runs within the mlflow UI.\n"
        "You can find your runs tracked within the `mlflow_example_pipeline`"
        "experiment. Here you'll also be able to compare the two runs.)"
    )


if __name__ == "__main__":
    main()



