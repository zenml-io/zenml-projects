from steps.ingest_data import ingest_data
from steps.clean_data import clean_data
from steps.model_train import train_model
from steps.evaluation import evaluation
from pipelines.training_pipeline import train_pipeline
from materializer.custom_materializer import cs_materializer
from zenml.environment import Environment
from zenml.integrations.mlflow.mlflow_environment import (
    MLFLOW_ENVIRONMENT_NAME,
)


def run_training(): 
    training = train_pipeline(
        ingest_data(),
        clean_data().with_return_materializers(cs_materializer),
        train_model(),
        evaluation(),
    )

    training.run()


if __name__ == "__main__":
    run_training()

    mlflow_env = Environment()[MLFLOW_ENVIRONMENT_NAME]
    print(
        "Now run \n "
        f"    mlflow ui --backend-store-uri {mlflow_env.tracking_uri}\n"
        "To inspect your experiment runs within the mlflow ui.\n"
        "You can find your runs tracked within the `mlflow_example_pipeline`"
        "experiment. Here you'll also be able to compare the two runs.)"
    )