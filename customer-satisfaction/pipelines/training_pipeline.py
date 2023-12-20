from zenml.config import DockerSettings
from zenml.integrations.constants import MLFLOW
from zenml import pipeline
from zenml.model.model_version import ModelVersion

from steps import ingest_data, clean_data, train_model, evaluation


docker_settings = DockerSettings(required_integrations=[MLFLOW])
model_version = ModelVersion(
    name="Customer_Satisfaction_Predictor",
    description="Predictor of Customer Satisfaction.",
    delete_new_version_on_failure=True,
    tags=["classification", "customer_satisfaction"],
)


@pipeline(
    enable_cache=True,
    settings={"docker": docker_settings},
    model_version=model_version
)
def customer_satisfaction_training_pipeline(model_type: str = "lightgbm"):
    """Training Pipeline.

    Args:
        model_type: str - available options ["lightgbm", "randomforest", "xgboost"]
    """
    df = ingest_data()
    x_train, x_test, y_train, y_test = clean_data(df)
    model = train_model(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test, model_type=model_type)
    mse, rmse = evaluation(model, x_test, y_test)
