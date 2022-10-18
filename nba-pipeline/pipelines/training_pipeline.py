from zenml.pipelines import pipeline
from zenml.integrations.constants import EVIDENTLY, SKLEARN, AWS, KUBEFLOW,MLFLOW
from zenml.config import DockerSettings


docker_settings = DockerSettings(required_integrations=[EVIDENTLY, SKLEARN, AWS, KUBEFLOW, MLFLOW])

@pipeline(enable_cache=False, settings={"docker": docker_settings})
def training_pipeline(
    importer,
    feature_engineer,
    encoder,
    ml_splitter,
    trainer,
    tester,
    drift_splitter,
    drift_detector,
    drift_alert,
):
    """Defines a training pipeline to train a model on NBA data.

    Args:
        importer: Import step to query data.
        feature_engineer: Creates relevant features for training.
        encoder: Encode strings to integers.
        ml_splitter: Split data into train, test, and validation.
        trainer: Produce a trained prediction model.
        tester: Test trained model on test set.
        drift_splitter: Split data for drift alerts.
        drift_detector: Calculate drift.
        drift_alert: Post drift alert on Discord.
    """
    # Training pipeline
    raw_data = importer()
    transformed_data = feature_engineer(raw_data)
    encoded_data, le_seasons, ohe_teams = encoder(transformed_data)
    (
        train_df_x,
        train_df_y,
        test_df_x,
        test_df_y,
        eval_df_x,
        eval_df_y,
    ) = ml_splitter(encoded_data)
    model = trainer(train_df_x, train_df_y, eval_df_x, eval_df_y)
    test_results = tester(model, test_df_x, test_df_y)

    # drift
    reference_dataset, comparison_dataset = drift_splitter(raw_data)
    drift_report, _ = drift_detector(reference_dataset, comparison_dataset)
    drift_alert(drift_report)
