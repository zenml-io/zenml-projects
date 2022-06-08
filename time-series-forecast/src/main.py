from zenml.pipelines import pipeline
from steps.importer import bigquery_importer
from steps.preparator import preparator
from steps.transformer import transformer
from steps.trainer import trainer
from steps.evaluator import evaluator

@pipeline(enable_cache=False)
def time_series_pipeline(
    bigquery_importer,
    preparator,
    transformer,
    trainer,
    evaluator,
):
    """Defines a training pipeline to train a model to predict el. power production based on weather forecast.

    Args:
        bigquery_importer: Fetch data from BQ.
        preparator: Clean and prepare the dataset.
        transformer: Change cardinal GPS directions into vector features.
        trainer: Produce a trained prediction model.
        evaluator: Evaluate the trained model on a test set (using R2 score).
    """
    data = bigquery_importer()
    prepared_data = preparator(data=data)
    X_train, X_test, y_train, y_test = transformer(data=prepared_data)
    model = trainer(X_train=X_train,y_train=y_train)
    evaluator(X_test=X_test, y_test=y_test, model=model)

pipeline = time_series_pipeline(
    bigquery_importer=bigquery_importer(),
    preparator=preparator(),
    transformer=transformer(),
    trainer=trainer(),
    evaluator=evaluator(),
)

if __name__ == "__main__":
    pipeline.run()
