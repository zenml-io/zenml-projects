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
