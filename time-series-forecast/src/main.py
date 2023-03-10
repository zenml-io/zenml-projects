from pipelines.time_series_pipeline import time_series_pipeline
from steps.evaluator import evaluator
from steps.importer import bigquery_importer
from steps.preparator import preparator
from steps.trainer import trainer
from steps.transformer import transformer


def pipeline_run():
    pipeline = time_series_pipeline(
        bigquery_importer=bigquery_importer(),
        preparator=preparator(),
        transformer=transformer(),
        trainer=trainer(),
        evaluator=evaluator(),
    )
    pipeline.run(config_path="config.yaml")


if __name__ == "__main__":
    pipeline_run()
