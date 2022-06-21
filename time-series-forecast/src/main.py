from steps.importer import bigquery_importer
from steps.preparator import preparator
from steps.transformer import transformer
from steps.trainer import trainer
from steps.evaluator import evaluator
from pipelines.time_series_pipeline import time_series_pipeline

def pipeline_run(): 
    pipeline = time_series_pipeline(
        bigquery_importer=bigquery_importer(),
        preparator=preparator(),
        transformer=transformer(),
        trainer=trainer(),
        evaluator=evaluator(),
    )
    pipeline.run() 

if __name__ == "__main__": 
     pipeline_run()
